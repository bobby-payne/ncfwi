import sys
import gc
import pandas as pd
import xarray as xr
import numpy as np
import pytz
import tqdm
from timezonefinder import TimezoneFinder
from typing import Union
from types import NoneType
from joblib import Parallel, delayed
from itertools import product
from formatting import *
from readwrite import *
from season import *


def load_and_preprocess_data() -> xr.Dataset:
    """
    Initialize by loading the configuration and data,
    and performing some preprocessing steps.
    This function should be called near/at the start of the script.

    Returns:
        data (xarray.Dataset): The preprocessed dataset ready for FWI calculation.
    """

    # Load in the config and data
    data = load_data()

    # Do a few preprocessing steps
    data = transpose_dims(data)
    data = rename_coordinates(data)
    data = rename_wx_variables(data)
    data = apply_transformations(data)
    data = apply_spatial_crop(data)

    return data


def get_timezone_UTC_offset(lat: float, lon: float) -> float:
    """
    Get the timezone offset from UTC at a given latitude and longitude.

    Parameters
    ----------
    lat : float
        Latitude of the location in (-90.0, 90.0).
    lon : float
        Longitude of the location in (-180.0, 180.0).

    Returns
    -------
    float
        The UTC offset in hours for the timezone at the given coordinates.
        e.g., -3.5 (for Newfoundland, Canada)
    """

    tz_str = TimezoneFinder().timezone_at(lng=lon, lat=lat)
    if tz_str is None:
        raise ValueError(f"Could not determine timezone for coordinates: lat={lat}, lon={lon}")
    tz = pytz.timezone(tz_str)
    tz_offset = tz.localize(datetime(2024, 1, 1)).utcoffset()
    return tz_offset.total_seconds() / 3600


def xarray_to_pandas_dataframe(wx_data: xr.Dataset) -> pd.DataFrame:
    """
    Turns the wx data for a single year into a pandas DataFrame, and formats it
    to be compatible with the ng-cffdrs hFWI function.

    Args:
        data (xarray.Dataset): The wx data for a single year.

    Returns:
        wx_dataframe (pandas.DataFrame): The wx data formatted as a pandas
        DataFrame, ready for FWI calculation.
    """

    wx_dataframe = wx_data.to_dataframe()
    wx_dataframe.index = pd.to_datetime(wx_dataframe.index)
    wx_dataframe['YR'] = wx_dataframe.index.year.astype('int')
    wx_dataframe["MON"] = wx_dataframe.index.month.astype('int')
    wx_dataframe["DAY"] = wx_dataframe.index.day.astype('int')
    wx_dataframe["HR"] = wx_dataframe.index.hour.astype('int')
    wx_dataframe = wx_dataframe.reset_index()

    return wx_dataframe


def hFWI_output_to_xarray_dataset(hFWI_dataframe: pd.DataFrame,
                                  season_mask: Union[NoneType, np.ndarray],
                                  dataset_coords: dict) -> xr.Dataset:
    """
    Converts the output of the cffdrs-ng hFWI function (pandas dataframe)
    to an xarray Dataset.

    Args:
        hFWI_dataframe : pandas.DataFrame
            The output from the hFWI function.
        season_mask : Union[NoneType, np.ndarray]
            The fire season mask for the year, computed from 
            season.py's compute_fire_season(return_as_xarray=False).
            If None, then the mask is not included as a variable in the output.
        dataset_coords : dict
            A dictionary containing the coordinates to assign to
            the xarray Dataset at its creation. Time coordinate must
            align with the MASKED data, (i.e., not the entire year.)

    Returns:
        fwi_dataset : xarray.Dataset
            The FWI data formatted as an xarray Dataset.
            Also includes the fire season mask if provided.
    """

    config = get_config()
    t_dim_name = config["data_vars"]["t_dim_name"]
    x_dim_name = config["data_vars"]["x_dim_name"]
    y_dim_name = config["data_vars"]["y_dim_name"]
    output_vars = config["settings"]["output_vars"]
    dims_list = [t_dim_name, x_dim_name, y_dim_name]

    # Reindex the hFWI dataframe in its time dimension such that
    # there is an entry for every hour of the year (fill_val=nan).
    hFWI_dataframe = hFWI_dataframe.set_index('timestamp')
    year = hFWI_dataframe.index[0].year
    full_index = pd.date_range(f'{year}-01-01', f'{year}-12-31 23:00', freq='h')
    hFWI_dataframe = hFWI_dataframe.reindex(full_index)

    # Select the desired output variables and
    # convert the pandas dataframe to an xarray Dataset
    # Note that mask is handled separately (see below)
    output_vars_no_mask = [var for var in output_vars if var.lower() != 'season_mask']
    hFWI_dataset = xr.Dataset(
        {
            var: (
                dims_list,
                hFWI_dataframe[var.lower()].to_numpy().reshape(-1, 1, 1)
            )
            for var in output_vars_no_mask
        },
        coords=dataset_coords
    )

    # The season_mask variable is handled differently
    if 'season_mask' in output_vars:
        season_mask_dataset = xr.Dataset(
            {
                'MASK': (dims_list, season_mask.reshape(-1, 1, 1)),
            },
            coords=dataset_coords
        )
        hFWI_dataset = xr.merge([season_mask_dataset, hFWI_dataset,], join='outer')

    return hFWI_dataset


def compute_FWIs_for_grid_point(wx_data_i: xr.Dataset,
                                loc_index: tuple[int, int],
                                year: int) -> xr.Dataset:
    """
    Compute the Fire Weather Index (FWI) for a specific grid point
    in the wx data for a given year.

    Args:
        wx_data (xr.Dataset): 1-year timeseries of weather data for the
            whole domain.
        loc_index (tuple[int,int]): The x and y index of the grid point,
            in that order, i.e., (x_index, y_index).
        year (int): The year for which to compute the FWI.
    Returns:
        xr.Dataset: The FWI data for the specified grid point and year
    """

    # Load and read config
    config = get_config()
    t_dim_name = config["data_vars"]["t_dim_name"]
    x_dim_name = config["data_vars"]["x_dim_name"]
    y_dim_name = config["data_vars"]["y_dim_name"]
    FFMC_DEFAULT = config["calculation_parameters"]["FFMC_default"]
    DMC_DEFAULT = config["calculation_parameters"]["DMC_default"]
    DC_DEFAULT = config["calculation_parameters"]["DC_default"]

    # Obtain the fire season mask for this year
    fire_season_mask_i = compute_fire_season(wx_data_i, return_as_xarray=True)

    # Select the data for the given grid point
    # then convert to the pandas DataFrame needed for hFWI
    x, y = loc_index
    wx_dataframe_ixy = xarray_to_pandas_dataframe(
        wx_data_i.sel({x_dim_name: x, y_dim_name: y})
    )

    # Get the UTC offset for the current grid point
    lon = wx_dataframe_ixy["long"].values[0]
    lat = wx_dataframe_ixy["lat"].values[0]
    UTC_offset = get_timezone_UTC_offset(lat, lon)

    # Apply the fire season mask to the data at this grid point
    fire_season_mask_ixy = fire_season_mask_i.sel(
        {x_dim_name: x, y_dim_name: y}
    )['fire_season_mask'].values
    wx_dataframe_masked_ixy = wx_dataframe_ixy[fire_season_mask_ixy]

    # If fire season is active for at least one time step, calculate the FWIs
    # else, return a DataFrame of numpy NaNs
    if any(fire_season_mask_ixy):
        FWI_dataframe_ixy = hFWI(
            wx_dataframe_masked_ixy,
            UTC_offset,
            ffmc_old=FFMC_DEFAULT,
            dmc_old=DMC_DEFAULT,
            dc_old=DC_DEFAULT,
            )
    else:
        FWI_dataframe_ixy = pd.DataFrame({
            'fwi': [np.nan] * len(fire_season_mask_ixy),
            'bui': [np.nan] * len(fire_season_mask_ixy),
            'isi': [np.nan] * len(fire_season_mask_ixy),
            'ffmc': [np.nan] * len(fire_season_mask_ixy),
            'dmc': [np.nan] * len(fire_season_mask_ixy),
            'dc': [np.nan] * len(fire_season_mask_ixy),
            'temp': [np.nan] * len(fire_season_mask_ixy),
        })
    # print(f"FWI calculation for ({x}, {y}) took {end_time - start_time:.2f}s")

    # Convert the output pandas dataframe into an xarray Dataset
    # IMPORTANT: The provided time coordinate in dataset_coords must line up
    # with the data that have been MASKED, not for the entire year.
    dataset_coords = {
            t_dim_name: wx_dataframe_ixy[t_dim_name].values,  # array of datetimes
            x_dim_name: [x],
            y_dim_name: [y],
            'lat': ([y_dim_name, x_dim_name], np.array([[lat]])),
            'lon': ([y_dim_name, x_dim_name], np.array([[lon]])),
            }
    FWI_dataset_ixy = hFWI_output_to_xarray_dataset(
        FWI_dataframe_ixy,
        fire_season_mask_ixy,
        dataset_coords,
    )

    return FWI_dataset_ixy


if __name__ == "__main__":

    # Open and read config
    print("Initializing...")
    config = get_config()
    t_dim_name = config["data_vars"]["t_dim_name"]
    x_dim_name = config["data_vars"]["x_dim_name"]
    y_dim_name = config["data_vars"]["y_dim_name"]
    start_year = config["settings"]["start_year"]
    end_year = config["settings"]["end_year"]
    path_to_cffdrs_ng = config["settings"]["path_to_cffdrs-ng"]
    parallel = config["settings"]["parallel"]
    n_cores = config["settings"]["n_cpu_cores"]
    time_range = np.arange(start_year, end_year + 1)
    if path_to_cffdrs_ng not in sys.path:
        sys.path.append(path_to_cffdrs_ng)
    from NG_FWI import hFWI

    # Initialize data
    print("Preprocessing data...")
    wx_data = load_and_preprocess_data()

    # Main computation loop
    for year in time_range:

        # Load data for the current year into memory
        print(f"Loading data for year {year} into memory...")
        print("(This may take a while!)")
        # wx_data_i = wx_data.sel({t_dim_name: str(year)}).compute()

        wx_data_i = xr.open_dataset(f"/users/rpayne/wx_data_{year}.nc")

        if parallel:  # Compute the FWIs at each grid point in parallel

            coordinate_tuples = list(product(wx_data_i[x_dim_name].values,
                                             wx_data_i[y_dim_name].values))
            FWIs_list = Parallel(n_jobs=n_cores)(
                delayed(compute_FWIs_for_grid_point)(wx_data_i, loc_index, year)
                for loc_index in tqdm.tqdm(coordinate_tuples)
            )

        else:  # Compute the FWIs at grid point by grid point in series

            FWIs_list = []
            for x in wx_data_i[x_dim_name].values:
                for y in wx_data_i[y_dim_name].values:

                    print(f"Computing FWI for (x={x}, y={y})...")
                    FWIs_at_xy = compute_FWIs_for_grid_point(wx_data_i, (x, y), year)
                    FWIs_list.append(FWIs_at_xy)

        FWIs_dataset = xr.merge(FWIs_list, join='outer')
        print(FWIs_dataset)

    gc.collect()

