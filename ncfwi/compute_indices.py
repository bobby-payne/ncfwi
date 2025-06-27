import sys
import gc
import pandas as pd
import xarray as xr
import numpy as np
import pytz
import tqdm
import time
from datetime import datetime
from timezonefinder import TimezoneFinder
from joblib import Parallel, delayed
from itertools import product
from dask.diagnostics import ProgressBar

from formatting import *
from readwrite import *
from season import *
from config import get_config


def preprocess_data(wx_data) -> xr.Dataset:
    """
    Initialize by loading the configuration and data,
    and performing some preprocessing steps.
    This function should be called near/at the start of the script.

    Parameters:
        wx_data (xarray.Dataset): The raw weather data to preprocess.

    Returns:
        xarray.Dataset
            The preprocessed dataset.
    """

    # Do a few preprocessing steps
    wx_data = transpose_dims(wx_data)
    wx_data = rename_coordinates(wx_data)
    wx_data = rename_wx_variables(wx_data)
    wx_data = apply_transformations(wx_data)

    return wx_data


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

    # Select the data for the given grid point
    x, y = loc_index
    wx_data_ixy = wx_data_i.sel({x_dim_name: [x], y_dim_name: [y]})
    
    # Obtain the fire season mask for this year in the form of an np array
    fire_season_mask_ixy = compute_fire_season(
        wx_data_ixy, return_as_xarray=True
    )['fire_season_mask'].values

    # convert the wx data to the pandas DataFrame needed for hFWI fn.
    wx_dataframe_ixy = xarray_to_pandas_dataframe(wx_data_ixy.squeeze())

    # Get the UTC offset for the current grid point
    lon = wx_dataframe_ixy["long"].values[0]
    lat = wx_dataframe_ixy["lat"].values[0]
    UTC_offset = get_timezone_UTC_offset(lat, lon)

    # Apply the fire season mask to the data at this grid point
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
        FWI_dataframe_ixy = get_empty_hFWI_dataframe(year, lat, lon, UTC_offset)

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

    start_time = time.time()

    # Open and read config
    print("Getting ready...")
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

    # Load (lazily) data
    wx_data = load_wx_data()

    # Main computation loop
    for year in time_range:

        # Load data for the current year into memory
        print(f"Loading data for year {year} into memory...")
        with ProgressBar():
            wx_data_i = wx_data.sel({t_dim_name: str(year)}).compute()

        print("Preprocessing data...")
        wx_data_i = preprocess_data(wx_data_i)

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

        print(f"Combining FWI data for year {year}...")
        FWIs_dataset = xr.combine_by_coords(FWIs_list, join='outer')
        print(f"Saving FWI data for year {year}...")
        save_to_netcdf(FWIs_dataset)
        gc.collect()
    
    end_time = time.time()
    n_minutes, n_seconds = divmod(end_time - start_time, 60)
    print(f"Finished in {int(n_minutes)} minutes and {n_seconds:.2f} seconds.")
