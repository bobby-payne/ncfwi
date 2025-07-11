import sys
import gc
import xarray as xr
import numpy as np
import pytz
import time
from datetime import datetime
from timezonefinder import TimezoneFinder
from joblib import Parallel, delayed
from tqdm import tqdm
from itertools import product
from dask.diagnostics import ProgressBar

from formatting import *
from readwrite import *
from season import *
from overwinter import *
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
    output_dir = config["settings"]["output_dir"]
    overwinter = config["settings"]["overwinter"]
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

    # Overwinter the drought code (DC)
    if overwinter and not (year == start_year):

        lastyear_DC = xr.open_dataset(output_dir + f"/DC/{year-1}.nc")
        lastyear_DC = lastyear_DC.sel({x_dim_name: [x], y_dim_name: [y]})
        lastyear_DC = lastyear_DC.where(~np.isnan(lastyear_DC['DC']),drop=True)
        lastyear_DC_fin = lastyear_DC['DC'].isel(time=-1).values.squeeze()

        lastyear_postfs_precip_accum = xr.open_dataset(output_dir + f"/PFS_PREC/{year-1}.nc")
        lastyear_postfs_precip_accum = lastyear_postfs_precip_accum.sel({x_dim_name: [x], y_dim_name: [y]})
        lastyear_postfs_precip_accum = lastyear_postfs_precip_accum['PFS_PREC'].values.squeeze()

        thisyear_prefs_precip_accum = get_prefs_precip_accum_for_grid_point(
            wx_data_ixy, fire_season_mask_ixy
        )

        net_precip_accum = lastyear_postfs_precip_accum + thisyear_prefs_precip_accum

        DC_startup = overwinter_DC(
            lastyear_DC_fin,
            net_precip_accum,
        )

    else:
        DC_startup = DC_DEFAULT

    # If fire season is active for at least one time step, calculate the FWIs
    # else, return a DataFrame of numpy NaNs
    if any(fire_season_mask_ixy):
        FWI_dataframe_ixy = hFWI(
            wx_dataframe_masked_ixy,
            UTC_offset,
            ffmc_old=FFMC_DEFAULT,
            dmc_old=DMC_DEFAULT,
            dc_old=DC_startup,
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

    # Add this year's post-fire season accumulated precip to FWI dataset
    # (needed for overwintering the DC next year)
    if overwinter:

        thisyear_postfs_precip_accum = get_postfs_precip_accum_for_grid_point(
            wx_data_ixy, fire_season_mask_ixy
        )

        FWI_dataset_ixy["PFS_PREC"] = xr.DataArray(
            np.array([[thisyear_postfs_precip_accum]]),  # shape (1, 1)
            dims=(y_dim_name, x_dim_name),
            coords={
                y_dim_name: wx_data_ixy[y_dim_name],
                x_dim_name: wx_data_ixy[x_dim_name]
            },
            name="PFS_PREC"
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
    crop_x_index = config["settings"]["crop_x_index"]
    start_year = config["settings"]["start_year"]
    end_year = config["settings"]["end_year"]
    path_to_cffdrs_ng = config["settings"]["path_to_cffdrs-ng"]
    parallel = config["settings"]["parallel"]
    n_cores = config["settings"]["n_cpu_cores"]
    save_in_batches = config["settings"]["output_in_batches"]
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

        print("Transpose dims and apply transformations...")
        wx_data_i = transpose_dims(wx_data_i)
        wx_data_i = apply_transformations(wx_data_i)

        print("Computing indices...")
        if parallel:  # Compute the FWIs at each grid point in parallel

            coordinate_tuples = list(product(wx_data_i[x_dim_name].values,
                                             wx_data_i[y_dim_name].values))
            with Parallel(n_jobs=n_cores) as joblib_parallel:
                FWIs_list = joblib_parallel(
                    delayed(compute_FWIs_for_grid_point)(wx_data_i, loc_index, year)
                    for loc_index in tqdm(coordinate_tuples)
                )

        else:  # Compute the FWIs at grid point by grid point in series

            FWIs_list = []
            for x in wx_data_i[x_dim_name].values:
                for y in wx_data_i[y_dim_name].values:

                    FWIs_at_xy = compute_FWIs_for_grid_point(wx_data_i, (x, y), year)
                    FWIs_list.append(FWIs_at_xy)

        if save_in_batches:

            batch_size = (crop_x_index[1] - crop_x_index[0]) * 20
            for i in tqdm(range(0, len(FWIs_list), batch_size)):
                batch = FWIs_list[i:i + batch_size]
                FWIs_batch_dataset = xr.combine_by_coords(batch)
                print(f"Saving batch of size {FWIs_batch_dataset.nbytes / 1e6:.2f} MB")
                print(f"{i}, {i+batch_size}, {len(FWIs_list)}, {i // batch_size + 1}")
                save_to_netcdf(FWIs_batch_dataset, year=year, file_suffix=f"_{i // batch_size + 1}")
                combine_batched_files(year)

        else:

            print("Combining data... (may take a while)")
            FWIs_dataset = xr.combine_by_coords(FWIs_list)

            print(f"Saving FWI data for year {year} of size {FWIs_dataset.nbytes / 1e6:.2f} MB...")
            save_to_netcdf(FWIs_dataset, year=year)

        gc.collect()

    end_time = time.time()
    n_minutes, n_seconds = divmod(end_time - start_time, 60)
    print(f"Finished in {int(n_minutes)} minutes and {n_seconds:.2f} seconds.")
