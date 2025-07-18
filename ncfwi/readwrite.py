import xarray as xr
import pandas as pd
import numpy as np
import os
from glob import glob

from config import get_config
from formatting import *


def get_paths_to_wx_data() -> dict:
    """
    Get paths to data from the configuration file.
    
    Returns:
        dict: Dictionary containing data paths.
    """
    config = get_config()
    path_dictionary = {
        "wind_speed": config["data_vars"]["wind_speed"]["path"],
        "air_temperature": config["data_vars"]["air_temperature"]["path"],
        "relative_humidity": config["data_vars"]["relative_humidity"]["path"],
        "precipitation": config["data_vars"]["precipitation"]["path"],
    }

    return path_dictionary


def preprocess_data(wx_data: xr.Dataset) -> xr.Dataset:
    """
    A preprocessing function for xarray.open_mfdataset to only select data
    for the time range specified in the configuration file. This does two things:
    (a) ensures that the time dim for each wx variable is the same so that they
    can be merged, and (b) speeds up open_mfdataset. Only meant to be used as an 
    argument to `xarray.open_mfdataset`'s `preprocess` parameter.

    Parameters
    ----------
    wx_data : xarray.Dataset
        An xarray dataset with a time dimension as specified in the config.

    Returns
    -------
    xarray.Dataset | NoneType
        The dataset filtered to only include data from the specified year.
        If no data is found for the specified time range, returns None.
    """

    config = get_config()
    time_dim_name = config["data_vars"]["t_dim_name"]
    start_year = config["settings"]["start_year"]
    end_year = config["settings"]["end_year"]
    data_timerange = pd.to_datetime(wx_data[time_dim_name].values)

    wx_data = rename_coordinates(wx_data)
    wx_data = rename_wx_variables(wx_data)
    wx_data = apply_spatial_crop(wx_data)

    if not any((start_year <= t.year <= end_year) for t in data_timerange):
        wx_data = wx_data.isel({time_dim_name: slice(0, 0)})
    else:
        wx_data = wx_data.sel({
            time_dim_name: slice(f"{start_year}-01-01", f"{end_year}-12-31")
            })

    return wx_data


def load_wx_data() -> xr.Dataset:
    """
    Open each wx data (assumed netcdf) separately and then
    merge them into a single xarray dataset.

    Parameters
    ----------
    preprocessing_function : Callable, optional
        A function passed on to `xarray.open_mfdataset` on how to
        preprocess the datasets in the data directory before merging.

    Returns
    -------
    xarray.Dataset
        The merged wx dataset.

    """

    # Get paths (as strings) to data
    config = get_config()
    x_dim_name = config["data_vars"]["x_dim_name"]
    y_dim_name = config["data_vars"]["y_dim_name"]
    x0, x1 = config["settings"]["crop_x_index"]
    y0, y1 = config["settings"]["crop_y_index"]
    wx_data_paths = get_paths_to_wx_data()
    wx_var_names = wx_data_paths.keys()

    # Open the data and temporarily store in a dictionary
    wx_data = {}
    for wx_var in wx_var_names:
        print("Opening data:", wx_var)
        path = wx_data_paths[wx_var]
        wx_data[wx_var] = xr.open_mfdataset(
            path,
            preprocess=preprocess_data,
            combine="nested",
            concat_dim=get_config()["data_vars"]["t_dim_name"],
            chunks={get_config()["data_vars"]["t_dim_name"]: 4380},
            )

    # Merge into an xarray.Dataset
    wx_data_xarray = xr.merge(wx_data.values(), join="inner")

    # Add spatial dimensions as coordinates, provided they aren't already one
    for dim in wx_data_xarray.dims:
        if (dim not in wx_data_xarray.coords) and (dim == x_dim_name):
            wx_data_xarray = wx_data_xarray.assign_coords({dim: np.arange(x0, x1)})
        elif (dim not in wx_data_xarray.coords) and (dim == y_dim_name):
            wx_data_xarray = wx_data_xarray.assign_coords({dim: np.arange(y0, y1)})

    return wx_data_xarray


def save_to_netcdf(dataset: xr.Dataset, year: int, file_suffix: str | None = None) -> None:
    """
    Saves an xarray dataset to a netCDF file at the location
    specified by the user in the config file.
    The filename will be in the format "{year}{file_suffix}.nc".

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset to save.
    year : int
        The year to use in the filename.
    file_suffix : str
        A string to append to the end of the filename,
        NOT including the file extension.
        If None, no suffix is added.
    """

    # Get the path to save the data
    config = get_config()
    path_out = config["settings"]["output_dir"]

    # Save each variable in its own folder
    for var_name in dataset.data_vars:

        # Create the output directory for the variable if it doesn't exist
        output_dir = os.path.join(path_out, str(var_name))
        os.makedirs(output_dir, exist_ok=True)

        # Select data corresponding to variable to be saved and then save
        var_data = dataset[var_name]
        var_data.to_netcdf(os.path.join(output_dir, f"{year}{file_suffix}.nc"))


def combine_batched_files(year: int) -> None:
    """
    Combines batched netCDF files into a single file for each variable.
    The files are expected to be in the output directory specified in the config.

    Parameters
    ----------
    year : int
        The year for which the files are to be combined.
        The files should be named in the format "{year}_{batch#}.nc".
    """

    config = get_config()
    path_out = config["settings"]["output_dir"]
    output_vars = config["settings"]["output_vars"]
    output_vars = np.append(output_vars, ["PFS_PREC"])

    for var_name in output_vars:
        var_path = os.path.join(path_out, str(var_name))
        files = sorted(glob(os.path.join(var_path, f"{year}_*.nc")))
        dataset = xr.open_mfdataset(files, combine="by_coords")
        dataset.to_netcdf(os.path.join(var_path, f"{year}.nc"))
        for f in files:  # Remove the individual batch files
            os.remove(f)
