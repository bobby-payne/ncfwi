import xarray as xr
import os

from config import get_config


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


def load_wx_data() -> xr.Dataset:
    """
    Open each wx data (assumed netcdf) separately and then
    merge them into a single xarray dataset.

    Returns
    -------
    xarray.Dataset
        The merged wx dataset.

    """

    # Get paths (as strings) to data
    wx_data_paths = get_paths_to_wx_data()
    wx_var_names = wx_data_paths.keys()

    # Open the data and temporarily store in a dictionary
    wx_data = {}
    for wx_var in wx_var_names:
        print("Opening data:", wx_var)
        path = wx_data_paths[wx_var]
        wx_data[wx_var] = xr.open_mfdataset(path)

    # Merge into an xarray.Dataset
    wx_data_xarray = xr.merge(wx_data.values(), join="exact")

    # Add all dimensions as coordinates, provided they aren't already one
    for dim in wx_data_xarray.dims:
        if dim not in wx_data_xarray.coords:
            wx_data_xarray = wx_data_xarray.assign_coords({dim: wx_data_xarray[dim]})

    return wx_data_xarray


def save_to_netcdf(dataset: xr.Dataset, filename: str) -> None:
    """
    Saves an xarray dataset to a netCDF file at the location
    specified by the user in the config file.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset to save.
    filename : str
        The name of the file to save the dataset as (WITHOUT path).
        The full path will be constructed using the output directory
        specified in the config file.

    """

    # Get the path to save the data
    config = get_config()
    path_out = config["settings"]["output_dir"]

    # Save the dataset to a netCDF file
    full_path = os.path.join(path_out, filename)
    dataset.to_netcdf(full_path)
