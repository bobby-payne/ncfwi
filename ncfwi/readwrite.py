import xarray as xr
import pandas as pd
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


def crop_before_merging(wx_data: xr.Dataset) -> xr.Dataset | None:
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
    x_dim = config["data_vars"]["x_dim_name"]
    y_dim = config["data_vars"]["y_dim_name"]
    crop_x_index = config["settings"]["crop_x_index"]
    crop_y_index = config["settings"]["crop_y_index"]
    time_dim_name = config["data_vars"]["t_dim_name"]
    start_year = config["settings"]["start_year"]
    end_year = config["settings"]["end_year"]
    data_timerange = pd.to_datetime(wx_data[time_dim_name].values)

    if not any((start_year <= t.year <= end_year) for t in data_timerange):
        wx_data = wx_data.isel({time_dim_name: slice(0, 0)})
    else:
        wx_data = wx_data.sel({
            time_dim_name: slice(f"{start_year}-01-01", f"{end_year}-12-31")
            })
    if crop_x_index:
        x0, x1 = crop_x_index
        if x0 > x1:
            raise ValueError(f"crop_x_index {crop_x_index} is invalid: x0 must be less than or equal to x1.")
        wx_data = wx_data.isel(
            {x_dim: slice(x0, x1 + 1)}
        )
    if crop_y_index:
        y0, y1 = crop_y_index
        if y0 > y1:
            raise ValueError(f"crop_y_index {crop_y_index} is invalid: y0 must be less than or equal to y1.")
        wx_data = wx_data.isel(
            {y_dim: slice(y0, y1 + 1)}
        )

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
    wx_data_paths = get_paths_to_wx_data()
    wx_var_names = wx_data_paths.keys()

    # Open the data and temporarily store in a dictionary
    wx_data = {}
    for wx_var in wx_var_names:
        print("Opening data:", wx_var)
        path = wx_data_paths[wx_var]
        wx_data[wx_var] = xr.open_mfdataset(
            path,
            preprocess=crop_before_merging,
            combine="nested",
            concat_dim=get_config()["data_vars"]["t_dim_name"],
            chunks={get_config()["data_vars"]["t_dim_name"]: 4380},
            )

    # Merge into an xarray.Dataset
    wx_data_xarray = xr.merge(wx_data.values(), join="exact")

    # Add all dimensions as coordinates, provided they aren't already one
    for dim in wx_data_xarray.dims:
        if dim not in wx_data_xarray.coords:
            wx_data_xarray = wx_data_xarray.assign_coords({dim: wx_data_xarray[dim]})

    return wx_data_xarray


def save_to_netcdf(dataset: xr.Dataset) -> None:
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
    t_dim_name = config["data_vars"]["t_dim_name"]

    # Save each variable in its own folder
    for var_name in dataset.data_vars:

        # Create the output directory for the variable if it doesn't exist
        output_dir = os.path.join(path_out, str(var_name))
        os.makedirs(output_dir, exist_ok=True)

        # Select data corresponding to variable to be saved
        var_data = dataset[var_name]
        year = var_data[t_dim_name].dt.year.values[0]

        # Save
        var_data.to_netcdf(os.path.join(output_dir, f"{year}.nc"))
