import xarray as xr
import yaml


def get_config() -> dict:
    """
    Load configuration from a YAML file.
    
    Returns:
        dict: Configuration dictionary.
    """
    config_path = "./conf/config.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def get_paths_to_data() -> dict:
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


def rename_wx_variables(data: xr.Dataset) -> xr.Dataset:
    """
    Renames weather variables in the dataset to an format common to all
    functions in the wx-to-fwi package. Specifically, the temperature
    variable should be named 'TEMP', relative humidity should be 'RH',
    wind speed should be 'WS', and precipitation should be 'PREC'.

    Parameters
    ----------
    data : xr.Dataset
        The dataset containing the weather variables.

    Returns
    -------
    xr.Dataset
        The dataset with renamed variables.
    """

    # Common alternative names for certain weather variables
    config = get_config()
    alternative_TEMP_names = config["data_vars"]["air_temperature"]["alternative_names"]
    alternative_RH_names = config["data_vars"]["relative_humidity"]["alternative_names"]
    alternative_WS_names = config["data_vars"]["wind_speed"]["alternative_names"]
    alternative_PREC_names = config["data_vars"]["precipitation"]["alternative_names"]

    # Loop through variables in the dataset and rename them if they match any of the alternative names
    for var in data.data_vars:

        if var in alternative_TEMP_names:
            data = data.rename({var: 'TEMP'})
        elif var in alternative_RH_names:
            data = data.rename({var: 'RH'})
        elif var in alternative_WS_names:
            data = data.rename({var: 'WS'})
        elif var in alternative_PREC_names:
            data = data.rename({var: 'PREC'})

    return data


def rename_coordinates(data: xr.Dataset) -> xr.Dataset:
    """
    Renames coordinate variables in the dataset to a format common to all
    functions in the wx-to-fwi package. Specifically, the latitude and
    longitude coordinates should be named 'lat' and 'long'.

    Parameters
    ----------
    data : xr.Dataset
        The dataset containing the coordinate variables.

    Returns
    -------
    xr.Dataset
        The dataset with renamed coordinate variables.
    """

    # Common alternative names for latitude and longitude coordinates
    alternative_latitude_names = ['lat', 'latitude', 'LAT', 'LATITUDE', 'XLAT']
    alternative_longitude_names = ['lon', 'longitude', 'LON', 'LONGITUDE', 'XLONG', 'LONG']
    alternative_time_names = ['TIME', 'Time']

    # Loop through coordinates in the dataset and rename them if they match any of the alternative names
    for coord in data.coords:

        if coord in alternative_latitude_names:
            data = data.rename({coord: 'lat'})
        elif coord in alternative_longitude_names:
            data = data.rename({coord: 'long'})
        elif coord in alternative_time_names:
            data = data.rename({coord: 'time'})

    return data


def apply_spatial_indexing(wx_data: xr.Dataset) -> xr.Dataset:
    """
    Select a specific region by indexing the x and y dimensions
    as given by the user in the config.

    Parameters
    ----------
    wx_data : xarray.Dataset
        The dataset containing the weather variables.

    Returns
    -------
    xarray.Dataset
        The dataset with spatial indexing applied.
    """

    config = get_config()
    x_dim = config["data_vars"]["x_dim_name"]
    y_dim = config["data_vars"]["y_dim_name"]
    x0, x1 = config["settings"]["domain_index_x"]
    y0, y1 = config["settings"]["domain_index_y"]

    wx_data_subset = wx_data.isel({x_dim: slice(x0, x1), y_dim: slice(y0, y1)})

    return wx_data_subset


def apply_transformations(wx_data: xr.Dataset) -> xr.Dataset:
    """
    Transform the dataset by applying a mathematical operation to the variable.
    Transforms are specified in the config file and evaluated using Python's
    in-built eval function.

    Parameters
    ----------
    data : xarray.Dataset
        The dataset containing the weather variables.

    Returns
    -------
    xarray.Dataset
        The transformed dataset with the transformations applied.
    """
    # read in the user-specified transforms
    config = get_config()
    transform_dictionary = {
        "WS": config["data_vars"]["wind_speed"]["transform"],
        "TEMP": config["data_vars"]["air_temperature"]["transform"],
        "RH": config["data_vars"]["relative_humidity"]["transform"],
        "PREC": config["data_vars"]["precipitation"]["transform"],
    }

    # Apply each transformation to the corresponding variable
    for wx_var in transform_dictionary.keys():
        expression = transform_dictionary[wx_var]
        if expression is not None:
            print(f"Applying transformation to {wx_var}: {expression}")
            transform = eval("lambda x:" + expression) #TO-DO: replace eval with a secure alternative (e.g numexpr)
            wx_data[wx_var] = transform(wx_data[wx_var])

    return wx_data
