import xarray as xr
import yaml


def get_config(path_to_config: str = "../conf/config.yaml") -> dict:
    """
    Load configuration from a YAML file.

    Parameters:
        path_to_config (str): Path to the configuration file. Default is "../conf/config.yaml".
    
    Returns:
        dict: Configuration dictionary.
    """
    with open(path_to_config, "r") as file:
        config = yaml.safe_load(file)
    return config


def get_paths_to_data(path_to_config: str = "../conf/config.yaml") -> dict:
    """
    Get paths to data from the configuration file.
    Parameters:
        path_to_config (str): Path to the configuration file. Default is "../conf/config.yaml".
    
    Returns:
        dict: Dictionary containing data paths.
    """
    config = get_config(path_to_config)
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
    alternative_TEMP_names = ['T2', 'Temperature', 'T', 'Temp', 'temp', 'temperature', 'air_temperature']
    alternative_RH_names = ['RH2', 'rh', 'RelativeHumidity', 'relativehumidity', 'relative_humidity']
    alternative_WS_names = ['WS10', 'WindSpeed', 'WindSpeed10m', 'wind_speed']
    alternative_PREC_names = ['Precipitation', 'Precip', 'Rainfall', 'Rain', 'prec', 'precip', 'precipitation']

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


def stack_spatial_dims(data: xr.Dataset, x_name: str, y_name: str) -> xr.Dataset:
    """
    Stacks the two spatial dimensions (x_name and y_name) of the dataset into a single
    coordinate variable.

    Parameters
    ----------
    data : xr.Dataset
        The dataset containing the spatial coordinates.

    Returns
    -------
    xr.Dataset
        The dataset with stacked spatial coordinates.
    """

    # Stack the latitude and longitude coordinates into a single coordinate variable
    data = data.stack(xy=[x_name, y_name])

    return data





# # TESTING; TEMPORARY
# from glob import glob
# years = [2008]
# paths = []
# for year in years:
#     paths += sorted(glob(f"/users/rpayne/data/unproc/WRF/ctl/TAS/*.nc"))
# data = xr.open_mfdataset(paths).isel(rlat=slice(20,276),rlon=slice(110,366)).sel(time=slice(f"{years[0]}-01-01",f"{years[-1]}-12-31"))
# data = rename_wx_variables(data)
# data = rename_coordinates(data)
# data = stack_spatial_dims(data, x_name='rlon', y_name='rlat')
# print(data.dims)