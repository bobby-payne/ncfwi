import xarray as xr
import pandas as pd
import numpy as np
from typing import Union
from types import NoneType

from config import get_config


def transpose_dims(data: xr.Dataset) -> xr.Dataset:
    """
    Transpose the dimensions of the dataset to match the expected format.
    The expected format is (time, spatial) where time is the first dimension,
    followed by the other dims in the same order they were provided in.

    Parameters
    ----------
    data : xr.Dataset
        The dataset containing the weather variables.

    Returns
    -------
    xr.Dataset
        The dataset with transposed dimensions.
    """

    # read in the user-specified dimensions
    config = get_config()
    t_dim_name = config["data_vars"]["t_dim_name"]

    # Transpose the dataset to the expected format
    for wx_var in data.data_vars:
        if not data[wx_var].dims[0] == t_dim_name:
            data = data.transpose(t_dim_name, ...)

    return data


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


def apply_spatial_crop(wx_data: xr.Dataset) -> xr.Dataset:
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

    crop_x_index = config["settings"]["crop_x_index"]
    if not (crop_x_index is None or crop_x_index == []):
        x0, x1 = crop_x_index
        wx_data = wx_data.isel(
            {x_dim: slice(x0, x1 + 1)}
        )
    
    crop_y_index = config["settings"]["crop_y_index"]
    if not (crop_y_index is None or crop_y_index == []):
        y0, y1 = crop_y_index
        wx_data = wx_data.isel(
            {y_dim: slice(y0, y1 + 1)}
        )

    return wx_data


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
            transform = eval("lambda x:" + expression)  # TODO: replace eval with a secure alternative (e.g numexpr)
            wx_data[wx_var] = transform(wx_data[wx_var])

    return wx_data


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