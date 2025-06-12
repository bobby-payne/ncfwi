import sys
import time
import gc
import pandas as pd
import numpy as np
from utils import *
from inout import *
from season import *
gc.collect()


def preprocess_data():
    """
    Initialize by loading the configuration and data,
    and performing some preprocessing steps.
    This function should be called at the start of the script.

    Returns:
        data (xarray.Dataset): The preprocessed dataset ready for FWI calculation.
    """

    # Load in the config and data
    print("Loading data...")
    data = load_data()

    # Do a few preprocessing steps
    print("Preprocessing data...")
    data = transpose_dims(data)
    data = rename_coordinates(data)
    data = rename_wx_variables(data)
    data = apply_transformations(data)
    data = apply_spatial_crop(data)

    return data


def to_pandas_dataframe(wx_data: xr.Dataset) -> pd.DataFrame:
    """
    Turns the wx data for a single year into a pandas DataFrame, and formats it
    to be compatible with the cffdrs FWI code.
    This is necessary because the cffdrs FWI code requires a pandas DataFrame in
    a specific format.

    Args:
        data (xarray.Dataset): The wx data for a single year.

    Returns:
        wx_dataframe (pandas.DataFrame): The wx data formatted as a pandas
        DataFrame, ready for FWI calculation.
    """

    print("Converting to dataframe and reformating...")
    start_time = time.time()

    wx_dataframe = wx_data.to_dataframe()
    wx_dataframe.index = pd.to_datetime(wx_dataframe.index)
    wx_dataframe['YR'] = wx_dataframe.index.year.astype('int')
    wx_dataframe["MON"] = wx_dataframe.index.month.astype('int')
    wx_dataframe["DAY"] = wx_dataframe.index.day.astype('int')
    wx_dataframe["HR"] = wx_dataframe.index.hour.astype('int')
    wx_dataframe = wx_dataframe.reset_index().drop(columns=['time'])

    end_time = time.time()
    print(f"to dataframe took {end_time - start_time:.2f} seconds")

    return wx_dataframe


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


if __name__ == "__main__":

    # Open and read config
    print("Opening config and preprocessing...")
    config = get_config()
    t_dim_name = config["data_vars"]["t_dim_name"]
    x_dim_name = config["data_vars"]["x_dim_name"]
    y_dim_name = config["data_vars"]["y_dim_name"]
    start_year = config["settings"]["start_year"]
    end_year = config["settings"]["end_year"]
    path_to_cffdrs_ng = config["settings"]["path_to_cffdrs-ng"]
    FFMC_DEFAULT = config["FWI_parameters"]["FFMC_default"]
    DMC_DEFAULT = config["FWI_parameters"]["DMC_default"]
    DC_DEFAULT = config["FWI_parameters"]["DC_default"]

    time_range = np.arange(start_year, end_year + 1)
    sys.path.append(path_to_cffdrs_ng)
    from NG_FWI import hFWI

    # Initialize data
    data = preprocess_data()
    print("Data preprocessed successfully.")

    # Main computation loop
    for year in time_range:

        #Load data for the current year into memory
        print(f"Loading data for year {year} into memory...")
        print("(This may take a while!)")
        data_i = data.sel({t_dim_name: str(year)}).compute()

        # Obtain the fire season mask for this year
        print(f"Computing fire season for {year}...")
        fire_season_mask_i = compute_fire_season(data_i, return_as_xarray=False)

        # Loop through each grid point in the data
        for x in data_i[x_dim_name].values:
            for y in data_i[y_dim_name].values:

                # Get the timezone (UTC offset) for the current grid point
                # lon, lat = data_ixy["long"].values, data_ixy["lat"].values
                # UTC_offset = get_timezone_UTC_offset(lat, lon)

                # Select the data for the current grid point
                # and convert to pandas DataFrame
                data_pd_ixy = to_pandas_dataframe(
                    data_i.sel({x_dim_name: x, y_dim_name: y})
                )

                # Get the UTC offset for the current grid point
                lon, lat = data_pd_ixy["long"].values[0], data_pd_ixy["lat"].values[0]
                UTC_offset = get_timezone_UTC_offset(lat, lon)
                print(lon, lat, UTC_offset)

                # Apply the fire season mask to the data for this grid point
                fireSeason_mask_ixy = fire_season_mask_i[:, x, y]
                data_pd_masked_ixy = data_pd_ixy[fireSeason_mask_ixy]

                # If there is data for the fire season, calculate the FWIs
                start_time = time.time()
                if any(fireSeason_mask_ixy):
                    FWI_pd_ixy = hFWI(
                        data_pd_masked_ixy,
                        UTC_offset,
                        ffmc_old=FFMC_DEFAULT,
                        dmc_old=DMC_DEFAULT,
                        dc_old=DC_DEFAULT,
                        )
                else:
                    FWI_pd_ixy = pd.DataFrame({
                        'fwi': [np.nan] * len(fireSeason_mask_ixy),
                        'bui': [np.nan] * len(fireSeason_mask_ixy),
                        'isi': [np.nan] * len(fireSeason_mask_ixy),
                        'ffmc': [np.nan] * len(fireSeason_mask_ixy),
                        'dmc': [np.nan] * len(fireSeason_mask_ixy),
                        'dc': [np.nan] * len(fireSeason_mask_ixy),
                    })
                end_time = time.time()
                print(f"FWI calculation for ({x}, {y}) took {end_time - start_time:.2f} seconds")


        print("success!!!!!")



    gc.collect()

