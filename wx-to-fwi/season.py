import sys
import numpy as np
import xarray as xr
from numba import njit
from utils import *
from inout import *


# index of the first hour of each month
# month_index = np.cumsum(24*np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]))
# month_index_leap = np.cumsum(24*np.array([0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30]))


def get_max_daily_temperature(wx_data: xr.Dataset) -> xr.Dataset:
    """
    Simple function to calculate the maximum daily temperature from the dataset.
    The function assumes that the dataset has a time dimension with hourly data
    and has just gone through io.load_data()

    Parameters
    ----------
    data : xr.Dataset
        The dataset containing hourly temperature data.
    Returns
    -------
    xr.Dataset
        A dataset containing the maximum daily temperature.
    """

    temperature_hourly = wx_data["TEMP"]
    temperature_dailymax = temperature_hourly.resample(time="1D").max()

    return temperature_dailymax


# def compute_fire_season(temperature_data: xr.Dataset) -> xr.Dataset:
#     """
#     Compute the fire season from the maximum daily temperature.

#     Parameters
#     ----------
#     temperature_data : xr.Dataset
#         The dataset containing daily maximum temperature data.
#     Returns
#     -------
#     xr.Dataset
#         A dataset indicating the fire season.
#     """

#     # read info from config
#     config = get_config()
#     path_to_Ensemble_FWI = config["settings"]["path_to_Ensemble_FWI"]
#     season_consecutive_days = config["settings"]["season_consecutive_days"]
#     season_start_temperature = config["settings"]["season_start_temp"]
#     season_stop_temperature = config["settings"]["season_stop_temp"]
#     x_dim_name = config["data_vars"]["air_temperature"]["x_dim_name"]
#     y_dim_name = config["data_vars"]["air_temperature"]["y_dim_name"]

#     # Add ENSEMBLE FWI to path and try importing the fireSeason function
#     if path_to_Ensemble_FWI not in sys.path:
#         sys.path.append(path_to_Ensemble_FWI)
#     if 'fireSeason' not in globals():
#         try:
#             from over_wintering import fireSeason
#         except ImportError:
#             raise ImportError("Could not import Ensemble_FWI."
#                               "Please make sure the path given in the config is correct.")

#     # the fire season function from Ensemble_FWI requires the coordinates to be stacked
#     temperature_data_stacked = temperature_data.stack(xy=[x_dim_name, y_dim_name])
#     # Drop unwanted coordinates if they exist
#     coords_to_drop = [coord for coord in temperature_data_stacked.coords if coord not in ["time", "xy"]]
#     temperature_data_stacked = temperature_data_stacked.drop_vars(coords_to_drop)
#     print(temperature_data_stacked)
#     fire_season_mask = fireSeason(temperature_data_stacked,
#                                   start_temp=season_start_temperature,
#                                   stop_temp=season_stop_temperature,
#                                   consecutive=season_consecutive_days,)[:,0]

#     return fire_season_mask

@njit
def apply_fire_season_logic(idx_above_start_threshold_consecutive, idx_below_start_threshold_consecutive, season_consecutive_days):
    """
    Function that applies the indexes computed in compute_fire_season to create a mask. Wrapped with numba's njit for performance.

    Parameters
    ----------
    idx_above_start_threshold_consecutive : np.ndarray
        A boolean array indicating whether the temperature is above the start threshold for X consecutive days.
    idx_below_start_threshold_consecutive : np.ndarray
        A boolean array indicating whether the temperature is below the start threshold for X consecutive days.
    season_consecutive_days : int
        The number of consecutive days required to consider it the fire season (i.e., X in the previous two parameters).

    Returns
    -------
    np.ndarray
        A boolean mask indicating whether the fire season is active or not.
    """

    T, H, W = idx_above_start_threshold_consecutive.shape
    mask = np.zeros((T, H, W), dtype=np.bool_)
    for i in range(H):
        for j in range(W):
            on = False
            for t in range(T):
                if t < season_consecutive_days: # e.g., if season_consecutive_days = 3, then the first two days are nans
                    on = False
                else:
                    if not on and idx_above_start_threshold_consecutive[t, i, j]:
                        on = True
                    elif on and idx_below_start_threshold_consecutive[t, i, j]:
                        on = False
                mask[t, i, j] = on
    return mask


def compute_fire_season(temperature_data: xr.Dataset) -> xr.Dataset:
    """
    Compute the fire season from the maximum daily temperature.

    Parameters
    ----------
    temperature_data : xr.Dataset
        The dataset containing daily maximum temperature data.
    Returns
    -------
    xr.Dataset
        A dataset indicating the fire season.
    """

    # read info from config
    config = get_config()
    season_consecutive_days = config["settings"]["season_consecutive_days"]
    season_start_temperature = config["settings"]["season_start_temp"]
    season_stop_temperature = config["settings"]["season_stop_temp"]
    t_dim_name = config["data_vars"]["air_temperature"]["t_dim_name"]

    # compute the indices for which the daily max temperature exceeds the threshold three days in a row
    idx_above_start_temp = (temperature_data > season_start_temperature)
    idx_below_stop_temp = (temperature_data < season_stop_temperature)
    idx_above_start_temp_consecutive = (idx_above_start_temp.rolling({t_dim_name: season_consecutive_days}).sum(skipna=False) == season_consecutive_days)
    idx_below_stop_temp_consecutive = (idx_below_stop_temp.rolling({t_dim_name: season_consecutive_days}).sum(skipna=False) == season_consecutive_days)
    idx_above_start_temp_consecutive = idx_above_start_temp_consecutive.values
    idx_below_stop_temp_consecutive = idx_below_stop_temp_consecutive.values
    fire_season_mask = apply_fire_season_logic(idx_above_start_temp_consecutive, idx_below_stop_temp_consecutive, season_consecutive_days)

    return fire_season_mask


print("Loading data...")
data = load_data()
print("Calculating maximum daily temperature...")
daily_max_temp_data = get_max_daily_temperature(data)
print("Computing fire season... (this will take a few minutes)")
fire_season_mask = compute_fire_season(daily_max_temp_data)
print(fire_season_mask)
print(fire_season_mask.shape)
