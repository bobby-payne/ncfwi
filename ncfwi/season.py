import numpy as np
import xarray as xr
import time
from scipy.ndimage import uniform_filter1d
from typing import Union
from numba import njit

from formatting import *
from readwrite import *
from config import get_config


def get_max_daily_temperature(wx_data: xr.Dataset) -> xr.DataArray:
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


@njit
def apply_fire_season_logic(idx_above_start_threshold_consecutive: np.ndarray,
                            idx_below_stop_threshold_consecutive: np.ndarray,
                            season_consecutive_days: int) -> np.ndarray:
    """
    Function that applies the indexes computed in compute_fire_season to create a mask. Wrapped with numba's njit for performance.

    Parameters
    ----------
    idx_above_start_threshold_consecutive : np.ndarray
        A boolean array indicating whether the temperature is above the start threshold for X consecutive days.
    idx_below_start_threshold_consecutive : np.ndarray
        A boolean array indicating whether the temperature is below the stop threshold for X consecutive days.
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
                if t < season_consecutive_days: # the first N=season_consecutive_days days are always off
                    on = False
                else:
                    if not on and idx_above_start_threshold_consecutive[t, i, j]:
                        on = True
                    elif on and idx_below_stop_threshold_consecutive[t, i, j]:
                        on = False
                mask[t, i, j] = on
    return mask


# This function can be rewritten to be faster
# should it become a bottleneck (as of now it is not?).
# (numba doesn't like numpy.where)
@njit
def adjust_for_shoulder_seasons(fire_season_mask: np.ndarray) -> np.ndarray:
    """
    Due to how the fire season is defined, the upper and lower
    maximum temperature thresholds may be met multiple times
    throughout the year resulting in short periods of fire season
    in the shoulder seasons. This function accounts for those shoulders
    by turning the mask off (i.e., turning the fire season on) for all dates
    between the first time the upper threshold is met and the last time the
    lower threshold is met. This is different than in McElhinny et al. (2020).

    Parameters
    ----------
    fire_season_mask : np.ndarray
        A boolean mask indicating the fire season.

    Returns
    -------
    np.ndarray
        A boolean mask indicating the adjusted fire season.
    """
    T, H, W = fire_season_mask.shape
    fire_season_mask_adjusted = fire_season_mask.copy()

    for i in range(H):
        for j in range(W):

            fire_season_mask_ij = fire_season_mask[:, i, j]
            active_season_indices = np.where(fire_season_mask_ij == 1)[0]
            if len(active_season_indices) == 0:  # if no fire season
                continue

            istart, istop = active_season_indices[0], active_season_indices[-1]
            fire_season_mask_adjusted[istart:istop, i, j] = 1

    return fire_season_mask_adjusted


def make_mask_hourly(fire_season_mask_daily: np.ndarray, utc_offset: int = 0) -> np.ndarray:
    """
    Convert a daily fire season mask to an hourly fire season mask.

    Parameters
    ----------
    fire_season_mask_daily : np.ndarray
        A boolean mask indicating the fire season on a daily basis. Shape (N_days x 1 x 1).
    utc_offset : int, optional
        The UTC offset to convert the time dimension to local time. Default is 0.

    Returns
    -------
    np.ndarray
        A boolean mask indicating the fire season on an hourly basis.
    """

    if utc_offset == 0:
        fire_season_mask_hourly = np.repeat(fire_season_mask_daily, 24, axis=0)
    elif utc_offset > 0:
        raise NotImplementedError("UTC offset > 0 not implemented yet.")
    else:
        fire_season_mask_hourly_start = np.repeat(fire_season_mask_daily[0], np.abs(utc_offset), axis=0)[:, :, np.newaxis]
        fire_season_mask_hourly_end = np.repeat(fire_season_mask_daily[-1], 24 - np.abs(utc_offset), axis=0)[:, :, np.newaxis]
        fire_season_mask_hourly_middle = np.repeat(fire_season_mask_daily[1:-1], 24, axis=0)
        fire_season_mask_hourly = np.concatenate((
            fire_season_mask_hourly_start,
            fire_season_mask_hourly_middle,
            fire_season_mask_hourly_end
        ), axis=0)

    return fire_season_mask_hourly



def compute_fire_season(wx_data: xr.Dataset,
                        return_as_xarray: bool = False,
                        utc_offset: int = 0,
                        ) -> Union[np.ndarray, xr.Dataset]:
    """
    Compute the fire season from the maximum daily temperature.

    Parameters
    ----------
    wx_data : xr.Dataset
        The dataset containing wx data, including temperature in deg C.
    return_as_xarray : bool, optional
        If True, returns the result as an xarray Dataset with the same
        dimensions as temperature_data. If False, returns a numpy array.
        Default is False.
    utc_offset : int, optional
        The UTC offset to convert the time dimension to local time.
        Default is 0.
    Returns
    -------
    xr.Dataset
        A dataset indicating the fire season.
    """

    # read info from config
    config = get_config()
    season_consecutive_days = config["calculation_parameters"]["season_consecutive_days"]
    season_start_temperature = config["calculation_parameters"]["season_start_temp"]
    season_stop_temperature = config["calculation_parameters"]["season_stop_temp"]

    # compute the daily max temperature
    temperature_data = get_max_daily_temperature(wx_data)

    # compute the indices for which the daily max temperature
    # exceeds the threshold three days in a row
    idx_above_start_temp = (temperature_data > season_start_temperature)
    idx_below_stop_temp = (temperature_data < season_stop_temperature)
    idx_above_start_temp_consecutive = uniform_filter1d(
        idx_above_start_temp.data.astype("int8"),
        size=season_consecutive_days,
        axis=0,
        mode="constant",
        cval=0.,
        origin=1
    )
    idx_below_stop_temp_consecutive = uniform_filter1d(
        idx_below_stop_temp.data.astype("int8"),
        size=season_consecutive_days,
        axis=0,
        mode="constant",
        cval=0.,
        origin=1
    )
    fire_season_mask = apply_fire_season_logic(
        idx_above_start_temp_consecutive,
        idx_below_stop_temp_consecutive,
        season_consecutive_days
    )
    fire_season_mask = adjust_for_shoulder_seasons(fire_season_mask)

    # daily -> hourly
    fire_season_mask = make_mask_hourly(fire_season_mask, utc_offset=utc_offset)

    if return_as_xarray:
        fire_season_mask_xr = xr.DataArray(
            fire_season_mask,
            dims=wx_data["TEMP"].dims,
            coords=wx_data["TEMP"].coords,
            name="MASK"
        ).to_dataset()

        return fire_season_mask_xr  # type: ignore

    return fire_season_mask  # type: ignore
