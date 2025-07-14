import numpy as np
import xarray as xr

from formatting import *
from readwrite import *
from config import get_config


def get_postfs_precip_accum_for_grid_point(
        wx_data: xr.Dataset,
        mask_data: np.ndarray,
        ) -> float:
    """
    Get the accumulated precipitation from after the fire season ends to
    the end of the given year. Assumes only a single fire season is present.
    (No shoulder fire seasons are allowed.)
    If the fire season is active or inactive for the entire year, returns zero.

    Parameters
    ----------
    wx_data: xr.Dataset
        A dataset containing an hourly PREC variable.
        First dim must be time, followed by the spatial dims.
    mask_data: np.ndarray
        A 3D array with the fire season mask.
        First dim must be time, followed by the spatial dims.

    Returns
    -------
    float
        The accumulated precipitation from after the fire season to the
        end of the year.
    """

    # Get the precipitation data
    hourly_precipitation_array = wx_data["PREC"].values[:, 0, 0]
    season_mask_array = mask_data[:, 0, 0]

    # The indices of the first time step after the fire season ends
    transition_indices = np.where(np.diff(season_mask_array) == 1)[0]
    if len(transition_indices) == 0:
        postfs_precip_accum = 0.
    else:
        postfs_precip_accum = np.sum(hourly_precipitation_array[transition_indices[-1] + 1:])

    return postfs_precip_accum


def get_prefs_precip_accum_for_grid_point(
        wx_data: xr.Dataset,
        mask_data: np.ndarray,
        ) -> float:
    """
    Get the accumulated precipitation from start of the year to
    start of the fire season. Assumes only a single fire season is present.
    (No shoulder fire seasons are allowed.)
    If the fire season is active or inactive for the entire year, returns zero.

    Parameters
    ----------
    wx_data: xr.Dataset
        A dataset containing an hourly PREC variable.
        First dim must be time, followed by the spatial dims.
    mask_data: np.ndarray
        A 3D array with the fire season mask.
        First dim must be time, followed by the spatial dims.

    Returns
    -------
    float
        The accumulated precipitation from start of year to
        the beginning of the fire season.
    """

    # Get the precipitation data
    hourly_precipitation_array = wx_data["PREC"].values[:, 0, 0]
    season_mask_array = mask_data[:, 0, 0]

    # The indices of the first time step of the fire season
    transition_indices = np.where(np.diff(season_mask_array) == 1)[0]
    if len(transition_indices) == 0:
        prefs_precip_accum = 0.
    else:
        prefs_precip_accum = np.sum(hourly_precipitation_array[:transition_indices[0]])

    return prefs_precip_accum


def overwinter_DC (final_fall_DC: float,
                   winter_precip_accum: float,
                   ) -> int:
    """
    Overwinter the Drought Code (DC) using the accumulated precipitation
    after the fire season ends.

    Parameters
    ----------
    final_fall_DC : float
        The Drought Code value at the end of the previous fire season.
    winter_precip_accum : float
        The accumulated precipitation after the previous fire season ends.

    Returns
    -------
    float
        The new Drought Code value after overwintering.
    """

    config = get_config()
    carry_over_fraction = config["calculation_parameters"]["carry_over_fraction"]
    wetting_efficiency_fraction = config["calculation_parameters"]["wetting_efficiency_fraction"]

    # Overwintering logic for Drought Code
    fall_moisture_equivalent_DC = 800. * np.exp(-final_fall_DC / 400.)
    spring_moisture_equivalent_DC = (
        carry_over_fraction * fall_moisture_equivalent_DC
        + 3.94 * winter_precip_accum * wetting_efficiency_fraction
    )
    startup_DC = 400. * np.log(spring_moisture_equivalent_DC / 800.)
    startup_DC = np.clip(startup_DC, 15, None)  # Ensure DC is within valid range

    return int(startup_DC)
