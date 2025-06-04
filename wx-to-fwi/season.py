import numpy as np
import xarray as xr
from utils import *
from inout import *

# index of the first hour of each month
month_index = np.cumsum(24*np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]))
month_index_leap = np.cumsum(24*np.array([0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30]))


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

data = load_data()
print(get_max_daily_temperature(data))