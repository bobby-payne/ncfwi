import xarray as xr
from utils import *


def load_data() -> xr.Dataset:
    """
    Open each wx data separately and then merge into an xarray dataset.

    Returns
    -------
    xarray.Dataset
        The merged wx dataset.

    """

    # Get paths (as strings) to data
    wx_data_paths = get_paths_to_data()
    wx_var_names = wx_data_paths.keys()

    # Open the data and temporarily store in a dictionary
    wx_data = {}
    for wx_var in wx_var_names:
        print("Opening data:", wx_var)
        path = wx_data_paths[wx_var] + f"2004*.nc"
        wx_data[wx_var] = xr.open_mfdataset(path)

    # Merge into an xarray.Dataset
    wx_data_xarray = xr.merge(wx_data.values(), join="exact")

    return wx_data_xarray
