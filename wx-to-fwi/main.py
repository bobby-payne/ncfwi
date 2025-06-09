import numpy as np
import xarray as xr
import pandas as pd
import numba
import time
import gc
from utils import *
from inout import *
from season import *
numba.set_num_threads(4)

# Load in the data and do a few preprocessing steps
print("Loading data...")
data = load_data()

print("Preprocessing data...")
data = rename_coordinates(data)
data = rename_wx_variables(data)
data = apply_transformations(data)
data = apply_spatial_indexing(data)

# Obtain the fire season mask
print("Calculating maximum daily temperature...")
daily_max_temp_data = get_max_daily_temperature(data)

print("Computing fire season... (this will take a few minutes)")
start_time = time.time()
fire_season_mask = compute_fire_season(daily_max_temp_data, return_as_xarray=False)
end_time = time.time()
print(f"compute_fire_season took {end_time - start_time:.2f} seconds")
print(fire_season_mask.shape)

gc.collect()
