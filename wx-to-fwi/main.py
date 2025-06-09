import numpy as np
import xarray as xr
from utils import *
from inout import *
from season import *
import time


print("Loading data...")
data = load_data()

print("Preprocessing data...")
data = rename_coordinates(data)
data = rename_wx_variables(data)
data = apply_transformations(data)
data = apply_spatial_indexing(data)

print("Calculating maximum daily temperature...")
daily_max_temp_data = get_max_daily_temperature(data)

print("Computing fire season... (this will take a few minutes)")
start_time = time.time()
fire_season_mask = compute_fire_season(daily_max_temp_data, return_as_xarray=True)
end_time = time.time()
print(f"compute_fire_season took {end_time - start_time:.2f} seconds")

fire_season_mask.to_netcdf("/users/rpayne/fire_season_mask.nc")
print("success")

# print(fire_season_mask)
