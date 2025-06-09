import numpy as np
import xarray as xr
from utils import *
from inout import *
from season import *


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
fire_season_mask = compute_fire_season(daily_max_temp_data)

print(fire_season_mask)
print(fire_season_mask.shape)
