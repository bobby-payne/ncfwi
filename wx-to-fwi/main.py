import time
import gc
from numpy import arange
from utils import *
from inout import *
from season import *
gc.collect()

# Load in the config, data
print("Loading data...")
config = get_config()
t_dim_name = config["data_vars"]["t_dim_name"]
start_year = config["settings"]["start_year"]
end_year = config["settings"]["end_year"]
time_range = arange(start_year, end_year + 1)
data = load_data()

# Do a few preprocessing steps
print("Preprocessing data...")
data = transpose_dims(data)
data = rename_coordinates(data)
data = rename_wx_variables(data)
data = apply_transformations(data)
data = apply_spatial_crop(data)

# main computation loop
for year in time_range:

    data_i = data.sel({t_dim_name: str(year)})

    # Obtain the fire season mask
    print(f"Computing fire season for {year}... (this may take a while)")
    start_time = time.time()
    fire_season_mask = compute_fire_season(data_i, return_as_xarray=False)
    end_time = time.time()
    print(f"compute_fire_season took {end_time - start_time:.2f} seconds")

    # the cffdrs FWI code requires a pandas DataFrame
    print("Converting to dataframe... (this may take a while)")
    start_time = time.time()
    data_i_stacked = data_i.stack(xy=('rlon', 'rlat'))
    data_i_pd = data_i_stacked.to_dataframe()
    end_time = time.time()
    print(f"to dataframe took {end_time - start_time:.2f} seconds")

    # print("Applying fire season mask...")
    # data_i_pd_masked = data_pd[fire_season_mask]

# print("success")
gc.collect()
