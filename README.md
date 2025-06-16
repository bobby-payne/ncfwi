## ncfwi
---
This is an **unofficial** package that ingests hourly gridded netcdf weather data (temperature, wind speed, relative humidity, and precipitation) to calculate and produce hourly gridded Fire Weather Indices (FWIs).

---

## Installation
```bash
git clone https://github.com/bobby-payne/ncfwi.git
cd ncfwi
pip install -r requirements.txt
```

## Usage
Edit `conf/config.yaml` with the relevant information for computing the FWIs from your netcdf data. Then simply run `compute_indices.py` with your python interpreter. The FWIs will be saved in your specified output directory like so.
```
your_output_directory/
├── FWI_2004.nc        
├── FWI_2005.nc    
├── FWI_2006.nc       
└── FWI_2007.nc    
```
You can specify which indices and/or weather variables you would like returned in the output, including the fire season mask. A full list of suitable output variables are listed in `available_outputs.txt`.

## Feedback and Bug Reports
If you have any suggestions or wish to report a bug, please feel free to contribute by [creating an issue](https://github.com/bobby-payne/ncfwi/issues). My coding skills are still amateur, so I'm sure there any many potential improvements to be made.
