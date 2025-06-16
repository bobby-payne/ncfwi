# ncfwi
---
This is a Python package that ingests hourly gridded netcdf weather data (temperature, wind speed, relative humidity, and precipitation) to calculate and produce hourly gridded Fire Weather Indices (FWIs). Note that this package is **unofficial** and is not endorsed by NRCAN/RNCAN—it is purely for the purposes of my own research.

---

## Installation
Using Bash:
```bash
git clone https://github.com/bobby-payne/ncfwi.git
cd ncfwi
pip install -r requirements.txt
```
You will also need the clone and point to (in the config) the [NG-CFFDRS GitHub Repository](https://github.com/nrcan-cfs-fire/cffdrs-ng), as my package implements their functions for the calculation of hourly FWIs:
```bash
git clone https://github.com/nrcan-cfs-fire/cffdrs-ng.git
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

### Overwintering: Coming Soon!

## Feedback and Bug Reports
If you have any suggestions or wish to report a bug, please feel free to contribute by [creating an issue](https://github.com/bobby-payne/ncfwi/issues). I'm far from being a programming expert, so I'm sure there are many potential improvements to be made.
