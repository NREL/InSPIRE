# InSPIRE_Agrivolt

``inspire_agrivolt`` contains CLI tools to assist in the creation of the 2025 Inspire Agrivoltaics Irradiance Dataset.


### Motivation

The CLI tools provided in this repo allow easy reproducibility of the dataset on NREL HPC resources (dataset produced on Kestrel).

### Tools


**ground_irradiance**: runs geospatial SAM simulation on NSRDB state using PySAM, and PVDeg.

    usage: agrivolt_ground_irradiance [-h] [--confs CONFS [CONFS ...]] [--partition PARTITION] [--account ACCOUNT] [--workers WORKERS] [--walltime WALLTIME] [--log_dir LOG_DIR] [--port PORT]
                                    [--local-weather LOCAL_WEATHER] [--local-meta LOCAL_META]
                                    state target_dir conf_dir

    Run Agrivoltaics Irradiance PySAM simulation for a U.S. state

    positional arguments:
    state                 State name to run analysis on as it appears in NSRDB meta data, i.e. 'Colorado'. Only used when pulling NSRDB data
    target_dir            path to base output directory
    conf_dir              path to base configs directory

    optional arguments:
    -h, --help            show this help message and exit
    --confs CONFS [CONFS ...]
                            List of config names (default: 01 through 10)
    --partition PARTITION
                            slurm partition to use when scheduling
    --account ACCOUNT     nrel hpc account with allocation
    --workers WORKERS     number of dask workers (max 64 on kestrel shared partition)
    --walltime WALLTIME   max length of job from start to finish
    --log_dir LOG_DIR     location of dask log files
    --port PORT           dask dashboard port
    --local-weather LOCAL_WEATHER
                            Path to local weather NetCDF file
    --local-meta LOCAL_META
                            Path to local meta CSV file

**beds_postprocess**: runs postprocessing to calculate planting beds irradiance.

#### SAM

conf_dir can be tricky but should look like the following from the sam outputs. The model only needs the subdirectories to contain the ``pvsamv1.json`` config file for the model to work.

    path/to/dir/SAM

    ├── 01
    │   ├── 01_cashloan.json
    │   ├── 01_grid.json
    │   ├── 01.json
    │   ├── 01_pvsamv1.json
    │   ├── 01_utilityrate5.json
    │   ├── sscapi.h
    │   └── ssc.dll
    ├── 02
    │   ├── 02_cashloan.json
    │   ├── 02_grid.json
    │   ├── 02.json
    │   ├── 02_pvsamv1.json
    │   ├── 02_utilityrate5.json
    │   ├── sscapi.h
    │   └── ssc.dll
    ├── 03
    │   ├── 03_cashloan.json
    │   ├── 03_grid.json
    │   ├── 03.json
    │   ├── 03_pvsamv1.json
    │   ├── 03_utilityrate5.json
    │   ├── sscapi.h
    │   └── ssc.dll
    ├── 04
    │   ├── 04_cashloan.json
    │   ├── 04_grid.json
    │   ├── 04.json
    │   ├── 04_pvsamv1.json
    │   ├── 04_utilityrate5.json
    │   ├── sscapi.h
    │   └── ssc.dll
    ├── 05
    │   ├── 05_cashloan.json
    │   ├── 05_grid.json
    │   ├── 05.json
    │   ├── 05_pvsamv1.json
    │   ├── 05_utilityrate5.json
    │   ├── sscapi.h
    │   └── ssc.dll
    ├── 06
    │   ├── 06_cashloan.json
    │   ├── 06_grid.json
    │   ├── 06.json
    │   ├── 06_pvsamv1.json
    │   ├── 06_utilityrate5.json
    │   ├── sscapi.h
    │   └── ssc.dll
    ├── 07
    │   ├── 07_cashloan.json
    │   ├── 07_grid.json
    │   ├── 07.json
    │   ├── 07_pvsamv1.json
    │   ├── 07_utilityrate5.json
    │   ├── sscapi.h
    │   └── ssc.dll
    ├── 08
    │   ├── 08_cashloan.json
    │   ├── 08_grid.json
    │   ├── 08.json
    │   ├── 08_pvsamv1.json
    │   ├── 08_utilityrate5.json
    │   ├── sscapi.h
    │   └── ssc.dll
    ├── 09
    │   ├── 09_cashloan.json
    │   ├── 09_grid.json
    │   ├── 09.json
    │   ├── 09_pvsamv1.json
    │   ├── 09_utilityrate5.json
    │   ├── sscapi.h
    │   └── ssc.dll
    ├── 10
    │   ├── 10_cashloan.json
    │   ├── 10_grid.json
    │   ├── 10.json
    │   ├── 10_pvsamv1.json
    │   ├── 10_utilityrate5.json
    │   ├── sscapi.h
    │   └── ssc.dll
    └── SAM_10setups.sam


#### Example kestrel run

**There are many scripts in the scripts/ folder which can dispatch many slurm jobs at the same time for various state and configuration combinations.**

``$ agrivolt_ground_irradiance Colorado ~/dev/InSPIRE/Studies/USMap_Doubleday_2024/Full-Outputs/Colorado/ ~/dev/InSPIRE/Studies/USMap_Doubleday_2024/SAM --confs 01 --port 22118``

    - calculates ground irradiance for full resolution colorado NSRDB TMY dataset
    - output_dir: ~/dev/InSPIRE/Studies/USMap_Doubleday_2024/Full-Outputs/Colorado/
    - conf_dir: ~/dev/InSPIRE/Studies/USMap_Doubleday_2024/SAM 
    - confs: only run conf 01
    - port: expose dask dashboard on port 22118

### Continuing the pipeline

Kestrel jobs may run out of time, If this is the case, use ``completeness.ipynb`` to determine which gids are missing then echo the gids files back into an agrivolt irradiance job. Information on how to do this coming soon...  

This is being done in `completeness.py` currently. The function `inspire_agrivolt.check_completeness` examines all possible gids in a state then checks the directory to see if they are present. If there are missing gids they will be outputted by the function, then we can use this list of gids to run the script again to produce the missing outputs. **Insert script name here.**

After running fill-in calculations we need to combine them with the original model outputs before continuing with postprocessing. Use ``merge-fill.ipynb`` to do this. If your fills are not complete they will have to be re-run at some point. ``merge-fill.ipynb`` will allow you to silently continue with missing data...

At this point the data can be visually inspected to check for missing locations. Repeat the steps in the previous 3 paragraphs to fill in the data, if it is still missing.

Finally, we will run the beds postprocessing....
