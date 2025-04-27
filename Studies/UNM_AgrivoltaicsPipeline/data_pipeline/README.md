# README

## Overview

This pipeline updates irradiance values to account for the presence of solar panels, combines them with standard TMY weather data, and then runs a crop model using pySTICS.  
The workflow consists of three main scripts: `bifacialVF.py`, `data_form.py`, and `pystics_run.py`.

## Requirements

- **Python version:** >= 3.11  
- **Packages:** Install required packages using:

```bash
pip install -r requirements.txt
```

## Workflow Instructions

1. **Run `bifacialVF.py`**

   This script generates daily irradiance values (MJ/m²) based on the bifacialVF model.

2. **Run `data_form.py`**

   Use the method `process_weather_with_irradiance(index, filename)` to process the data.  
   - **index**: The location (integer) in the array of generated irradiance values you want to use (e.g., `43`).
   - **filename**: The name of the output `.txt` file (e.g., `"row_43_irradiance.txt"`).

   **Important:**  

    This step requires you to provide a TMY (Typical Meteorological Year) weather file.

    You can download a TMY file from the NREL NSRDB

    When downloading, select the following attributes:

    Year, Month, Day, Hour, Minute

    Temperature

    GHI

    Dew Point

    Relative Humidity

    Wind Speed

    Pressure is not needed.
    Set the time interval to 60 minutes.

3. **Run `pystics_run.py`**

   This script uses pySTICS to run a crop simulation model using the updated weather file.  
   - pySTICS simulates crop growth based on the new weather conditions that account for the impact of solar panels.
