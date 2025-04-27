import pandas as pd
import os
from pystics.params import CropParams, SoilParams, ManageParams, InitialParams, Constants, StationParams
from pystics.params import parametrization_from_stics_example_files
from pystics.simulation import run_pystics_simulation

mocked_weather_path = "row_61_irradiance.txt"
mocked_weather_path_year2 = "row_61_irradiance.txt"
species = "common_wheat"
variety = "Talent"

_, crop, manage, soil, station, constants, initial = parametrization_from_stics_example_files(species, variety)

soil = SoilParams(
    CLAY=0.5, SAND=0.4, SILT=0.3, SOC=5, ARGI=0.3,
    DAF_1=1.2, EPC_1=150, Q0=12, HMINF_1=1.8, HCCF_1=3.6
)

weather_year1 = pd.read_csv(mocked_weather_path, header=None, sep='\s+')
weather_year1.columns = ['file', 'year', 'month', 'day', 'doy', 'temp_min', 'temp_max', 
                         'radiation', 'etp', 'rain', 'wind', 'tpm', 'co2']

weather_year2 = pd.read_csv(mocked_weather_path_year2, header=None, sep='\s+')
weather_year2.columns = ['file', 'year', 'month', 'day', 'doy', 'temp_min', 'temp_max', 
                         'radiation', 'etp', 'rain', 'wind', 'tpm', 'co2']

weather = pd.concat([weather_year1, weather_year2], axis=0)
weather = weather.drop(columns=["file"], errors="ignore")
weather['date'] = pd.to_datetime(dict(year=weather.year, month=weather.month, day=weather.day))
weather = weather.reset_index(drop=True)

numeric_cols = ['doy', 'temp_min', 'temp_max', 'radiation', 'etp', 'rain', 'wind', 'tpm', 'co2']
weather[numeric_cols] = weather[numeric_cols].apply(pd.to_numeric, errors="coerce")

weather.replace(-999.9, pd.NA, inplace=True)

weather['temp_min'] = weather['temp_min'].fillna(10.0)
weather['temp_max'] = weather['temp_max'].fillna(20.0)
weather['tpm'] = weather['tpm'].fillna((weather['temp_min'] + weather['temp_max']) / 2)
weather['etp'] = weather['etp'].fillna(2.5)
weather['wind'] = weather['wind'].fillna(2.0)
weather['co2'] = weather['co2'].fillna(400.0)
weather['rain'] = weather['rain'].fillna(0.0)

pystics_df, pystics_mat_list = run_pystics_simulation(
    weather, crop, soil, constants, manage, station, initial
)

print(pystics_df.loc[0, ['tmoy', 'lev', 'temp_max', 'temp_min', 'et', 'z0', 'tpm', 'wind', 'lai', 'ratm', 'tcultmin', 'tcultmax']])
print(pystics_df.mafruit_rec.max())

pystics_df.to_csv('stics_run.csv', index=False)
