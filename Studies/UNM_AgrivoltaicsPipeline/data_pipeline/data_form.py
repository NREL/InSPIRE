import pandas as pd
import os

here = os.getcwd()
weather_file = os.path.join(here, "489708_40.69_-104.78_tdy-2023.csv")
irradiance_file = os.path.join(here, "TEMP", "daily_sensors.csv")
ghi_file = os.path.join(here, "TEMP", "daily_ghi.csv")

weather_df = pd.read_csv(weather_file, skiprows=2)
dirrad_df = pd.read_csv(irradiance_file, header=None, skiprows=1)
ghi_df = pd.read_csv(ghi_file)

date_columns = ['Year', 'Month', 'Day']
temp_column = 'Temperature'
if not all(col in weather_df.columns for col in date_columns):
    raise ValueError("Missing Year, Month, or Day columns.")
if temp_column not in weather_df.columns:
    raise ValueError("Missing Temperature column.")

daily_weather = weather_df.groupby(date_columns).agg(
    temp_min=(temp_column, 'min'),
    temp_max=(temp_column, 'max')
).reset_index()

def get_daily_irradiance(row_index, day_of_year):
    if 0 <= day_of_year < dirrad_df.shape[1] - 1 and row_index + 1 < dirrad_df.shape[0]:
        return float(dirrad_df.iloc[row_index + 1, day_of_year + 1])
    return -999.9

def get_daily_ghi(day_of_year):
    if 0 <= day_of_year < ghi_df.shape[0]:
        return float(ghi_df.iloc[day_of_year, 1])
    return -999.9

def process_weather_with_irradiance(row_index, output_file):
    modified_weather_df = daily_weather.copy()
    modified_weather_df['Year'] = 2023
    modified_weather_df['doy'] = -1
    modified_weather_df['radiation'] = -999.9

    for idx, row in modified_weather_df.iterrows():
        date = pd.Timestamp(year=int(row['Year']), month=int(row['Month']), day=int(row['Day']))
        doy = date.timetuple().tm_yday
        modified_weather_df.at[idx, 'doy'] = doy
        modified_weather_df.at[idx, 'radiation'] = get_daily_irradiance(row_index, doy - 1)

    additional_cols = {
        'etp': -999.9,
        'rain': -999.9,
        'wind': -999.9,
        'tpm': -999.9,
        'co2': -999.99
    }

    for col, default_val in additional_cols.items():
        if col == 'wind' and 'Wind Speed' in weather_df.columns:
            col_df = weather_df.groupby(date_columns).agg({'Wind Speed': 'mean'}).reset_index()
            col_df.rename(columns={'Wind Speed': 'wind'}, inplace=True)
            modified_weather_df = pd.merge(modified_weather_df, col_df, on=date_columns, how='left')
            modified_weather_df['wind'] = modified_weather_df['wind'].fillna(default_val)
        elif col == 'tpm' and 'Temperature' in weather_df.columns:
            col_df = weather_df.groupby(date_columns).agg({'Temperature': 'mean'}).reset_index()
            col_df.rename(columns={'Temperature': 'tpm'}, inplace=True)
            modified_weather_df = pd.merge(modified_weather_df, col_df, on=date_columns, how='left')
            modified_weather_df['tpm'] = modified_weather_df['tpm'].fillna(default_val)
        else:
            modified_weather_df[col] = default_val

    modified_weather_df.insert(0, 'file', 'weather')
    modified_weather_df['year'] = modified_weather_df.pop('Year')
    modified_weather_df['month'] = modified_weather_df.pop('Month')
    modified_weather_df['day'] = modified_weather_df.pop('Day')

    modified_weather_df = modified_weather_df[
        ['file', 'year', 'month', 'day', 'doy', 'temp_min', 'temp_max',
         'radiation', 'etp', 'rain', 'wind', 'tpm', 'co2']
    ]

    cols_to_clean = ['radiation', 'etp', 'tpm', 'co2', 'temp_min', 'temp_max']
    modified_weather_df[cols_to_clean] = modified_weather_df[cols_to_clean].replace(0, -999.9)

    modified_weather_df = modified_weather_df.astype({
        'year': int, 'month': int, 'day': int, 'doy': int,
        'temp_min': float, 'temp_max': float, 'radiation': float,
        'etp': float, 'rain': float, 'wind': float, 'tpm': float, 'co2': float
    })

    modified_weather_df.to_csv(output_file, index=False, header=False, sep=' ')
    print(f"Saved: {output_file}")

def process_weather_with_ghi(output_file):
    modified_weather_df = daily_weather.copy()
    modified_weather_df['Year'] = 2023
    modified_weather_df['doy'] = -1
    modified_weather_df['radiation'] = -999.9

    for idx, row in modified_weather_df.iterrows():
        date = pd.Timestamp(year=int(row['Year']), month=int(row['Month']), day=int(row['Day']))
        doy = date.timetuple().tm_yday
        modified_weather_df.at[idx, 'doy'] = doy
        modified_weather_df.at[idx, 'radiation'] = get_daily_ghi(doy - 1)

    additional_cols = {
        'etp': -999.9,
        'rain': -999.9,
        'wind': -999.9,
        'tpm': -999.9,
        'co2': -999.99
    }

    for col, default_val in additional_cols.items():
        if col == 'wind' and 'Wind Speed' in weather_df.columns:
            col_df = weather_df.groupby(date_columns).agg({'Wind Speed': 'mean'}).reset_index()
            col_df.rename(columns={'Wind Speed': 'wind'}, inplace=True)
            modified_weather_df = pd.merge(modified_weather_df, col_df, on=date_columns, how='left')
            modified_weather_df['wind'] = modified_weather_df['wind'].fillna(default_val)
        elif col == 'tpm' and 'Temperature' in weather_df.columns:
            col_df = weather_df.groupby(date_columns).agg({'Temperature': 'mean'}).reset_index()
            col_df.rename(columns={'Temperature': 'tpm'}, inplace=True)
            modified_weather_df = pd.merge(modified_weather_df, col_df, on=date_columns, how='left')
            modified_weather_df['tpm'] = modified_weather_df['tpm'].fillna(default_val)
        else:
            modified_weather_df[col] = default_val

    modified_weather_df.insert(0, 'file', 'weather')
    modified_weather_df['year'] = modified_weather_df.pop('Year')
    modified_weather_df['month'] = modified_weather_df.pop('Month')
    modified_weather_df['day'] = modified_weather_df.pop('Day')

    modified_weather_df = modified_weather_df[
        ['file', 'year', 'month', 'day', 'doy', 'temp_min', 'temp_max',
         'radiation', 'etp', 'rain', 'wind', 'tpm', 'co2']
    ]

    cols_to_clean = ['radiation', 'etp', 'tpm', 'co2', 'temp_min', 'temp_max']
    modified_weather_df[cols_to_clean] = modified_weather_df[cols_to_clean].replace(0, -999.9)

    modified_weather_df = modified_weather_df.astype({
        'year': int, 'month': int, 'day': int, 'doy': int,
        'temp_min': float, 'temp_max': float, 'radiation': float,
        'etp': float, 'rain': float, 'wind': float, 'tpm': float, 'co2': float
    })

    modified_weather_df.to_csv(output_file, index=False, header=False, sep=' ')
    print(f"Saved: {output_file}")

process_weather_with_irradiance(43, "row_43_irradiance.txt")
process_weather_with_irradiance(61, "row_61_irradiance.txt")
process_weather_with_irradiance(79, "row_79_irradiance.txt")

process_weather_with_ghi("row_ghi_irradiance.txt")
