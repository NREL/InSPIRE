import pandas as pd

# File paths
weather_file = "PSM3_TMY.csv"
irradiance_file = "daily_test_data.csv"

# Detect the correct header row
with open(weather_file, 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines[:10]):  
        if 'Year' in line and 'Month' in line and 'Day' in line:
            header_row = i
            break
    else:
        raise ValueError("Could not detect the header row in the weather file.")

# Load weather data using detected header row
weather_df = pd.read_csv(weather_file, skiprows=header_row)

# Print actual column names for debugging
print("Weather file columns:", weather_df.columns)

# Load irradiance data
dirrad_df = pd.read_csv(irradiance_file, header=None, skiprows=1)  


date_columns = ['Year', 'Month', 'Day']
temp_column = 'Tdry' 

if not all(col in weather_df.columns for col in date_columns):
    raise ValueError("Could not find Year, Month, and Day columns in the weather file.")

if temp_column not in weather_df.columns:
    raise ValueError("Could not find the temperature column in the weather file.")

# Aggregate to daily min/max temperatures
daily_weather = weather_df.groupby(date_columns).agg(
    Tmin=(temp_column, 'min'),  
    Tmax=(temp_column, 'max')   
).reset_index()

# Function to get irradiance values from a specific row
def get_daily_irradiance(row_index, day_of_year):
    """ Extracts the correct irradiance value for a given day of the year from a specific row """
    if 0 <= day_of_year < dirrad_df.shape[1]:
        if row_index + 1 >= dirrad_df.shape[0]:
            return None  
    return float(dirrad_df.iloc[row_index + 1, day_of_year + 1])   


# Function to process and save weather data with specific irradiance row
def process_weather_with_irradiance(row_index, output_file):
    modified_weather_df = daily_weather.copy()
    modified_weather_df['DNI'] = -999.9 
    
    for index, row in modified_weather_df.iterrows():
        date = pd.Timestamp(year=int(row['Year']), month=int(row['Month']), day=int(row['Day']))
        day_of_year = date.timetuple().tm_yday - 1  
        irradiance_value = get_daily_irradiance(row_index, day_of_year)
        if irradiance_value is not None:
            modified_weather_df.at[index, 'DNI'] = irradiance_value  
    
    # Replace 0 values with -999
    modified_weather_df.replace(0, -999.9, inplace=True)
    
    # Save the processed file
    modified_weather_df.to_csv(output_file, index=False)
    print("Processed file saved as", output_file)

# Process and save three weather files with different irradiance sources
process_weather_with_irradiance(0, "weather_with_irradiance_1.csv")  
process_weather_with_irradiance(50, "weather_with_irradiance_2.csv")  
process_weather_with_irradiance(98, "weather_with_irradiance_3.csv")  
