import os
import glob
import pandas as pd

# Initialize an empty DataFrame to store the results
final_df = pd.DataFrame()

# Define the root directory for your simulations
root_dir = r'/projects/inspire/HSAT_AGU24/'

columns_to_keep = ['x', 'mattype', 'Wm2Front']
ii = 0
# Traverse through all 'Days' folders (using glob for timestamp-based folder names)
for day_folder in sorted(glob.glob(os.path.join(root_dir, '*'))):
    # Check if the 'results' folder exists in the current day folder
    results_folder = os.path.join(day_folder, 'results')
    
    if os.path.isdir(results_folder):
         # Iterate over all CSV files that end with '_Front.csv' in the 'results' folder
        for csv_file in sorted(glob.glob(os.path.join(results_folder, '*_Front.csv'))):
            # Read the CSV into a DataFrame
            day_df = pd.read_csv(csv_file)
            
            # Keep only the desired columns if they exist in the DataFrame
            day_df = day_df[day_df.columns.intersection(columns_to_keep)]

            # Remove the last 8 characters of the day folder name and save it to the 'day' column
            day_df['date'] = os.path.basename(day_folder)[:-10]  # Remove last 8 characters
            
            # Save the first 10 characters of the CSV file name to a new column 'csv_name'
            day_df['hour'] = os.path.basename(csv_file)[21:23]  # Selecting only the hour characters of the filename

            # Save the first 10 characters of the CSV file name to a new column 'csv_name'
            day_df['sensor'] = os.path.basename(csv_file)[33:34]  # Selecting only the hour characters of the filename

            # Append the data to the final DataFrame
            final_df = pd.concat([final_df, day_df], ignore_index=True)

        #ii += 1
        #if ii >= 24:
        #    break
        
# Now, final_df contains all the data from the '_Front.csv' files in each Day folder
# You can save or process the data as needed
print(final_df.head())  # Preview the data
final_df.to_csv('COMPILED_RESULTS_07Dec24_sorted.csv')