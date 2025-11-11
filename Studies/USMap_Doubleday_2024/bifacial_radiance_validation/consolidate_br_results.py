"""
Consolidate Bifacial Radiance Results Script
This script consolidates bifacial radiance validation data
from the validation_results folder structure: setup -> GID -> day folders -> results -> CSV files

Usage as Python function:
    from consolidate_br_results import consolidate_br_results
    data = consolidate_br_results("validation_results", 
                                   base_path="bifacial_radiance_validation")

Usage from command line:
    python consolidate_br_results.py validation_results
    python consolidate_br_results.py validation_results --base-path "/scratch/kdoubled/"
    python consolidate_br_results.py validation_results --output output_file.csv

The function returns a pandas DataFrame with columns:
    - gid: GID location identifier
    - setup: Setup number (1-10)
    - datetime: Hourly datetime
    - x: Location coordinate
    - Wm2Front: Irradiance value
    
    Note: Only timestamps that exist in the CSV files are included (no grid completion).
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import re
import warnings
import argparse
import sys


def consolidate_br_results(folder_name, base_path="."):
    """
    Consolidate bifacial radiance results from validation_results folder.
    
    Parameters
    ----------
    folder_name : str
        Name of the folder (e.g., "validation_results")
    base_path : str, default "."
        Base path to the folder. Defaults to current directory.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: gid, setup, datetime, x, Wm2Front
        Contains only timestamps that exist in the CSV files.
    
    Examples
    --------
    >>> data = consolidate_br_results("validation_results")
    """
    # Construct full path to folder
    folder_path = Path(base_path) / folder_name
    
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Get all setup folders (should be numbered 1-10)
    setup_folders = [d for d in folder_path.iterdir() if d.is_dir()]
    setup_numbers = []
    
    for setup_folder in setup_folders:
        try:
            setup_num = int(setup_folder.name)
            setup_numbers.append(setup_num)
        except ValueError:
            continue
    
    setup_numbers = sorted(setup_numbers)
    
    if len(setup_numbers) == 0:
        raise ValueError("No setup folders found")
    
    print(f"Found {len(setup_numbers)} setup folders")
    
    # Initialize list to store data from each setup/GID combination
    all_data = []
    
    # Process each setup
    for setup_num in setup_numbers:
        print(f"Processing setup {setup_num}...")
        
        setup_path = folder_path / str(setup_num)
        
        # Get all GID folders within this setup
        gid_folders = [d for d in setup_path.iterdir() if d.is_dir()]
        
        # Process each GID folder
        for gid_folder in gid_folders:
            gid = gid_folder.name
            print(f"  Processing GID {gid}...")
            
            # Get all day folders within this GID folder
            # Pattern: {GID}_setup_{setup}_YYYY-MM-DD__00_00_00
            day_folders = [d for d in gid_folder.iterdir() if d.is_dir()]
            
            if len(day_folders) == 0:
                warnings.warn(f"No day folders found in: {gid_folder}")
                continue
            
            # Process each day folder
            for day_folder in day_folders:
                results_path = day_folder / "results"
                
                if not results_path.exists() or not results_path.is_dir():
                    warnings.warn(f"Results folder not found: {results_path}")
                    continue
                
                # Find all Ground CSV files
                csv_files = list(results_path.glob("*Ground*.csv"))
                
                if len(csv_files) == 0:
                    warnings.warn(f"No Ground CSV files found in: {results_path}")
                    continue
                
                # Process each CSV file
                for csv_file in csv_files:
                    # Extract datetime from filename
                    # Pattern: irr_1axis_YYYY-MM-DD_HHMMGround_Scene0_Row4_Module10_Front.csv
                    filename = csv_file.name
                    
                    # Extract date and time from filename
                    datetime_match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{4})", filename)
                    
                    if datetime_match is None:
                        warnings.warn(f"Could not extract datetime from: {filename}")
                        continue
                    
                    # Parse datetime (format: YYYY-MM-DD_HHMM)
                    datetime_str = datetime_match.group(1)
                    date_part = re.search(r"\d{4}-\d{2}-\d{2}", datetime_str).group(0)
                    time_part = re.search(r"\d{4}$", datetime_str).group(0)
                    hour = int(time_part[:2])
                    minute = int(time_part[2:])
                    
                    # Create datetime object
                    file_datetime = datetime.strptime(
                        f"{date_part} {hour:02d}:{minute:02d}", 
                        "%Y-%m-%d %H:%M"
                    )
                    
                    # Read CSV file
                    try:
                        csv_data = pd.read_csv(
                            csv_file,
                            dtype={
                                'x': float,
                                'y': float,
                                'z': float,
                                'mattype': str,
                                'Wm2Front': float
                            }
                        )
                        
                        # Add metadata columns
                        csv_data['gid'] = int(gid)
                        csv_data['setup'] = setup_num
                        csv_data['datetime'] = file_datetime
                        
                        # Reorder columns, drop mattype, y, and z
                        csv_data = csv_data[['gid', 'setup', 'datetime', 'x', 'Wm2Front']]
                        
                        all_data.append(csv_data)
                        
                    except Exception as e:
                        warnings.warn(f"Error reading {csv_file}: {e}")
    
    # Combine all data
    if len(all_data) == 0:
        raise ValueError("No data loaded. Check folder structure and file names.")
    
    combined_data = pd.concat(all_data, ignore_index=True)
    
    print(f"\nLoaded {len(combined_data)} rows of data")
    print(f"Found {combined_data['gid'].nunique()} GIDs")
    print(f"Found {combined_data['setup'].nunique()} setups")
    print(f"Date range: {combined_data['datetime'].min()} to {combined_data['datetime'].max()}")
    
    # Sort the data and return only timestamps that exist
    final_data = combined_data.sort_values(['gid', 'setup', 'datetime', 'x']).reset_index(drop=True)
    
    print(f"\nTotal rows: {len(final_data)}")
    
    return final_data


def main():
    """Command-line interface for consolidating bifacial radiance results."""
    parser = argparse.ArgumentParser(
        description='Consolidate bifacial radiance validation data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python consolidate_br_results.py validation_results
  python consolidate_br_results.py validation_results --base-path /path/to/folder
  python consolidate_br_results.py validation_results --output output_file.csv
        """
    )
    
    parser.add_argument(
        'folder_name',
        type=str,
        help='Name of the folder to process (e.g., "validation_results")'
    )
    
    parser.add_argument(
        '--base-path',
        type=str,
        default='.',
        help='Base path to the folder (defaults to current directory)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file name (defaults to "<folder_name>_data.csv")'
    )
    
    args = parser.parse_args()
    
    # Set default output filename if not provided
    output_file = args.output if args.output else f"{args.folder_name}_data.csv"
    
    print("Consolidating bifacial radiance results...")
    print(f"Folder: {args.folder_name}")
    print(f"Base path: {args.base_path}")
    print(f"Output file: {output_file}\n")
    
    # Consolidate the data
    data = consolidate_br_results(
        args.folder_name,
        base_path=args.base_path
    )
    
    # Write to CSV
    print(f"\nWriting data to {output_file}...")
    data.to_csv(output_file, index=False)
    
    file_size = Path(output_file).stat().st_size
    print(f"Done! Data exported to {output_file}")
    print(f"File size: {file_size:,} bytes")


if __name__ == "__main__":
    main()

