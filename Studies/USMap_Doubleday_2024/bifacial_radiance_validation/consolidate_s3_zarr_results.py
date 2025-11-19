"""
Consolidate S3 Zarr Results Script
This script consolidates agrivoltaics irradiance data from S3 zarr files
and filters to only include GIDs found in the validation_results folder.

The script reads from:
    s3://oedi-data-lake/inspire/agrivoltaics_irradiance/v1.1/preliminary_XX.zarr
    
Where XX is the setup number (01-10).

Usage as Python function:
    from consolidate_s3_zarr_results import consolidate_s3_zarr_results
    data = consolidate_s3_zarr_results("validation_results", 
                                        base_path="bifacial_radiance_validation")

Usage from command line:
    python consolidate_s3_zarr_results.py validation_results
    python consolidate_s3_zarr_results.py validation_results --base-path "/path/to/folder"
    python consolidate_s3_zarr_results.py validation_results --output pysam_outputs.csv

The function returns a pandas DataFrame with columns:
    - gid: GID location identifier
    - setup: Setup number (1-10)
    - datetime: Hourly datetime
    - x: Location coordinate (mapped from distance dimension)
    - Wm2Front: Irradiance value (mapped from ground_irradiance)
    
    Note: Includes all timestamps from the zarr files for the GIDs in validation_results.
"""

import pandas as pd
import xarray as xr
import fsspec
from pathlib import Path
import warnings
import argparse
import numpy as np


def get_validation_gids(folder_name, base_path="."):
    """
    Extract the list of unique GIDs from the validation_results folder structure.
    
    Parameters
    ----------
    folder_name : str
        Name of the folder (e.g., "validation_results")
    base_path : str, default "."
        Base path to the folder. Defaults to current directory.
    
    Returns
    -------
    set
        Set of unique GID values (as integers)
    """
    folder_path = Path(base_path) / folder_name
    
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Get all setup folders (should be numbered 1-10)
    setup_folders = [d for d in folder_path.iterdir() if d.is_dir()]
    gids = set()
    
    for setup_folder in setup_folders:
        try:
            setup_num = int(setup_folder.name)
        except ValueError:
            continue
        
        setup_path = folder_path / str(setup_num)
        
        # Get all GID folders within this setup
        gid_folders = [d for d in setup_path.iterdir() if d.is_dir()]
        
        for gid_folder in gid_folders:
            try:
                gid = int(gid_folder.name)
                gids.add(gid)
            except ValueError:
                continue
    
    return gids


def consolidate_s3_zarr_results(folder_name, base_path=".", s3_bucket_path="oedi-data-lake/inspire/agrivoltaics_irradiance/v1.1"):
    """
    Consolidate agrivoltaics irradiance data from S3 zarr files.
    
    Parameters
    ----------
    folder_name : str
        Name of the validation_results folder (e.g., "validation_results")
    base_path : str, default "."
        Base path to the validation_results folder. Defaults to current directory.
    s3_bucket_path : str, default "oedi-data-lake/inspire/agrivoltaics_irradiance/v1.1"
        S3 path to the zarr files directory
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: gid, setup, datetime, x, Wm2Front
        Contains all timestamps from zarr files for GIDs in validation_results.
    
    Examples
    --------
    >>> data = consolidate_s3_zarr_results("validation_results")
    """
    # Get list of GIDs from validation_results
    print("Extracting GIDs from validation_results folder...")
    validation_gids = get_validation_gids(folder_name, base_path=base_path)
    validation_gids = sorted(list(validation_gids))
    
    if len(validation_gids) == 0:
        raise ValueError("No GIDs found in validation_results folder")
    
    print(f"Found {len(validation_gids)} unique GIDs: {validation_gids}")
    
    # Initialize list to store data from each setup
    all_data = []
    
    # Process each setup (1-10)
    for setup_num in range(1, 11):
        print(f"\nProcessing setup {setup_num}...")
        
        # Construct zarr file path
        zarr_filename = f"preliminary_{setup_num:02d}.zarr"
        zarr_path = f"s3://{s3_bucket_path}/{zarr_filename}"
        
        print(f"  Opening: {zarr_path}")
        
        try:
            # Create fsspec mapper for S3
            mapper = fsspec.get_mapper(zarr_path, anon=True)
            
            # Open zarr dataset
            ds = xr.open_zarr(mapper)
            
            print(f"  Dataset dimensions: {dict(ds.sizes)}")
            print(f"  Available variables: {list(ds.data_vars)}")
            
            # Check if ground_irradiance exists
            if 'ground_irradiance' not in ds.data_vars:
                warnings.warn(f"ground_irradiance not found in setup {setup_num}")
                continue
            
            # Get ground_irradiance data
            ground_irr = ds['ground_irradiance']
            
            # Get GIDs from the dataset
            dataset_gids = ds['gid'].values
            
            # Find which validation GIDs are in this dataset
            gid_mask = np.isin(dataset_gids, validation_gids)
            matching_gids = dataset_gids[gid_mask]
            
            if len(matching_gids) == 0:
                print(f"  No matching GIDs found in setup {setup_num}")
                continue
            
            print(f"  Found {len(matching_gids)} matching GIDs")
            
            # Get indices of matching GIDs
            gid_indices = np.where(gid_mask)[0]
            
            # Select data for matching GIDs
            # ground_irradiance has dimensions (gid, time, distance)
            selected_data = ground_irr.isel(gid=gid_indices)
            
            # Convert to DataFrame
            # Stack distance and time to create a long format
            df = selected_data.to_dataframe(name='Wm2Front').reset_index()
            
            # Rename columns to match consolidate_br_results format
            df = df.rename(columns={
                'distance': 'x',
                'gid': 'gid'
            })
            
            # Add setup column
            df['setup'] = setup_num
            
            # Reorder columns: gid, setup, datetime (time), x, Wm2Front
            df = df[['gid', 'setup', 'time', 'x', 'Wm2Front']]
            
            # Rename time to datetime
            df = df.rename(columns={'time': 'datetime'})
            
            # Convert datetime to pandas datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
                df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Convert gid to int
            df['gid'] = df['gid'].astype(int)
            
            # Convert x to float (distance is int32 in zarr, but we want float like in consolidate_br_results)
            df['x'] = df['x'].astype(float)
            
            # Convert Wm2Front to float
            df['Wm2Front'] = df['Wm2Front'].astype(float)
            
            # Remove any NaN values
            df = df.dropna(subset=['Wm2Front'])
            
            print(f"  Extracted {len(df)} rows")
            
            all_data.append(df)
            
        except Exception as e:
            warnings.warn(f"Error processing setup {setup_num}: {e}")
            continue
    
    # Combine all data
    if len(all_data) == 0:
        raise ValueError("No data loaded. Check S3 access and zarr file structure.")
    
    combined_data = pd.concat(all_data, ignore_index=True)
    
    print(f"\n\nLoaded {len(combined_data)} rows of data")
    print(f"Found {combined_data['gid'].nunique()} GIDs")
    print(f"Found {combined_data['setup'].nunique()} setups")
    print(f"Date range: {combined_data['datetime'].min()} to {combined_data['datetime'].max()}")
    
    # Sort the data
    final_data = combined_data.sort_values(['gid', 'setup', 'datetime', 'x']).reset_index(drop=True)
    
    print(f"\nTotal rows: {len(final_data)}")
    
    return final_data


def main():
    """Command-line interface for consolidating S3 zarr results."""
    parser = argparse.ArgumentParser(
        description='Consolidate agrivoltaics irradiance data from S3 zarr files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python consolidate_s3_zarr_results.py validation_results
  python consolidate_s3_zarr_results.py validation_results --base-path /path/to/folder
  python consolidate_s3_zarr_results.py validation_results --output output_file.csv
        """
    )
    
    parser.add_argument(
        'folder_name',
        type=str,
        help='Name of the validation_results folder (e.g., "validation_results")'
    )
    
    parser.add_argument(
        '--base-path',
        type=str,
        default='.',
        help='Base path to the validation_results folder (defaults to current directory)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file name (defaults to "<folder_name>_s3_data.csv")'
    )
    
    parser.add_argument(
        '--s3-path',
        type=str,
        default='oedi-data-lake/inspire/agrivoltaics_irradiance/v1.1',
        help='S3 path to zarr files directory (default: oedi-data-lake/inspire/agrivoltaics_irradiance/v1.1)'
    )
    
    args = parser.parse_args()
    
    # Set default output filename if not provided
    output_file = args.output if args.output else f"{args.folder_name}_s3_data.csv"
    
    print("Consolidating S3 zarr results...")
    print(f"Validation results folder: {args.folder_name}")
    print(f"Base path: {args.base_path}")
    print(f"S3 path: {args.s3_path}")
    print(f"Output file: {output_file}\n")
    
    # Consolidate the data
    data = consolidate_s3_zarr_results(
        args.folder_name,
        base_path=args.base_path,
        s3_bucket_path=args.s3_path
    )
    
    # Write to CSV
    print(f"\nWriting data to {output_file}...")
    data.to_csv(output_file, index=False)
    
    file_size = Path(output_file).stat().st_size
    print(f"Done! Data exported to {output_file}")
    print(f"File size: {file_size:,} bytes")


if __name__ == "__main__":
    main()


