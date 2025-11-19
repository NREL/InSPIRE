"""
Compare Validation Results and PySAM Outputs
This script compares two solar irradiance datasets from a combined CSV file:
- Bifacial Radiance validation data (half-hour timestamps)
- PySAM model outputs from S3 zarr (hourly timestamps)

The combined CSV file should have a 'data_source' column indicating 'bifacial_radiance' or 'pysam'.

The script:
- Determines the most likely time alignment between datasets
- Matches data by gid, setup, and x (distance) coordinates
- Filters to only sun-up times (times present in bifacial_radiance data)
- Calculates summary statistics: MBD, RMSE (both percentage and absolute)

Usage:
    python compare_datasets.py
    python compare_datasets.py --data-file all_results.csv
    python compare_datasets.py --output comparison_results.csv
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import warnings


def MBD(meas, model):
    """
    Mean Bias Difference (percentage)
    MBD = 100 * [((1/m) * sum(y_i - x_i)) / mean(x_i)]
    
    Parameters
    ----------
    meas : array-like
        Measured values (validation data)
    model : array-like
        Modeled values (pysam outputs)
    
    Returns
    -------
    float
        Mean Bias Difference as percentage
    """
    df = pd.DataFrame({'model': model, 'meas': meas})
    
    # Remove NaN values
    df = df.dropna()
    
    if len(df) == 0:
        return np.nan
    
    # Filter: model must be greater than minimum measured irradiance
    minirr = df['meas'].min()
    df = df[df['model'] > minirr]
    
    if len(df) == 0:
        return np.nan
    
    m = len(df)
    
    # Calculate MBD
    out = 100 * ((1/m) * (df['model'] - df['meas']).sum()) / df['meas'].mean()
    
    return out


def RMSE(meas, model):
    """
    Root Mean Squared Error (percentage)
    RMSE = 100 * sqrt((1/m) * sum((y_i - x_i)^2)) / mean(x_i)
    
    Parameters
    ----------
    meas : array-like
        Measured values (validation data)
    model : array-like
        Modeled values (pysam outputs)
    
    Returns
    -------
    float
        Root Mean Squared Error as percentage
    """
    df = pd.DataFrame({'model': model, 'meas': meas})
    
    # Remove NaN values
    df = df.dropna()
    
    if len(df) == 0:
        return np.nan
    
    # Filter: model must be greater than minimum measured irradiance
    minirr = df['meas'].min()
    df = df[df['model'] > minirr]
    
    if len(df) == 0:
        return np.nan
    
    m = len(df)
    
    # Calculate RMSE
    out = 100 * np.sqrt((1/m) * ((df['model'] - df['meas'])**2).sum()) / df['meas'].mean()
    
    return out


def MBD_abs(meas, model):
    """
    Mean Bias Difference (absolute, not percentage)
    MBD = (1/m) * sum(y_i - x_i)
    
    Parameters
    ----------
    meas : array-like
        Measured values (validation data)
    model : array-like
        Modeled values (pysam outputs)
    
    Returns
    -------
    float
        Mean Bias Difference (absolute)
    """
    df = pd.DataFrame({'model': model, 'meas': meas})
    
    # Remove NaN values
    df = df.dropna()
    
    if len(df) == 0:
        return np.nan
    
    # Filter: model must be greater than minimum measured irradiance
    minirr = df['meas'].min()
    df = df[df['model'] > minirr]
    
    if len(df) == 0:
        return np.nan
    
    m = len(df)
    
    # Calculate MBD (absolute)
    out = (1/m) * (df['model'] - df['meas']).sum()
    
    return out


def RMSE_abs(meas, model):
    """
    Root Mean Squared Error (absolute, not percentage)
    RMSE = sqrt((1/m) * sum((y_i - x_i)^2))
    
    Parameters
    ----------
    meas : array-like
        Measured values (validation data)
    model : array-like
        Modeled values (pysam outputs)
    
    Returns
    -------
    float
        Root Mean Squared Error (absolute)
    """
    df = pd.DataFrame({'model': model, 'meas': meas})
    
    # Remove NaN values
    df = df.dropna()
    
    if len(df) == 0:
        return np.nan
    
    # Filter: model must be greater than minimum measured irradiance
    minirr = df['meas'].min()
    df = df[df['model'] > minirr]
    
    if len(df) == 0:
        return np.nan
    
    m = len(df)
    
    # Calculate RMSE (absolute)
    out = np.sqrt((1/m) * ((df['model'] - df['meas'])**2).sum())
    
    return out


def find_time_alignment(validation_times, pysam_times):
    """
    Determine the most likely time alignment between validation (half-hour) 
    and pysam (hourly) timestamps.
    
    The validation times are on the half-hour (e.g., 06:30, 07:30).
    The pysam times are on the hour (e.g., 06:00, 07:00).
    
    We test different offsets to find the best alignment.
    Since years are arbitrary, we match by day-of-year and hour.
    
    Parameters
    ----------
    validation_times : pd.Series
        Validation datetime series
    pysam_times : pd.Series
        PySAM datetime series
    
    Returns
    -------
    float
        Best offset in hours (-0.5 or +0.5, since validation is on :30)
    """
    # Convert to datetime if needed
    val_times = pd.to_datetime(validation_times)
    pysam_times = pd.to_datetime(pysam_times)
    
    # Extract day of year and hour
    val_doy = val_times.dt.dayofyear
    val_hour = val_times.dt.hour
    val_minute = val_times.dt.minute
    
    pysam_doy = pysam_times.dt.dayofyear
    pysam_hour = pysam_times.dt.hour
    
    # Validation times are on :30, pysam on :00
    # Test two alignments: -0.5 hour (match to previous hour) or +0.5 hour (match to next hour)
    offsets = [-0.5, 0.5]
    best_offset = -0.5
    best_score = -np.inf
    
    for offset in offsets:
        # Shift validation hour by offset
        # -0.5 means shift down by 1 hour (06:30 -> 06:00)
        # +0.5 means shift up by 1 hour (06:30 -> 07:00)
        if offset < 0:
            shifted_val_hour = val_hour - 1
        else:
            shifted_val_hour = val_hour + 1
        # Handle wrap-around
        shifted_val_hour = shifted_val_hour % 24
        
        # Count how many validation times can be matched to pysam times
        matches = 0
        for i, (doy, hour) in enumerate(zip(val_doy, shifted_val_hour)):
            # Find matching pysam times (same day of year and hour)
            pysam_match = (pysam_doy == doy) & (pysam_hour == hour)
            if pysam_match.sum() > 0:
                matches += 1
        
        score = matches / len(val_times)
        
        if score > best_score:
            best_score = score
            best_offset = offset
    
    return best_offset


def match_x_values(validation_x, pysam_x, validation_irr, pysam_irr):
    """
    Match x (distance) values between validation and pysam datasets.
    Validation has irregular x values, pysam has regular integer x values.
    Uses interpolation to match validation x values to nearest pysam x values.
    
    Parameters
    ----------
    validation_x : array-like
        Validation x coordinates
    pysam_x : array-like
        PySAM x coordinates (sorted)
    validation_irr : array-like
        Validation irradiance values
    pysam_irr : array-like
        PySAM irradiance values
    
    Returns
    -------
    tuple
        (matched_validation_irr, matched_pysam_irr) arrays
    """
    # Convert to numpy arrays
    val_x = np.array(validation_x)
    pysam_x_sorted = np.sort(np.unique(pysam_x))
    val_irr = np.array(validation_irr)
    pysam_irr = np.array(pysam_irr)
    
    # Create mapping: for each validation x, find closest pysam x
    matched_val_irr = []
    matched_pysam_irr = []
    
    for vx, virr in zip(val_x, val_irr):
        # Find closest pysam x value
        closest_idx = np.argmin(np.abs(pysam_x_sorted - vx))
        closest_pysam_x = pysam_x_sorted[closest_idx]
        
        # Find pysam irradiance at this x value
        pysam_mask = (pysam_x == closest_pysam_x)
        if pysam_mask.sum() > 0:
            # If multiple matches, take the first (should be same value)
            pysam_irr_val = pysam_irr[pysam_mask][0]
            matched_val_irr.append(virr)
            matched_pysam_irr.append(pysam_irr_val)
    
    return np.array(matched_val_irr), np.array(matched_pysam_irr)


def compare_datasets(data_file='all_results.csv', output_file=None):
    """
    Compare bifacial radiance and pysam outputs from a combined CSV file.
    
    Parameters
    ----------
    data_file : str
        Path to combined results CSV file with data_source column
    output_file : str, optional
        Path to output CSV file for summary statistics
    
    Returns
    -------
    pd.DataFrame
        Summary statistics table
    """
    print("Loading dataset...")
    
    # Load combined dataset
    all_data = pd.read_csv(data_file)
    
    # Convert datetime columns
    all_data['datetime'] = pd.to_datetime(all_data['datetime'])
    
    # Split into bifacial_radiance and pysam datasets
    validation = all_data[all_data['data_source'] == 'bifacial_radiance'].copy()
    pysam = all_data[all_data['data_source'] == 'pysam'].copy()
    
    print(f"Bifacial Radiance data: {len(validation)} rows, {validation['datetime'].nunique()} unique times")
    print(f"PySAM data: {len(pysam)} rows, {pysam['datetime'].nunique()} unique times")
    
    # Extract day of year and hour (ignoring year)
    validation['dayofyear'] = validation['datetime'].dt.dayofyear
    validation['hour'] = validation['datetime'].dt.hour
    validation['minute'] = validation['datetime'].dt.minute
    
    pysam['dayofyear'] = pysam['datetime'].dt.dayofyear
    pysam['hour'] = pysam['datetime'].dt.hour
    
    # Get unique validation times (these are the "sun-up" times)
    validation_times = validation['datetime'].unique()
    print(f"\nValidation time range: {validation_times.min()} to {validation_times.max()}")
    print(f"Validation day of year range: {validation['dayofyear'].min()} to {validation['dayofyear'].max()}")
    print(f"Validation hours: {sorted(validation['hour'].unique())}")
    
    # Determine time alignment
    print("\nDetermining time alignment...")
    time_offset = find_time_alignment(validation['datetime'], pysam['datetime'])
    print(f"Best time offset: {time_offset} hours")
    
    # Adjust validation hour by offset to match pysam times
    validation_adjusted = validation.copy()
    # Shift hour by the offset (validation is on :30, need to match to :00)
    # -0.5 means shift down by 1 hour (06:30 -> 06:00)
    # +0.5 means shift up by 1 hour (06:30 -> 07:00)
    if time_offset < 0:
        validation_adjusted['hour_adjusted'] = validation_adjusted['hour'] - 1
    else:
        validation_adjusted['hour_adjusted'] = validation_adjusted['hour'] + 1
    # Handle wrap-around
    validation_adjusted['hour_adjusted'] = validation_adjusted['hour_adjusted'] % 24
    
    # Get unique combinations of gid, setup
    gid_setup_combos = validation[['gid', 'setup']].drop_duplicates()
    print(f"\nFound {len(gid_setup_combos)} unique (gid, setup) combinations")
    
    # Initialize results list
    results = []
    distance_stats_data = []  # Store data for distance-based statistics
    
    # Process each (gid, setup) combination
    for idx, row in gid_setup_combos.iterrows():
        gid = row['gid']
        setup = row['setup']
        
        print(f"\nProcessing GID {gid}, Setup {setup}...")
        
        # Filter data for this gid and setup
        val_subset = validation_adjusted[
            (validation_adjusted['gid'] == gid) & 
            (validation_adjusted['setup'] == setup)
        ].copy()
        
        pysam_subset = pysam[
            (pysam['gid'] == gid) & 
            (pysam['setup'] == setup)
        ].copy()
        
        if len(val_subset) == 0 or len(pysam_subset) == 0:
            warnings.warn(f"No data for GID {gid}, Setup {setup}")
            continue
        
        # Match by day of year, hour, and x
        matched_data = []
        
        # Get unique combinations of day of year and adjusted hour
        for (doy, hour_adj), val_group in val_subset.groupby(['dayofyear', 'hour_adjusted']):
            # Find corresponding pysam data (same day of year and hour)
            pysam_time_data = pysam_subset[
                (pysam_subset['dayofyear'] == doy) & 
                (pysam_subset['hour'] == hour_adj)
            ]
            
            if len(pysam_time_data) == 0:
                continue
            
            # Match x values
            for _, val_row in val_group.iterrows():
                val_x = val_row['x']
                val_irr = val_row['Wm2Front']
                
                # Find closest pysam x value
                pysam_x_values = pysam_time_data['x'].values
                closest_x_idx = np.argmin(np.abs(pysam_x_values - val_x))
                closest_x = pysam_x_values[closest_x_idx]
                
                # Get pysam irradiance at closest x
                pysam_match = pysam_time_data[pysam_time_data['x'] == closest_x]
                if len(pysam_match) > 0:
                    pysam_irr = pysam_match['Wm2Front'].iloc[0]
                    
                    matched_data.append({
                        'gid': gid,
                        'setup': setup,
                        'dayofyear': doy,
                        'hour': hour_adj,
                        'x': val_x,
                        'validation_irr': val_irr,
                        'pysam_irr': pysam_irr
                    })
        
        if len(matched_data) == 0:
            warnings.warn(f"No matched data for GID {gid}, Setup {setup}")
            continue
        
        matched_df = pd.DataFrame(matched_data)
        
        # Calculate statistics for this gid/setup combination
        meas = matched_df['validation_irr'].values
        model = matched_df['pysam_irr'].values
        
        # Filter out zero or very small values for meaningful comparison
        mask = (meas > 0.1) & (model > 0.1)
        meas_filtered = meas[mask]
        model_filtered = model[mask]
        
        if len(meas_filtered) == 0:
            warnings.warn(f"No valid data points for GID {gid}, Setup {setup} after filtering")
            continue
        
        # Calculate statistics
        mbd_pct = MBD(meas_filtered, model_filtered)
        rmse_pct = RMSE(meas_filtered, model_filtered)
        mbd_abs_val = MBD_abs(meas_filtered, model_filtered)
        rmse_abs_val = RMSE_abs(meas_filtered, model_filtered)
        
        # Additional statistics
        n_points = len(meas_filtered)
        mean_meas = meas_filtered.mean()
        mean_model = model_filtered.mean()
        corr = np.corrcoef(meas_filtered, model_filtered)[0, 1]
        
        results.append({
            'gid': gid,
            'setup': setup,
            'n_points': n_points,
            'mean_validation': mean_meas,
            'mean_pysam': mean_model,
            'MBD_percent': mbd_pct,
            'RMSE_percent': rmse_pct,
            'MBD_absolute': mbd_abs_val,
            'RMSE_absolute': rmse_abs_val,
            'correlation': corr
        })
        
        print(f"  Matched {n_points} data points")
        print(f"  MBD: {mbd_pct:.2f}%, RMSE: {rmse_pct:.2f}%")
        
        # Store matched data for distance-based analysis
        matched_df_filtered = matched_df[mask].copy()
        if len(matched_df_filtered) > 0:
            distance_stats_data.append(matched_df_filtered)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(results)
    
    if len(summary_df) == 0:
        raise ValueError("No matched data found. Check data alignment.")
    
    # Add overall statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS (by GID and Setup)")
    print("="*60)
    print(summary_df.to_string(index=False))
    
    # Calculate distance-based statistics
    print("\n" + "="*60)
    print("Calculating statistics by distance (x)...")
    print("="*60)
    
    distance_results = []
    
    if len(distance_stats_data) > 0:
        # Combine all matched data
        all_matched_df = pd.concat(distance_stats_data, ignore_index=True)
        
        # Group by gid, setup, and x (distance)
        for (gid, setup, x), group in all_matched_df.groupby(['gid', 'setup', 'x']):
            meas = group['validation_irr'].values
            model = group['pysam_irr'].values
            
            # Filter out zero or very small values
            mask = (meas > 0.1) & (model > 0.1)
            meas_filtered = meas[mask]
            model_filtered = model[mask]
            
            if len(meas_filtered) == 0:
                continue
            
            # Calculate statistics
            mbd_pct = MBD(meas_filtered, model_filtered)
            rmse_pct = RMSE(meas_filtered, model_filtered)
            mbd_abs_val = MBD_abs(meas_filtered, model_filtered)
            rmse_abs_val = RMSE_abs(meas_filtered, model_filtered)
            
            # Additional statistics
            n_points = len(meas_filtered)
            mean_meas = meas_filtered.mean()
            mean_model = model_filtered.mean()
            corr = np.corrcoef(meas_filtered, model_filtered)[0, 1] if len(meas_filtered) > 1 else np.nan
            
            distance_results.append({
                'gid': gid,
                'setup': setup,
                'x': x,
                'n_points': n_points,
                'mean_validation': mean_meas,
                'mean_pysam': mean_model,
                'MBD_percent': mbd_pct,
                'RMSE_percent': rmse_pct,
                'MBD_absolute': mbd_abs_val,
                'RMSE_absolute': rmse_abs_val,
                'correlation': corr
            })
    
    distance_summary_df = pd.DataFrame(distance_results)
    
    if len(distance_summary_df) > 0:
        # Sort by gid, setup, and x
        distance_summary_df = distance_summary_df.sort_values(['gid', 'setup', 'x']).reset_index(drop=True)
        
        print("\n" + "="*60)
        print("SUMMARY STATISTICS (by GID, Setup, and Distance)")
        print("="*60)
        print(distance_summary_df.to_string(index=False))
    else:
        print("No distance-based statistics calculated.")
    
    # Calculate matched time-points per GID
    print("\n" + "="*60)
    print("MATCHED TIME-POINTS PER GID")
    print("="*60)
    
    if len(distance_stats_data) > 0:
        all_matched_df = pd.concat(distance_stats_data, ignore_index=True)
        
        # Count unique time-points (dayofyear + hour) per GID
        timepoint_counts = []
        for gid in all_matched_df['gid'].unique():
            gid_data = all_matched_df[all_matched_df['gid'] == gid]
            # Count unique combinations of dayofyear and hour
            unique_timepoints = gid_data[['dayofyear', 'hour']].drop_duplicates()
            n_timepoints = len(unique_timepoints)
            
            # Also get the range of times
            unique_days = sorted(gid_data['dayofyear'].unique())
            unique_hours = sorted(gid_data['hour'].unique())
            
            timepoint_counts.append({
                'gid': gid,
                'n_matched_timepoints': n_timepoints,
                'dayofyear_range': f"{unique_days[0]}-{unique_days[-1]}" if len(unique_days) > 1 else str(unique_days[0]),
                'hour_range': f"{unique_hours[0]}-{unique_hours[-1]}" if len(unique_hours) > 1 else str(unique_hours[0]),
                'unique_days': len(unique_days),
                'unique_hours': len(unique_hours)
            })
        
        timepoint_summary = pd.DataFrame(timepoint_counts).sort_values('gid')
        print(timepoint_summary.to_string(index=False))
        
        # Save to file
        if output_file:
            timepoint_output_file = output_file.replace('.csv', '_timepoints_per_gid.csv')
            timepoint_summary.to_csv(timepoint_output_file, index=False)
            print(f"\nTime-point summary saved to: {timepoint_output_file}")
    else:
        print("No matched data available for time-point analysis.")
    
    # Calculate statistics by hour of day
    print("\n" + "="*60)
    print("Calculating statistics by hour of day...")
    print("="*60)
    
    hour_results_overall = []
    hour_results_by_gid_setup = []
    
    if len(distance_stats_data) > 0:
        all_matched_df = pd.concat(distance_stats_data, ignore_index=True)
        
        # Overall statistics by hour (averaging across all GIDs, setups, distances)
        for hour in sorted(all_matched_df['hour'].unique()):
            hour_data = all_matched_df[all_matched_df['hour'] == hour]
            
            meas = hour_data['validation_irr'].values
            model = hour_data['pysam_irr'].values
            
            # Filter out zero or very small values
            mask = (meas > 0.1) & (model > 0.1)
            meas_filtered = meas[mask]
            model_filtered = model[mask]
            
            if len(meas_filtered) == 0:
                continue
            
            # Calculate statistics
            mbd_pct = MBD(meas_filtered, model_filtered)
            rmse_pct = RMSE(meas_filtered, model_filtered)
            mbd_abs_val = MBD_abs(meas_filtered, model_filtered)
            rmse_abs_val = RMSE_abs(meas_filtered, model_filtered)
            
            # Additional statistics
            n_points = len(meas_filtered)
            mean_meas = meas_filtered.mean()
            mean_model = model_filtered.mean()
            corr = np.corrcoef(meas_filtered, model_filtered)[0, 1] if len(meas_filtered) > 1 else np.nan
            
            hour_results_overall.append({
                'hour': hour,
                'n_points': n_points,
                'mean_validation': mean_meas,
                'mean_pysam': mean_model,
                'MBD_percent': mbd_pct,
                'RMSE_percent': rmse_pct,
                'MBD_absolute': mbd_abs_val,
                'RMSE_absolute': rmse_abs_val,
                'correlation': corr
            })
        
        # Statistics by hour, GID, and setup (averaging over distance)
        for (gid, setup, hour), group in all_matched_df.groupby(['gid', 'setup', 'hour']):
            meas = group['validation_irr'].values
            model = group['pysam_irr'].values
            
            # Filter out zero or very small values
            mask = (meas > 0.1) & (model > 0.1)
            meas_filtered = meas[mask]
            model_filtered = model[mask]
            
            if len(meas_filtered) == 0:
                continue
            
            # Calculate statistics
            mbd_pct = MBD(meas_filtered, model_filtered)
            rmse_pct = RMSE(meas_filtered, model_filtered)
            mbd_abs_val = MBD_abs(meas_filtered, model_filtered)
            rmse_abs_val = RMSE_abs(meas_filtered, model_filtered)
            
            # Additional statistics
            n_points = len(meas_filtered)
            mean_meas = meas_filtered.mean()
            mean_model = model_filtered.mean()
            corr = np.corrcoef(meas_filtered, model_filtered)[0, 1] if len(meas_filtered) > 1 else np.nan
            
            hour_results_by_gid_setup.append({
                'gid': gid,
                'setup': setup,
                'hour': hour,
                'n_points': n_points,
                'mean_validation': mean_meas,
                'mean_pysam': mean_model,
                'MBD_percent': mbd_pct,
                'RMSE_percent': rmse_pct,
                'MBD_absolute': mbd_abs_val,
                'RMSE_absolute': rmse_abs_val,
                'correlation': corr
            })
        
        # Create DataFrames
        hour_summary_overall = pd.DataFrame(hour_results_overall).sort_values('hour')
        hour_summary_by_gid_setup = pd.DataFrame(hour_results_by_gid_setup).sort_values(['gid', 'setup', 'hour'])
        
        # Print overall statistics by hour
        print("\n" + "="*60)
        print("SUMMARY STATISTICS BY HOUR (Overall - All GIDs, Setups, Distances)")
        print("="*60)
        print(hour_summary_overall.to_string(index=False))
        
        # Print statistics by hour, GID, and setup
        print("\n" + "="*60)
        print("SUMMARY STATISTICS BY HOUR, GID, AND SETUP (Averaged over Distance)")
        print("="*60)
        print(hour_summary_by_gid_setup.to_string(index=False))
        
        # Save to files
        if output_file:
            hour_overall_output = output_file.replace('.csv', '_by_hour_overall.csv')
            hour_summary_overall.to_csv(hour_overall_output, index=False)
            print(f"\nHour statistics (overall) saved to: {hour_overall_output}")
            
            hour_gid_setup_output = output_file.replace('.csv', '_by_hour_gid_setup.csv')
            hour_summary_by_gid_setup.to_csv(hour_gid_setup_output, index=False)
            print(f"Hour statistics (by GID and setup) saved to: {hour_gid_setup_output}")
    else:
        print("No matched data available for hour-based analysis.")
    
    # Calculate overall statistics across all gid/setup combinations
    all_meas = []
    all_model = []
    for _, row in summary_df.iterrows():
        gid = row['gid']
        setup = row['setup']
        
        val_subset = validation_adjusted[
            (validation_adjusted['gid'] == gid) & 
            (validation_adjusted['setup'] == setup)
        ]
        pysam_subset = pysam[
            (pysam['gid'] == gid) & 
            (pysam['setup'] == setup)
        ]
        
        # Re-match for overall stats
        for (doy, hour_adj), val_group in val_subset.groupby(['dayofyear', 'hour_adjusted']):
            pysam_time_data = pysam_subset[
                (pysam_subset['dayofyear'] == doy) & 
                (pysam_subset['hour'] == hour_adj)
            ]
            
            for _, val_row in val_group.iterrows():
                val_x = val_row['x']
                val_irr = val_row['Wm2Front']
                
                if val_irr > 0.1:
                    pysam_x_values = pysam_time_data['x'].values
                    closest_x_idx = np.argmin(np.abs(pysam_x_values - val_x))
                    closest_x = pysam_x_values[closest_x_idx]
                    pysam_match = pysam_time_data[pysam_time_data['x'] == closest_x]
                    if len(pysam_match) > 0:
                        pysam_irr = pysam_match['Wm2Front'].iloc[0]
                        if pysam_irr > 0.1:
                            all_meas.append(val_irr)
                            all_model.append(pysam_irr)
    
    if len(all_meas) > 0:
        all_meas = np.array(all_meas)
        all_model = np.array(all_model)
        
        overall_mbd_pct = MBD(all_meas, all_model)
        overall_rmse_pct = RMSE(all_meas, all_model)
        overall_mbd_abs = MBD_abs(all_meas, all_model)
        overall_rmse_abs = RMSE_abs(all_meas, all_model)
        overall_corr = np.corrcoef(all_meas, all_model)[0, 1]
        
        overall_row = {
            'gid': 'ALL',
            'setup': 'ALL',
            'n_points': len(all_meas),
            'mean_validation': all_meas.mean(),
            'mean_pysam': all_model.mean(),
            'MBD_percent': overall_mbd_pct,
            'RMSE_percent': overall_rmse_pct,
            'MBD_absolute': overall_mbd_abs,
            'RMSE_absolute': overall_rmse_abs,
            'correlation': overall_corr
        }
        
        summary_df = pd.concat([summary_df, pd.DataFrame([overall_row])], ignore_index=True)
        
        print("\n" + "="*60)
        print("OVERALL STATISTICS (All GIDs and Setups Combined)")
        print("="*60)
        print(f"Total matched points: {len(all_meas)}")
        print(f"MBD: {overall_mbd_pct:.2f}%")
        print(f"RMSE: {overall_rmse_pct:.2f}%")
        print(f"MBD (absolute): {overall_mbd_abs:.2f} W/m²")
        print(f"RMSE (absolute): {overall_rmse_abs:.2f} W/m²")
        print(f"Correlation: {overall_corr:.4f}")
    
    # Save to file if requested
    if output_file:
        summary_df.to_csv(output_file, index=False)
        print(f"\nSummary statistics (by GID and Setup) saved to: {output_file}")
        
        # Save distance-based statistics to a separate file
        if len(distance_summary_df) > 0:
            distance_output_file = output_file.replace('.csv', '_by_distance.csv')
            distance_summary_df.to_csv(distance_output_file, index=False)
            print(f"Summary statistics (by GID, Setup, and Distance) saved to: {distance_output_file}")
    
    return summary_df, distance_summary_df if len(distance_summary_df) > 0 else pd.DataFrame()


def main():
    """Command-line interface for comparing datasets."""
    parser = argparse.ArgumentParser(
        description='Compare validation results and PySAM outputs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_datasets.py
  python compare_datasets.py --data-file all_results.csv
  python compare_datasets.py --output comparison_results.csv
        """
    )
    
    parser.add_argument(
        '--data-file',
        type=str,
        default='all_results.csv',
        help='Path to combined results CSV file with data_source column (default: all_results.csv)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file for summary statistics (default: comparison_summary.csv)'
    )
    
    args = parser.parse_args()
    
    # Set default output filename if not provided
    output_file = args.output if args.output else 'comparison_summary.csv'
    
    # Compare datasets
    summary, distance_summary = compare_datasets(
        data_file=args.data_file,
        output_file=output_file
    )
    
    return summary, distance_summary


if __name__ == "__main__":
    main()

