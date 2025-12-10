"""
Plot Wm2Front vs Distance Comparison
This script generates line plots comparing PySAM and bifacial radiance data
for Wm2Front vs distance (x) for each of the 10 setups.

The plot shows data from a combined pickle file (all_results.pkl) with a data_source column
indicating whether data is from 'bifacial_radiance' or 'pysam'.

Usage:
    python plot_wm2front_vs_distance.py --gid 886847 --hour 12
    python plot_wm2front_vs_distance.py --gid 886847 --hour 12 --dayofyear 59
    python plot_wm2front_vs_distance.py --gid 886847 --hour 12 --output comparison_plots.png
    python plot_wm2front_vs_distance.py --data-file all_results.pkl --gid 886847 --hour 12
    python plot_wm2front_vs_distance.py --gid 886847 --hour 12 --dayofyear 172 --output June_21_comparison_plots.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
import warnings


def plot_wm2front_vs_distance(
    data_file='all_results.pkl',
    gid=None,
    hour=12,
    dayofyear=None,
    output_file=None,
    time_offset=-0.5
):
    """
    Generate line plots comparing PySAM and bifacial radiance data for Wm2Front vs distance.
    
    Parameters
    ----------
    data_file : str
        Path to combined results pickle file with data_source column
    gid : int
        GID to plot (if None, uses first GID found)
    hour : int, default 12
        Hour of day to plot (0-23)
    dayofyear : int, optional
        Day of year to plot (if None, uses first day found)
    output_file : str, optional
        Output file path for the plot (default: wm2front_vs_distance_gid{gid}_hour{hour}.png)
    time_offset : float, default -0.5
        Time offset in hours for matching bifacial_radiance to pysam times
        (bifacial_radiance is on :30, pysam is on :00)
    
    Returns
    -------
    str
        Path to saved plot file
    """
    print("Loading dataset...")
    
    # Load combined dataset
    all_data = pd.read_pickle(data_file)
    
    # Convert datetime columns
    all_data['datetime'] = pd.to_datetime(all_data['datetime'])
    
    # Extract day of year and hour (ignoring year)
    all_data['dayofyear'] = all_data['datetime'].dt.dayofyear
    all_data['hour'] = all_data['datetime'].dt.hour
    all_data['minute'] = all_data['datetime'].dt.minute
    
    # Split into bifacial_radiance and pysam datasets
    validation = all_data[all_data['data_source'] == 'bifacial_radiance'].copy()
    pysam = all_data[all_data['data_source'] == 'pysam'].copy()
    
    # Adjust validation hour by offset to match pysam times
    validation_adjusted = validation.copy()
    if time_offset < 0:
        validation_adjusted['hour_adjusted'] = validation_adjusted['hour'] - 1
    else:
        validation_adjusted['hour_adjusted'] = validation_adjusted['hour'] + 1
    validation_adjusted['hour_adjusted'] = validation_adjusted['hour_adjusted'] % 24
    
    # Select GID
    if gid is None:
        gid = validation['gid'].iloc[0]
        print(f"No GID specified, using first GID found: {gid}")
    else:
        print(f"Using GID: {gid}")
    
    # Check if GID exists in both datasets
    if gid not in validation['gid'].unique():
        raise ValueError(f"GID {gid} not found in validation data")
    if gid not in pysam['gid'].unique():
        raise ValueError(f"GID {gid} not found in pysam data")
    
    # Select day of year
    if dayofyear is None:
        dayofyear = validation_adjusted[validation_adjusted['gid'] == gid]['dayofyear'].iloc[0]
        print(f"No day of year specified, using first day found: {dayofyear}")
    else:
        print(f"Using day of year: {dayofyear}")
    
    print(f"Using hour: {hour}")
    
    # Create figure with subplots for each setup (2 rows, 5 columns)
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    fig.suptitle(f'GID: {gid}, Day of Year: {dayofyear}, Hour: {hour:02d}:00', 
                 fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    # Process each setup (1-10)
    for setup in range(1, 11):
        ax = axes[setup - 1]
        
        # Filter validation data for this GID, setup, day, and hour
        val_data = validation_adjusted[
            (validation_adjusted['gid'] == gid) &
            (validation_adjusted['setup'] == setup) &
            (validation_adjusted['dayofyear'] == dayofyear) &
            (validation_adjusted['hour_adjusted'] == hour)
        ].copy()
        
        # Filter pysam data for this GID, setup, day, and hour
        pysam_data = pysam[
            (pysam['gid'] == gid) &
            (pysam['setup'] == setup) &
            (pysam['dayofyear'] == dayofyear) &
            (pysam['hour'] == hour)
        ].copy()
        
        if len(val_data) == 0 and len(pysam_data) == 0:
            ax.text(0.5, 0.5, f'Setup {setup}\nNo data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Setup {setup}', fontweight='bold')
            continue
        
        # Sort by x for plotting
        if len(val_data) > 0:
            val_data = val_data.sort_values('x')
            ax.plot(val_data['x'], val_data['Wm2Front'], 
                   'o-', label='bifacial_radiance', linewidth=2, markersize=6, 
                   color="#0079C2", alpha=0.7)
        
        if len(pysam_data) > 0:
            pysam_data = pysam_data.sort_values('x')
            ax.plot(pysam_data['x'], pysam_data['Wm2Front'], 
                   's-', label='PySAM', linewidth=2, markersize=6, 
                   color="#F7A11A", alpha=0.7)
        
        # Formatting
        ax.set_xlabel('Location within row-to-row pitch (m)', fontsize=10)
        ax.set_ylabel('Ground Irradiance (W/m²)', fontsize=10)
        ax.set_title(f'Setup {setup}', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        
        # Set reasonable axis limits
        if len(val_data) > 0 or len(pysam_data) > 0:
            all_x = []
            all_y = []
            if len(val_data) > 0:
                all_x.extend(val_data['x'].values)
                all_y.extend(val_data['Wm2Front'].values)
            if len(pysam_data) > 0:
                all_x.extend(pysam_data['x'].values)
                all_y.extend(pysam_data['Wm2Front'].values)
            
            if len(all_x) > 0:
                x_min, x_max = min(all_x), max(all_x)
                y_min, y_max = 0, max(all_y)
                
                # Add some padding
                x_range = x_max - x_min
                y_range = y_max - y_min
                ax.set_xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)
                ax.set_ylim(max(0, y_min - 0.1 * y_range), y_max + 0.1 * y_range)
    
    plt.tight_layout()
    
    # Save plot
    if output_file is None:
        output_file = f'wm2front_vs_distance_gid{gid}_hour{hour:02d}.png'
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    return output_file


def main():
    """Command-line interface for plotting Wm2Front vs distance."""
    parser = argparse.ArgumentParser(
        description='Plot Wm2Front vs Distance comparison for PySAM and validation data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_wm2front_vs_distance.py --gid 886847 --hour 12
  python plot_wm2front_vs_distance.py --gid 886847 --hour 12 --dayofyear 59
  python plot_wm2front_vs_distance.py --gid 886847 --hour 12 --output my_plot.png
        """
    )
    
    parser.add_argument(
        '--data-file',
        type=str,
        default='all_results.pkl',
        help='Path to combined results pickle file with data_source column (default: all_results.pkl)'
    )
    
    parser.add_argument(
        '--gid',
        type=int,
        default=None,
        help='GID to plot (if not specified, uses first GID found)'
    )
    
    parser.add_argument(
        '--hour',
        type=int,
        default=12,
        help='Hour of day to plot (0-23, default: 12 for midday)'
    )
    
    parser.add_argument(
        '--dayofyear',
        type=int,
        default=None,
        help='Day of year to plot (if not specified, uses first day found)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path for the plot (default: wm2front_vs_distance_gid{gid}_hour{hour:02d}.png)'
    )
    
    args = parser.parse_args()
    
    # Generate plot
    output_path = plot_wm2front_vs_distance(
        data_file=args.data_file,
        gid=args.gid,
        hour=args.hour,
        dayofyear=args.dayofyear,
        output_file=args.output
    )
    
    print(f"\nPlot generation complete!")
    return output_path


if __name__ == "__main__":
    main()

