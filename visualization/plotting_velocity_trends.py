import sys
sys.path.append('../')
from mysql_dataset import database_helper
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict
import itertools
from scipy import stats
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from scipy import ndimage as ndi
from skimage.morphology import skeletonize, remove_small_objects, closing, disk, binary_closing
from skimage.measure import label, regionprops
from skimage.morphology import dilation, disk, binary_erosion, binary_dilation
from scipy.ndimage import convolve
import networkx as nx
from skimage.graph import MCP_Geometric
import math
import cv2
from scipy.ndimage import distance_transform_edt
from skimage.transform import resize
from scipy.spatial import cKDTree



def bootstrap(data, n):
    """Randomly resample n times and take the mean each time"""
    bootstrapped_data = np.zeros(n)
    for i in range(0, n):
        sample = np.random.choice(data, size=len(data))
        bootstrapped_data[i] = np.mean(np.array(sample))
    return bootstrapped_data

def confidence_interval(data):
    """Get the 95% confidence interval by getting the 2.5th and 97.5th percentile of the data"""
    conf_interval = np.percentile(data, [2.5, 97.5])
    return conf_interval[0], conf_interval[1]

def calculate_velocity_per_frame(data):
    """
    Calculate velocity for each ant on a per-frame basis.
    Returns a DataFrame with frame_number, ant_id, center_x, center_y, velocity, angle, direction.
    This is more efficient for flow field visualizations.
    """
    fps = 20  # frames per second
    velocity_data = []
    largest_frame_diff = 0
    # Group by ant_id to get individual tracks
    for ant_id, ant_data in data.groupby('ant_id'):
        ant_data = ant_data.sort_values('frame_number').copy()
        
        if len(ant_data) < 2:
            continue  # Need at least 2 points to calculate velocity
            
        # Calculate center points
        ant_data['center_x'] = (ant_data['x1'] + ant_data['x2']) / 2
        ant_data['center_y'] = (ant_data['y1'] + ant_data['y2']) / 2
        
        # Initialize velocity and angle columns
        ant_data['velocity'] = 0.0
        ant_data['angle'] = 0.0
        
        # Calculate velocity between consecutive frames
        for i in range(1, len(ant_data)):
            prev_frame = ant_data.iloc[i-1]
            curr_frame = ant_data.iloc[i]
            
            # Distance between consecutive frames (in pixels)
            dx = curr_frame['center_x'] - prev_frame['center_x']
            dy = curr_frame['center_y'] - prev_frame['center_y']
            distance = np.sqrt(dx**2 + dy**2)
            
            # Time difference in frames
            frame_diff = curr_frame['frame_number'] - prev_frame['frame_number']
            if frame_diff > largest_frame_diff:
                largest_frame_diff = frame_diff
            if frame_diff > 0:  # Avoid division by zero and s
                # Convert to pixels per second
                velocity = (distance / frame_diff) * fps
                # Calculate angle in radians (0 = right, π/2 = up, π = left, 3π/2 = down)
                #angle = np.arctan2(dy, dx)
                angle = np.arctan2(dy, dx) * 180 / np.pi  # Convert to degrees
                
                # Update the current frame with velocity and angle
                ant_data.iloc[i, ant_data.columns.get_loc('velocity')] = velocity
                ant_data.iloc[i, ant_data.columns.get_loc('angle')] = angle
        
        # Add to velocity_data list
        velocity_data.append(ant_data[['frame_number', 'ant_id', 'center_x', 'center_y', 'velocity', 'angle', 'direction']])
    print (f'largest frame diff is {largest_frame_diff}')
    if velocity_data:
        return pd.concat(velocity_data, ignore_index=True)
    else:
        return pd.DataFrame(columns=['frame_number', 'ant_id', 'center_x', 'center_y', 'velocity', 'angle', 'direction'])

def calculate_velocity(data):
    """
    Calculate velocity for each ant track (legacy function for backward compatibility).
    Velocity is calculated as the distance between consecutive frames divided by time.
    Data was collected at 20 frames per second.
    """
    velocity_df = calculate_velocity_per_frame(data)
    if len(velocity_df) > 0:
        # Return only non-zero velocities for backward compatibility
        return velocity_df[velocity_df['velocity'] > 0]['velocity'].tolist()
    else:
        return []

def save_velocity_data_for_video(tracking_csv_path, output_csv_path):
    """
    Calculate and save velocity data for a single video to a CSV file.
    This pre-calculates all velocity data for efficient future use.
    """
    print(f'Processing {tracking_csv_path}')
    data = pd.read_csv(tracking_csv_path)
    
    velocity_df = calculate_velocity_per_frame(data)
    
    # Save to CSV
    velocity_df.to_csv(output_csv_path, index=False)
    print(f'Saved velocity data to {output_csv_path}')
    
    return velocity_df

def preprocess_all_velocity_data(start_day, number_of_days=0, site_id=1):
    """
    Pre-process and save velocity data for all videos in the specified date range.
    This creates CSV files with per-frame velocity data for efficient future analysis.
    """
    df_per_site = df.loc[df.site_id == site_id]
    
    for i in range(0, number_of_days):
        # These are all the videos for that day (should be 24 videos for a full day)
        df_per_site_per_day = df_per_site.loc[(df_per_site.time_stamp.dt.day == (start_day+i).start_time.day) & 
                                             (df_per_site.time_stamp.dt.month == (start_day+i).start_time.month)]

        for index, video in df_per_site_per_day.iterrows():
            tracking_with_direction_csv = video['herdnet_tracking_with_direction_closest_boundary_method_csv']
            
            # Create output path for velocity data
            output_path = tracking_with_direction_csv.replace('_direction_and_angle_closest_boundary_thresh_20_7_1_0.1.csv', '_velocity_data.csv')
            # Process and save velocity data
            save_velocity_data_for_video(tracking_with_direction_csv, output_path)

def get_velocity_data_from_db(start_day, number_of_days=0, site_id=1, use_preprocessed=True):
    """
    Modified version of get_data_from_db to calculate velocities instead of counts.
    Returns velocity data for away/toward movements per hour.
    
    Args:
        use_preprocessed: If True, tries to load pre-calculated velocity data first.
                         If False or files don't exist, calculates on the fly.
    """
    df_per_site = df.loc[df.site_id == site_id]

    velocities_away_per_hour_across_days = defaultdict(list)
    velocities_toward_per_hour_across_days = defaultdict(list)
    velocities_loitering_per_hour_across_days = defaultdict(list)
    velocities_total = defaultdict(list)

    temperature = defaultdict(list)
    humidity = defaultdict(list)
    lux = defaultdict(list)

    for i in range(0, number_of_days):
        # These are all the videos for that day (should be 24 videos for a full day)
        df_per_site_per_day = df_per_site.loc[(df_per_site.time_stamp.dt.day == (start_day+i).start_time.day) & 
                                             (df_per_site.time_stamp.dt.month == (start_day+i).start_time.month)]

        for index, video in df_per_site_per_day.iterrows():
            hour = video['time_stamp'].hour
            
            temperature[hour].append(video['temperature'])
            humidity[hour].append(video['humidity'])
            lux[hour].append(video['LUX'])
            
            tracking_with_direction_csv = video['herdnet_tracking_with_direction_closest_boundary_method_csv']

            ### replace with closest_boundary method
            #tracking_with_direction_csv = tracking_with_direction_csv.replace('_direction_and_angle_7_1_0.1.csv','_direction_and_angle_closest_boundary_thresh_30_7_1_0.1.csv')

            velocity_csv_path = tracking_with_direction_csv.replace('_direction_and_angle_closest_boundary_thresh_20_7_1_0.1.csv', '_velocity_data.csv')

            # Try to load pre-calculated velocity data first
            if use_preprocessed and os.path.exists(velocity_csv_path):
                print(f'Loading pre-calculated velocity data from {velocity_csv_path}')
                velocity_data = pd.read_csv(velocity_csv_path)
            else:
                print(f'Calculating velocity data for {tracking_with_direction_csv}')
                data = pd.read_csv(tracking_with_direction_csv)
                velocity_data = calculate_velocity_per_frame(data)
            
            # Calculate average velocities for away direction
            data_away = velocity_data[velocity_data.direction == 'away']
            if len(data_away) > 0:
                away_velocities = data_away['velocity'].tolist()
                if away_velocities:
                    avg_velocity_away = np.mean(away_velocities)
                    velocities_away_per_hour_across_days[hour].append(avg_velocity_away)
                else:
                    velocities_away_per_hour_across_days[hour].append(0)
            else:
                velocities_away_per_hour_across_days[hour].append(0)

            # Calculate average velocities for toward direction
            data_toward = velocity_data[velocity_data.direction == 'toward']
            if len(data_toward) > 0:
                toward_velocities = data_toward['velocity'].tolist()
                if toward_velocities:
                    avg_velocity_toward = np.mean(toward_velocities)
                    velocities_toward_per_hour_across_days[hour].append(avg_velocity_toward)
                else:
                    velocities_toward_per_hour_across_days[hour].append(0)
            else:
                velocities_toward_per_hour_across_days[hour].append(0)

            # Calculate average velocities for unknown/loitering direction
            data_loitering = velocity_data[velocity_data.direction == 'unknown']
            if len(data_loitering) > 0:
                loitering_velocities = data_loitering['velocity'].tolist()
                if loitering_velocities:
                    avg_velocity_loitering = np.mean(loitering_velocities)
                    velocities_loitering_per_hour_across_days[hour].append(avg_velocity_loitering)
                else:
                    velocities_loitering_per_hour_across_days[hour].append(0)
            else:
                velocities_loitering_per_hour_across_days[hour].append(0)

            # Calculate total average velocities
            if len(velocity_data) > 0:
                total_velocities = velocity_data['velocity'].tolist()
                if total_velocities:
                    avg_velocity_total = np.mean(total_velocities)
                    velocities_total[hour].append(avg_velocity_total)
                else:
                    velocities_total[hour].append(0)
            else:
                velocities_total[hour].append(0)
    
    return velocities_away_per_hour_across_days, velocities_toward_per_hour_across_days, velocities_loitering_per_hour_across_days, velocities_total, temperature, humidity, lux

def load_velocity_data_for_flow_field(tracking_csv_path):
    """
    Load velocity data for flow field visualization.
    Returns a DataFrame with frame_number, ant_id, center_x, center_y, velocity, angle, direction.
    """
    velocity_csv_path = tracking_csv_path.replace('_direction_and_angle_closest_boundary_thresh_20_7_1_0.1.csv', '_velocity_data.csv')
    
    if os.path.exists(velocity_csv_path):
        print(f'Loading velocity data from {velocity_csv_path}')
        return pd.read_csv(velocity_csv_path)
    else:
        print(f'Velocity data not found. Calculating from {tracking_csv_path}')
        data = pd.read_csv(tracking_csv_path)
        velocity_data = calculate_velocity_per_frame(data)
        
        # Save for future use
        velocity_data.to_csv(velocity_csv_path, index=False)
        print(f'Saved velocity data to {velocity_csv_path}')
        
        return velocity_data

def get_flow_field_data_for_hour(start_day, hour, site_id=1):
    """
    Get velocity data for a specific hour for flow field visualization.
    Returns a list of DataFrames, one for each video in that hour.
    """
    df_per_site = df.loc[df.site_id == site_id]
    
    # Get videos for the specific hour
    df_per_site_per_hour = df_per_site.loc[(df_per_site.time_stamp.dt.day == start_day.start_time.day) & 
                                          (df_per_site.time_stamp.dt.month == start_day.start_time.month) &
                                          (df_per_site.time_stamp.dt.hour == hour)]
    
    flow_field_data = []
    
    for index, video in df_per_site_per_hour.iterrows():
        tracking_with_direction_csv = video['herdnet_tracking_with_direction_closest_boundary_method_csv']
        velocity_data = load_velocity_data_for_flow_field(tracking_with_direction_csv)
        flow_field_data.append(velocity_data)
    
    return flow_field_data

def get_flow_field_data_for_day(day, site_id=1):
    """
    Get velocity data for a specific day for flow field visualization.
    Returns a list of length 24, where each entry is a list of DataFrames
    (one for each video in that hour).
    """
    df_per_site = df.loc[df.site_id == site_id]

    # Filter by day/month
    df_per_site_per_day = df_per_site.loc[
        (df_per_site.time_stamp.dt.day == day.start_time.day) & 
        (df_per_site.time_stamp.dt.month == day.start_time.month)
    ]

    # Initialize list of 24 entries
    flow_field_data = [[] for _ in range(24)]

    for _, video in df_per_site_per_day.iterrows():
        tracking_with_direction_csv = video['herdnet_tracking_with_direction_closest_boundary_method_csv']
        velocity_data = load_velocity_data_for_flow_field(tracking_with_direction_csv)
        hour = video['time_stamp'].hour
        flow_field_data[hour].append(velocity_data)

    return flow_field_data






def plot_flux_field_for_hour(flow_field_data, hour, site_id=1, bin_size=20, 
                             direction_filter=None, ax=None, 
                             vmin=None, vmax=None, cmap="Reds", show_colorbar=False):
    """
    Plot flux field for a given hour using velocity data.
    Flux = number of ants * velocity vector (magnitude & direction).
    """
    if not flow_field_data:
        print(f"No flow field data available for hour {hour}")
        return None, None
    
    all_velocity_data = pd.concat(flow_field_data, ignore_index=True)
    
    # Filter by direction
    if direction_filter:
        all_velocity_data = all_velocity_data[all_velocity_data['direction'] == direction_filter]
        if len(all_velocity_data) == 0:
            print(f"No data for direction '{direction_filter}' in hour {hour}")
            return None, None
    
    # Exclude stationary ants
    moving_data = all_velocity_data[all_velocity_data['velocity'] > 0].copy()
    if len(moving_data) == 0:
        print(f"No moving ants found for hour {hour}")
        return None, None
    
    # Bin setup
    x_min, x_max = 0, 1920
    y_min, y_max = 0, 1080
    x_bins = np.arange(x_min, x_max + bin_size, bin_size)
    y_bins = np.arange(y_min, y_max + bin_size, bin_size)
    
    moving_data['x_bin'] = pd.cut(moving_data['center_x'], bins=x_bins, labels=False)
    moving_data['y_bin'] = pd.cut(moving_data['center_y'], bins=y_bins, labels=False)
    
    # Compute vector components for each ant
    moving_data['u'] = np.cos(moving_data['angle']) * moving_data['velocity']
    moving_data['v'] = np.sin(moving_data['angle']) * moving_data['velocity']  # don't flip y here we do it when plotting
    
    # Flux = sum of velocity vectors per bin
    bin_stats = moving_data.groupby(['x_bin', 'y_bin']).agg({
        'u': 'sum',
        'v': 'sum',
        'velocity': 'count'   # number of ants
    }).reset_index().dropna()
    
    if len(bin_stats) == 0:
        print(f"No valid bins found for hour {hour}")
        return None, None
    
    # Bin centers
    bin_stats['x_center'] = x_bins[bin_stats['x_bin'].astype(int)] + bin_size / 2
    bin_stats['y_center'] = y_bins[bin_stats['y_bin'].astype(int)] + bin_size / 2
    
    # Flux magnitude and normalized direction
    bin_stats['flux_mag'] = np.sqrt(bin_stats['u']**2 + bin_stats['v']**2)
    bin_stats['angle'] = np.arctan2(bin_stats['v'], bin_stats['u'])
    
    # Normalize flux magnitude with clipping
    if vmin is None:
        vmin = 0
    if vmax is None:
        # avoid zero-range
        nonzero = bin_stats['flux_mag'][bin_stats['flux_mag'] > 0]
        vmax = float(np.percentile(nonzero, 95)) if len(nonzero) > 0 else bin_stats['flux_mag'].max()


    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    
    flux_clipped = np.clip(bin_stats['flux_mag'], vmin, vmax)
    flux_scale = (flux_clipped - vmin) / (vmax - vmin + 1e-6)
    
    # Scale arrow length instead of using raw u,v
    base_arrow_length = bin_size * 0.8
    arrow_lengths = base_arrow_length * (0.5 + flux_scale * 1.5)
    arrow_alpha = 0.3 + 0.7 * flux_scale
    
    # Unit vectors * arrow length
    u = np.cos(bin_stats['angle']) * arrow_lengths
    v = -np.sin(bin_stats['angle']) * arrow_lengths
    
    q = ax.quiver(bin_stats['x_center'], bin_stats['y_center'], u, v,  # flip y
                  flux_clipped, cmap=cmap, norm=norm,
                  scale_units="xy", scale=1, width=0.005,
                  headwidth=3, headlength=3, alpha=arrow_alpha)
    
    if show_colorbar:
        fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.06)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label('Flux (ant·pixels/s)', rotation=270, labelpad=12)
    
    # Titles/labels
    direction_text = f" ({direction_filter})" if direction_filter else " (all)"
    site_name = {1: 'beer', 2: 'shack', 3: 'rain'}.get(site_id, f'site_{site_id}')
    ax.set_title(f'Flux Field - {site_name.title()} Tree, Hour {hour:02d}:00{direction_text}')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    
    stats_text = f'Bins: {len(bin_stats)}, Points: {len(moving_data)}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return bin_stats, q


def plot_flow_field_for_hour(flow_field_data, hour, site_id=1, bin_size=20, 
                             direction_filter=None, ax=None, 
                             vmin=None, vmax=None, cmap="Reds"):
    """
    Plot flow field for a given hour using velocity data.
    """
    if not flow_field_data:
        print(f"No flow field data available for hour {hour}")
        return None, None
    
    all_velocity_data = pd.concat(flow_field_data, ignore_index=True)
    
    # Filter by direction
    if direction_filter:
        all_velocity_data = all_velocity_data[all_velocity_data['direction'] == direction_filter]
        if len(all_velocity_data) == 0:
            print(f"No data for direction '{direction_filter}' in hour {hour}")
            return None, None
    
    # Exclude stationary ants
    moving_data = all_velocity_data[all_velocity_data['velocity'] > 0].copy()
    if len(moving_data) == 0:
        print(f"No moving ants found for hour {hour}")
        return None, None
    
    # Bin setup
    x_min, x_max = 0, 1920
    y_min, y_max = 0, 1080
    x_bins = np.arange(x_min, x_max + bin_size, bin_size)
    y_bins = np.arange(y_min, y_max + bin_size, bin_size)
    
    moving_data['x_bin'] = pd.cut(moving_data['center_x'], bins=x_bins, labels=False)
    moving_data['y_bin'] = pd.cut(moving_data['center_y'], bins=y_bins, labels=False)
    
    # Aggregate stats
    bin_stats = moving_data.groupby(['x_bin', 'y_bin']).agg({
        'velocity': 'mean',
        'angle': lambda x: np.arctan2(np.mean(np.sin(x)), np.mean(np.cos(x)))
    }).reset_index().dropna()
    
    if len(bin_stats) == 0:
        print(f"No valid bins found for hour {hour}")
        return None, None
    
    # Bin centers
    bin_stats['x_center'] = x_bins[bin_stats['x_bin'].astype(int)] + bin_size / 2
    bin_stats['y_center'] = y_bins[bin_stats['y_bin'].astype(int)] + bin_size / 2
    
    # Normalize velocity with clipping
    min_velocity = vmin if vmin is not None else bin_stats['velocity'].min()
    max_velocity = vmax if vmax is not None else bin_stats['velocity'].max()
    norm = plt.Normalize(vmin=min_velocity, vmax=max_velocity)
    
    velocity_clipped = np.clip(bin_stats['velocity'], min_velocity, max_velocity)
    velocity_scale = (velocity_clipped - min_velocity) / (max_velocity - min_velocity + 1e-6)
    
    base_arrow_length = bin_size * 0.8
    arrow_lengths = base_arrow_length * (0.5 + velocity_scale * 1.5)
    arrow_alpha = 0.3 + 0.7 * velocity_scale
    
    # Arrow components
    u = np.cos(np.asarray(bin_stats['angle'])) * np.asarray(arrow_lengths)
    v = -np.sin(np.asarray(bin_stats['angle'])) * np.asarray(arrow_lengths)  # flip y
    
    # Plot
    q = ax.quiver(bin_stats['x_center'], bin_stats['y_center'], u, v,
                  velocity_clipped, cmap=cmap, norm=norm,
                  scale_units="xy", scale=1, width=0.005,
                  headwidth=3, headlength=3, alpha=arrow_alpha)
    
    # Titles/labels
    direction_text = f" ({direction_filter})" if direction_filter else " (all)"
    site_name = {1: 'beer', 2: 'shack', 3: 'rain'}.get(site_id, f'site_{site_id}')
    ax.set_title(f'Flow Field - {site_name.title()} Tree, Hour {hour:02d}:00{direction_text}')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    
    stats_text = f'Bins: {len(bin_stats)}, Points: {len(moving_data)}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return bin_stats, q



def plot_flux_field_comparison(flow_field_data, hour, site_id=1, bin_size=20,
                               flux_cap_percentile=95):
    """
    Plot away vs toward flux fields side by side with shared capped color scale.
    """
    # Get all velocity data
    all_data = pd.concat(flow_field_data, ignore_index=True)
    moving_data = all_data[all_data['velocity'] > 0].copy()
    
    if len(moving_data) == 0:
        print(f"No ants for hour {hour}")
        return None, None
    
    # Compute flux magnitude per bin (ignoring direction filter for normalization)
    x_bins = np.arange(0, moving_data['center_x'].max() + bin_size, bin_size)
    y_bins = np.arange(0, moving_data['center_y'].max() + bin_size, bin_size)
    
    moving_data['u'] = np.cos(moving_data['angle']) * moving_data['velocity']
    moving_data['v'] = -np.sin(moving_data['angle']) * moving_data['velocity']
    
    bin_flux = moving_data.groupby([
        pd.cut(moving_data['center_x'], bins=x_bins, labels=False),
        pd.cut(moving_data['center_y'], bins=y_bins, labels=False)
    ]).agg({'u':'sum', 'v':'sum'})
    bin_flux['flux_mag'] = np.sqrt(bin_flux['u']**2 + bin_flux['v']**2)
    
    fluxes = bin_flux['flux_mag'].values
    non_zero_fluxes = fluxes[fluxes > 0]
    
    vmin = 0
    vmax = np.percentile(non_zero_fluxes, flux_cap_percentile) if len(non_zero_fluxes) > 0 else 1
    
    # Now plot both maps with the shared vmin/vmax
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    away_stats, q1 = plot_flux_field_for_hour(
        flow_field_data, hour, site_id, bin_size,
        direction_filter='away', ax=ax1, vmin=vmin, vmax=vmax, cmap="Reds"
    )
    toward_stats, q2 = plot_flux_field_for_hour(
        flow_field_data, hour, site_id, bin_size,
        direction_filter='toward', ax=ax2, vmin=vmin, vmax=vmax, cmap="Reds"
    )
    
    sm = plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Flux (ants × velocity)', rotation=270, labelpad=20)
    
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.show()
    
    return away_stats, toward_stats



def plot_daily_flux_grid(day_period, site_id=1, direction_filter='away',
                         bin_size=20, normalize='per_hour', flux_cap_percentile=95,
                         cmap="Reds"):
    """
    Make a 6x4 grid of hourly flux maps for a given day (day_period is your Period object).
    normalize: 'per_hour' (each subplot has its own colorbar) or 'shared' (single colorbar for grid).
    Returns nothing (shows figure).
    """
    # figure + axes (6 rows x 4 cols)
    fig, axes = plt.subplots(6, 4, figsize=(20, 28))
    axes = axes.flatten()

    # If shared normalization, compute vmax across the day
    shared_vmin, shared_vmax = 0, None
    if normalize == 'shared':
        all_flux_vals = []
        for h in range(24):
            flow_data = get_flow_field_data_for_hour(day_period, h, site_id=site_id)
            if not flow_data:
                continue
            # compute per-hour flux magnitudes quickly (reuse logic)
            all_df = pd.concat(flow_data, ignore_index=True)
            moving = all_df[all_df['velocity'] > 0].copy()
            if moving.shape[0] == 0:
                continue
            moving['u'] = np.cos(moving['angle'].astype(float)) * moving['velocity'].astype(float)
            moving['v'] = -np.sin(moving['angle'].astype(float)) * moving['velocity'].astype(float)
            moving['x_bin'] = pd.cut(moving['center_x'], bins=np.arange(0, 1920 + bin_size, bin_size), labels=False)
            moving['y_bin'] = pd.cut(moving['center_y'], bins=np.arange(0, 1080 + bin_size, bin_size), labels=False)
            bin_flux = moving.groupby(['x_bin', 'y_bin']).agg({'u':'sum','v':'sum'}).reset_index()
            if bin_flux.shape[0] > 0:
                bin_flux['flux_mag'] = np.sqrt(bin_flux['u']**2 + bin_flux['v']**2)
                all_flux_vals.append(bin_flux['flux_mag'].values)
        if len(all_flux_vals) > 0:
            all_flux_concat = np.concatenate(all_flux_vals)
            nonzero = all_flux_concat[all_flux_concat > 0]
            if len(nonzero) > 0:
                shared_vmax = float(np.percentile(nonzero, flux_cap_percentile))
            else:
                shared_vmax = float(all_flux_concat.max()) if all_flux_concat.size > 0 else 1.0
        else:
            shared_vmax = 1.0

    
    # loop hours and plot each subplot
    for hour in range(24):
        ax = axes[hour]
        flow_data = get_flow_field_data_for_hour(day_period, hour, site_id=site_id)
        if not flow_data:
            ax.set_title(f"Hour {hour:02d}:00 — no data")
            ax.axis('off')
            continue

        # compute per-hour vmax if per_hour normalization
        if normalize == 'per_hour':
            # compute per-hour fluxes and pick percentile
            all_df = pd.concat(flow_data, ignore_index=True)
            moving = all_df[all_df['velocity'] > 0].copy()
            if moving.shape[0] == 0:
                ax.set_title(f"Hour {hour:02d}:00 — no movers")
                ax.axis('off')
                continue
            moving['u'] = np.cos(moving['angle'].astype(float)) * moving['velocity'].astype(float)
            moving['v'] = -np.sin(moving['angle'].astype(float)) * moving['velocity'].astype(float)
            moving['x_bin'] = pd.cut(moving['center_x'], bins=np.arange(0, 1920 + bin_size, bin_size), labels=False)
            moving['y_bin'] = pd.cut(moving['center_y'], bins=np.arange(0, 1080 + bin_size, bin_size), labels=False)
            bin_flux = moving.groupby(['x_bin', 'y_bin']).agg({'u':'sum','v':'sum'}).reset_index()
            if bin_flux.shape[0] > 0:
                bin_flux['flux_mag'] = np.sqrt(bin_flux['u']**2 + bin_flux['v']**2)
                nonzero = bin_flux['flux_mag'][bin_flux['flux_mag'] > 0]
                if len(nonzero) > 0:
                    vmin = 0
                    vmax = float(np.percentile(nonzero, flux_cap_percentile))
                else:
                    vmin, vmax = 0, bin_flux['flux_mag'].max() if bin_flux['flux_mag'].size>0 else 1.0
            else:
                vmin, vmax = 0, 1.0
            show_cbar = True
        else:
            # shared normalization
            vmin, vmax = shared_vmin, shared_vmax
            show_cbar = False  # show a single shared one later

        # call the per-hour plotting routine
        try:
            bin_stats, q = plot_flux_field_for_hour(flow_data, hour, site_id=site_id,
                                                    bin_size=bin_size, direction_filter=direction_filter,
                                                    ax=ax, vmin=vmin, vmax=vmax, cmap=cmap,
                                                    show_colorbar=show_cbar)
        except Exception as e:
            ax.set_title(f"Hour {hour:02d}:00 — error")
            ax.text(0.5, 0.5, str(e), transform=ax.transAxes, fontsize=8)
            ax.axis('off')

    # If shared normalization, add a single colorbar for the figure
    if normalize == 'shared':
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=shared_vmin, vmax=shared_vmax))
        sm.set_array([])
        # place colorbar on the right of the grid
        cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  # [left, bottom, width, height] in fraction coords
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label('Flux (ant·pixels/s)', rotation=270, labelpad=12)

    plt.suptitle(f"Flux grid ({direction_filter}) — Day {day_period}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    plt.show()







def plot_flow_field_comparison(flow_field_data, hour, site_id=1, bin_size=20, 
                               velocity_cap_percentile=95):
    """
    Plot away vs toward flow fields side by side with shared capped color scale.
    """
    all_data = pd.concat(flow_field_data, ignore_index=True)
    velocities = all_data.loc[all_data['velocity'] > 0, 'velocity']
    if len(velocities) == 0:
        print(f"No moving ants for hour {hour}")
        return None, None
    
    vmin = velocities.min()
    vmax = np.percentile(velocities, velocity_cap_percentile)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    away_stats, q1 = plot_flow_field_for_hour(
        flow_field_data, hour, site_id, bin_size,
        direction_filter='away', ax=ax1, vmin=vmin, vmax=vmax
    )
    toward_stats, q2 = plot_flow_field_for_hour(
        flow_field_data, hour, site_id, bin_size,
        direction_filter='toward', ax=ax2, vmin=vmin, vmax=vmax
    )
    
    # Shared colorbar on the right side
    sm = plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    
    # create an axis on the right for the colorbar
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Average Velocity (pixels/second)', rotation=270, labelpad=20)
    
    plt.tight_layout(rect=[0, 0, 0.95, 1])  # leave space for colorbar
    plt.show()
    
    return away_stats, toward_stats






def statistic_test_for_velocity_differences(away_values, toward_values, hour):
    """Statistical test (t-test) between away and toward velocities for every hour"""
    if len(away_values) > 0 and len(toward_values) > 0:
        t_statistic, p_value = stats.ttest_ind(away_values, toward_values)
        print(f'Hour {hour} - t_stat: {t_statistic:.4f}, p_value: {p_value:.4f}')

        # Check for normality
        if len(away_values) >= 3:
            stat, p = stats.shapiro(away_values)
            if p <= 0.05:
                print(f'Shapiro test: away velocity data is not normally distributed for hour {hour}')
                print(f'Mann-Whitney U: {stats.mannwhitneyu(away_values, toward_values)}')

        if len(toward_values) >= 3:
            stat, p = stats.shapiro(toward_values)
            if p <= 0.05:
                print(f'Shapiro test: toward velocity data is not normally distributed for hour {hour}')
                print(f'Mann-Whitney U: {stats.mannwhitneyu(away_values, toward_values)}')

def scatter_plot_velocity_for_day_range(start_day, number_of_days=0, site_id=1):
    """
    Create a scatter plot showing average velocity of ants moving away vs toward for each hour.
    Similar to scatter_plot_for_day_range but for velocities instead of counts.
    """
    velocities_away_per_hour_across_days, velocities_toward_per_hour_across_days, velocities_loitering_per_hour_across_days, velocities_total, temperature, humidity, lux = get_velocity_data_from_db(start_day, number_of_days, site_id)
    
    fig, axes = plt.subplots()
    x_list = []
    y_list_away, y_list_toward, y_list_loitering = [], [], []
    y_list_total = []

    y_away_err_lower, y_away_err_upper = [], []
    y_toward_err_lower, y_toward_err_upper = [], []
    y_loitering_err_lower, y_loitering_err_upper = [], []
    y_total_err_lower, y_total_err_upper = [], []

    bootstrapped_means_away, bootstrapped_means_toward, bootstrapped_means_loitering = [], [], []
    bootstrapped_means_total = []

    for h in range(24):
        away_values = velocities_away_per_hour_across_days[h]
        y_list_away.append(away_values)
        
        if len(away_values) > 0:
            bootstrapped_data = bootstrap(away_values, 10000)
            bootstrapped_means_away.append(np.mean(bootstrapped_data))
            conf_min, conf_max = confidence_interval(bootstrapped_data)
            y_away_err_lower.append(np.mean(bootstrapped_data) - conf_min)
            y_away_err_upper.append(conf_max - np.mean(bootstrapped_data))
        else:
            bootstrapped_means_away.append(0)
            y_away_err_lower.append(0)
            y_away_err_upper.append(0)
        
        toward_values = velocities_toward_per_hour_across_days[h]
        y_list_toward.append(toward_values)
        
        if len(toward_values) > 0:
            bootstrapped_data = bootstrap(toward_values, 10000)
            conf_min, conf_max = confidence_interval(bootstrapped_data)
            bootstrapped_means_toward.append(np.mean(bootstrapped_data))
            y_toward_err_lower.append(np.mean(bootstrapped_data) - conf_min)
            y_toward_err_upper.append(conf_max - np.mean(bootstrapped_data))
        else:
            bootstrapped_means_toward.append(0)
            y_toward_err_lower.append(0)
            y_toward_err_upper.append(0)

        loitering_values = velocities_loitering_per_hour_across_days[h]
        y_list_loitering.append(loitering_values)
        
        if len(loitering_values) > 0:
            bootstrapped_data = bootstrap(loitering_values, 10000)
            conf_min, conf_max = confidence_interval(bootstrapped_data)
            bootstrapped_means_loitering.append(np.mean(bootstrapped_data))
            y_loitering_err_lower.append(np.mean(bootstrapped_data) - conf_min)
            y_loitering_err_upper.append(conf_max - np.mean(bootstrapped_data))
        else:
            bootstrapped_means_loitering.append(0)
            y_loitering_err_lower.append(0)
            y_loitering_err_upper.append(0)

        
        total_velocities = velocities_total[h]
        y_list_total.append(total_velocities)
        
        if len(total_velocities) > 0:
            bootstrapped_data = bootstrap(total_velocities, 10000)
            conf_min, conf_max = confidence_interval(bootstrapped_data)
            bootstrapped_means_total.append(np.mean(bootstrapped_data))
            y_total_err_lower.append(np.mean(bootstrapped_data) - conf_min)
            y_total_err_upper.append(conf_max - np.mean(bootstrapped_data))
        else:
            bootstrapped_means_total.append(0)
            y_total_err_lower.append(0)
            y_total_err_upper.append(0)

        x_list.append([h] * len(away_values))

        statistic_test_for_velocity_differences(away_values, toward_values, h)
    
    x_axis = list(itertools.chain.from_iterable(x_list))

    # Add some jitter
    x_axis_away = x_axis + np.random.uniform(low=0.10, high=0.10, size=len(x_axis))
    x_axis_toward = x_axis + np.random.uniform(low=-0.10, high=-0.10, size=len(x_axis))
    
    ## Plot total velocities
    # axes.scatter(x_axis, list(itertools.chain.from_iterable(y_list_total)), marker='.', c='k', s=20, alpha=0.3)
    # axes.plot(range(24), bootstrapped_means_total, c='k', alpha=0.3)
    # if number_of_days > 1:	
    #     axes.errorbar(range(24), bootstrapped_means_total, yerr=[y_total_err_lower, y_total_err_upper], 
    #                  fmt=".", c='k', ecolor='k', elinewidth=1, label='average velocity')

    ## Plot away vs toward velocities
    axes.scatter(x_axis_away, list(itertools.chain.from_iterable(y_list_away)), marker='.', c='g', s=20, alpha=0.3)
    axes.errorbar(range(24) + np.random.uniform(low=0.10, high=0.10, size=24), bootstrapped_means_away, 
                yerr=[y_away_err_lower, y_away_err_upper], fmt=".", c='g', ecolor='k', elinewidth=1, label='away')

    axes.scatter(x_axis_toward, list(itertools.chain.from_iterable(y_list_toward)), marker='.', c='r', s=20, alpha=0.3)
    axes.errorbar(range(24) + np.random.uniform(low=-0.10, high=-0.10, size=24), bootstrapped_means_toward, 
                 yerr=[y_toward_err_lower, y_toward_err_upper], fmt=".", c='r', ecolor='k', elinewidth=1, label='toward')

    axes.scatter(x_axis_toward, list(itertools.chain.from_iterable(y_list_loitering)), marker='.', c='y', s=20, alpha=0.3)
    axes.errorbar(range(24) + np.random.uniform(low=-0.10, high=-0.10, size=24), bootstrapped_means_loitering, 
                 yerr=[y_loitering_err_lower, y_loitering_err_upper], fmt=".", c='y', ecolor='k', elinewidth=1, label='loitering/unknown')
    
    axes.legend()
    plt.title('Average ant velocity Rain tree 2024-11-15 to 2024-12-06')
    plt.xticks(range(24))
    plt.xlabel('Hour')
    plt.ylabel('Average Velocity (pixels/second)')
    plt.show()






def compute_flux_vectors(flow_field_data, site_id=1, bin_size=20, direction_filter="away", 
                         normalize=True):
    """
    Convert hourly flux maps into flattened vectors for comparison.
    Returns dict {hour: flux_vector}.
    """
    flux_vectors = {}

    for hour in range(24):
        all_data = pd.concat(flow_field_data[hour], ignore_index=True) if flow_field_data[hour] else None
        if all_data is None or len(all_data) == 0:
            flux_vectors[hour] = None
            continue

        moving_data = all_data[all_data['velocity'] > 0].copy()
        if direction_filter:
            moving_data = moving_data[moving_data['direction'] == direction_filter]
        if len(moving_data) == 0:
            flux_vectors[hour] = None
            continue

        # Bin setup
        x_bins = np.arange(0, 1920 + bin_size, bin_size)
        y_bins = np.arange(0, 1080 + bin_size, bin_size)
        moving_data['u'] = np.cos(moving_data['angle']) * moving_data['velocity']
        moving_data['v'] = -np.sin(moving_data['angle']) * moving_data['velocity']

        bin_flux = moving_data.groupby([
            pd.cut(moving_data['center_x'], bins=x_bins, labels=False),
            pd.cut(moving_data['center_y'], bins=y_bins, labels=False)
        ]).agg({'u':'sum', 'v':'sum'})
        bin_flux['flux_mag'] = np.sqrt(bin_flux['u']**2 + bin_flux['v']**2)

        # Flatten into vector
        flux_vec = bin_flux['flux_mag'].reindex(
            pd.MultiIndex.from_product([range(len(x_bins)-1), range(len(y_bins)-1)]),
            fill_value=0
        ).values

        # Normalize
        if normalize and flux_vec.sum() > 0:
            flux_vec = flux_vec / flux_vec.sum()

        flux_vectors[hour] = flux_vec

    return flux_vectors


def plot_flux_similarity_matrix(dayA_data, dayB_data, site_id=1, bin_size=20, 
                                direction_filter="away", metric="cosine"):
    """
    Compute and plot a 24x24 similarity matrix between two days of flux maps.
    """
    if direction_filter == 'same_day_away_vs_toward':
        fluxA = compute_flux_vectors(dayA_data, site_id, bin_size, 'away')
        fluxB = compute_flux_vectors(dayA_data, site_id, bin_size, 'toward')
    
    else:
        # Compute flux vectors for both days
        fluxA = compute_flux_vectors(dayA_data, site_id, bin_size, direction_filter)
        fluxB = compute_flux_vectors(dayB_data, site_id, bin_size, direction_filter)

    # Build matrix
    sim_matrix = np.zeros((24, 24))
    for i in range(24):
        for j in range(24):
            vecA = fluxA[i]
            vecB = fluxB[j]
            if vecA is None or vecB is None:
                sim_matrix[i, j] = np.nan
            else:
                if metric == "cosine":
                    sim_matrix[i, j] = cosine_similarity([vecA], [vecB])[0, 0]
                elif metric == "correlation":
                    sim_matrix[i, j] = np.corrcoef(vecA, vecB)[0, 1]
                elif metric == "euclidean":
                    sim_matrix[i, j] = -np.linalg.norm(vecA - vecB)  # negative distance for heatmap
                else:
                    raise ValueError(f"Unknown metric {metric}")

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, cmap="viridis", annot=False, square=True,
                xticklabels=[f"{h:02d}" for h in range(24)],
                yticklabels=[f"{h:02d}" for h in range(24)],
                cbar_kws={'label': f'{metric.title()} similarity'})
    plt.title(f"Flux similarity ({direction_filter})\n — Site {site_id}")
    plt.xlabel("Towards flux maps (hour)")
    plt.ylabel("Away flux maps (hour)")
    plt.show()

    return sim_matrix






def diagonal_mean_from_pairs(sim_matrix, perm):
    """
    Helper: compute mean of diagonal after pairing row i with column perm[i],
    ignoring NaNs.
    """
    n = sim_matrix.shape[0]
    vals = []
    for i in range(n):
        v = sim_matrix[i, perm[i]]
        if not np.isnan(v):
            vals.append(v)
    return np.nan if len(vals) == 0 else np.nanmean(vals)


def quantify_diagonal_strength_extended(sim_matrix,
                                        n_permutations=2000,
                                        random_state=None,
                                        method='permute_pairs',
                                        group_labels=None,
                                        plot_hist=True,
                                        plot_title=None):
    """
    Quantify diagonal strength with different null models and plot null histogram.

    Parameters
    ----------
    sim_matrix : (n,n) np.ndarray
        Similarity matrix (may contain NaNs).
    n_permutations : int
        Number of permutations to draw for null distribution.
    random_state : int or None
        RNG seed.
    method : str
        One of:
        - 'permute_pairs' : randomly permute columns (preserves row/col marginals and block structure)
        - 'shuffle_entries' : shuffle all non-NaN entries and re-fill matrix (breaks block structure)
        - 'grouped_permute' : permute columns *within groups* defined by group_labels
    group_labels : sequence or None
        If method == 'grouped_permute', group_labels must be provided (len == n).
    plot_hist : bool
        If True, plot histogram of null distribution and mark observed diag mean.
    plot_title : str or None
        Title to put on the histogram.

    Returns
    -------
    results : dict
        {
          mean_diag: observed diagonal mean,
          mean_offdiag: observed off-diagonal mean,
          contrast: mean_diag / mean_offdiag (or np.inf),
          p_value: p-value (fraction null >= observed),
          null_distribution: np.ndarray of permutation diagonal means,
          method: method
        }
    """
    rng = np.random.default_rng(random_state)
    sim = np.asarray(sim_matrix, dtype=float)
    n = sim.shape[0]
    if sim.shape[0] != sim.shape[1]:
        raise ValueError("sim_matrix must be square")

    # Observed diag and off-diag
    mask_valid = ~np.isnan(sim)
    diag_vals = np.array([sim[i, i] for i in range(n) if mask_valid[i, i]])
    offmask = mask_valid & (~np.eye(n, dtype=bool))
    offdiag_vals = sim[offmask]
    mean_diag = np.nan if diag_vals.size == 0 else np.nanmean(diag_vals)
    mean_offdiag = np.nan if offdiag_vals.size == 0 else np.nanmean(offdiag_vals)
    contrast = (mean_diag / mean_offdiag) if (mean_offdiag is not None and mean_offdiag != 0) else np.inf

    null_distribution = []

    if method == 'permute_pairs':
        # Permute columns, preserving row/col marginals
        for _ in range(n_permutations):
            perm = rng.permutation(n)
            val = diagonal_mean_from_pairs(sim, perm)
            if not np.isnan(val):
                null_distribution.append(val)

    elif method == 'shuffle_entries':
        # Shuffle all non-NaN entries across the matrix
        valid_idx = np.argwhere(mask_valid)
        values_orig = sim[mask_valid].copy()
        for _ in range(n_permutations):
            values = values_orig.copy()
            rng.shuffle(values)
            mat = np.full_like(sim, np.nan)
            mat[mask_valid] = values
            diag_vals_sh = np.array([mat[i, i] for i in range(n) if not np.isnan(mat[i, i])])
            if diag_vals_sh.size > 0:
                null_distribution.append(np.nanmean(diag_vals_sh))

    elif method == 'grouped_permute':
        # Permute columns within groups. group_labels must be length n.
        if group_labels is None:
            raise ValueError("group_labels required for grouped_permute")
        group_labels = np.asarray(group_labels)
        if len(group_labels) != n:
            raise ValueError("group_labels must have length n (number of rows/cols)")
        # build index lists per group
        groups = {}
        for i, g in enumerate(group_labels):
            groups.setdefault(g, []).append(i)

        for _ in range(n_permutations):
            perm = np.arange(n)
            for g, idxs in groups.items():
                if len(idxs) > 1:
                    perm_group = rng.permutation(idxs)
                    perm[idxs] = perm_group
                else:
                    perm[idxs] = idxs
            val = diagonal_mean_from_pairs(sim, perm)
            if not np.isnan(val):
                null_distribution.append(val)

    else:
        raise ValueError("Unknown method. Choose 'permute_pairs','shuffle_entries' or 'grouped_permute'.")

    null_distribution = np.array(null_distribution)
    # p-value: fraction of null >= observed (one-sided test)
    p_value = np.mean(null_distribution >= mean_diag) if null_distribution.size > 0 else np.nan

    results = {
        "mean_diag": mean_diag,
        "mean_offdiag": mean_offdiag,
        "contrast": contrast,
        "p_value": p_value,
        "null_distribution": null_distribution,
        "method": method
    }

    # Optionally plot histogram
    if plot_hist:
        plt.figure(figsize=(6, 4))
        sns.histplot(null_distribution, bins=50, kde=True, color='C0')
        # mark observed
        ymin, ymax = plt.ylim()
        # small arrow marker just above x axis
        plt.annotate('', xy=(mean_diag, ymax*0.03), xytext=(mean_diag, ymax*0.20),
                     arrowprops=dict(facecolor='red', edgecolor='red', shrink=0.1, width=4, headwidth=8))
        plt.axvline(mean_diag, color='red', linestyle='--', label=f'obs mean={mean_diag:.3f}')
        plt.xlabel('Mean diagonal similarity (null distribution)')
        plt.title(plot_title or f"Null histogram ({method})  p={p_value:.4f}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return results





def band_strength_test(sim_matrix,
                       offsets=[1, 2],
                       n_permutations=1000,
                       random_state=None,
                       method="permute_pairs",
                       plot_hist=True):
    """
    Test whether the main diagonal (offset=0) is significantly stronger than near-diagonal bands.
    
    Parameters
    ----------
    sim_matrix : (n,n) array
        Similarity matrix.
    offsets : list of int
        Offsets (>=1) to compare against diagonal.
    n_permutations : int
        Number of permutations.
    random_state : int or None
        RNG seed.
    method : str
        'permute_pairs' or 'shuffle_entries'.
    plot_hist : bool
        If True, plot null distribution vs observed.
    
    Returns
    -------
    results : dict
        {
          "obs_diag": mean diag,
          "obs_neighbors": mean of offsets,
          "contrast": obs_diag - obs_neighbors,
          "p_value": float,
          "null_distribution": np.ndarray of contrasts under null
        }
    """
    rng = np.random.default_rng(random_state)
    n = sim_matrix.shape[0]
    sim = np.asarray(sim_matrix, dtype=float)
    
    def band_mean(mat, offset):
        vals = [mat[i, i+offset] for i in range(n-offset) if not np.isnan(mat[i, i+offset])]
        vals += [mat[i+offset, i] for i in range(n-offset) if not np.isnan(mat[i+offset, i])]
        return np.nan if not vals else np.nanmean(vals)
    
    # observed
    obs_diag = band_mean(sim, 0)
    obs_neighbors = np.nanmean([band_mean(sim, k) for k in offsets])
    contrast_obs = obs_diag - obs_neighbors
    
    null_distribution = []
    for _ in range(n_permutations):
        if method == "permute_pairs":
            perm = rng.permutation(n)
            mat = sim[:, perm]
        elif method == "shuffle_entries":
            mask = ~np.isnan(sim)
            values = sim[mask].copy()
            rng.shuffle(values)
            mat = np.full_like(sim, np.nan)
            mat[mask] = values
        else:
            raise ValueError("Unknown method")
        
        d = band_mean(mat, 0)
        neigh = np.nanmean([band_mean(mat, k) for k in offsets])
        if np.isnan(d) or np.isnan(neigh):
            continue
        null_distribution.append(d - neigh)
    
    null_distribution = np.array(null_distribution)
    p_value = np.mean(null_distribution >= contrast_obs)
    
    # plotting
    if plot_hist:
        plt.figure(figsize=(6,4))
        sns.histplot(null_distribution, bins=50, kde=True, color="C0")
        plt.axvline(contrast_obs, color="red", linestyle="--", 
                    label=f"obs contrast={contrast_obs:.3f}")
        plt.xlabel("Diagonal - neighbor band mean (null)")
        plt.ylabel("Count")
        plt.title(f"Band contrast test (p={p_value:.4f})")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    return {
        "obs_diag": obs_diag,
        "obs_neighbors": obs_neighbors,
        "contrast": contrast_obs,
        "p_value": p_value,
        "null_distribution": null_distribution
    }



def flux_map_to_vector(flow_field_data, hour, site_id=1, bin_size=20, direction_filter="away"):
    """
    Convert an hourly flux field into a flattened flux magnitude vector.
    """
    bin_stats, _ = plot_flux_field_for_hour(
        flow_field_data, hour, site_id=site_id, bin_size=bin_size,
        direction_filter=direction_filter, ax=plt.gca(), 
        vmin=0, vmax=None, cmap="Reds", show_colorbar=False
    )
    plt.close()  # suppress plot output
    
    if bin_stats is None or len(bin_stats) == 0:
        return None

    # grid shape (same as binning scheme)
    x_bins = np.arange(0, 1920 + bin_size, bin_size)
    y_bins = np.arange(0, 1080 + bin_size, bin_size)
    grid = np.zeros((len(y_bins)-1, len(x_bins)-1))

    for _, row in bin_stats.iterrows():
        grid[int(row["y_bin"]), int(row["x_bin"])] = row["flux_mag"]

    return grid.flatten()

def build_flux_dataset(days_period, num_days, site_id=1, direction_filter="away", bin_size=20):
    """
    Build a dataframe where each row is an hour (day, hour, site),
    and the vector is the flattened flux map.
    """
    records = []
    for d in range(num_days):
        day = days_period + d
        for hour in range(24):
            flow_data = get_flow_field_data_for_hour(day, hour, site_id=site_id)
            if not flow_data:
                continue
            vec = flux_map_to_vector(flow_data, hour, site_id, bin_size, direction_filter)
            if vec is not None:
                records.append({
                    "day": str(day),
                    "hour": hour,
                    "vector": vec
                })
    return pd.DataFrame(records)

def cluster_flux_maps(df, n_components=20, n_clusters=6, method="kmeans"):
    """
    Reduce dimensionality and cluster flux map vectors.
    """
    # Stack vectors
    X = np.vstack(df["vector"].values)

    # Dimensionality reduction
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
    print("\nCumulative explained variance:")
    print(cumulative_explained_variance)

    # Clustering
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=0)
        labels = model.fit_predict(X_pca)
    else:
        raise ValueError("Only kmeans implemented here")
    
    df["cluster"] = labels
    return df, X_pca, model

def plot_clusters(df, X_pca):
    """
    Quick 2D visualization using first two PCA components.
    """
    
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=df["cluster"], cmap="tab10", alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Clusters of flux maps across days/hours")
    plt.colorbar(scatter, label="Cluster")
    plt.show()
    plt.figure(figsize=(8,5))

    plt.hist(df[df.cluster==0].hour, 
            bins=np.arange(25), 
            alpha=0.5, 
            label='Cluster 0', 
            align='left')

    plt.hist(df[df.cluster==1].hour, 
            bins=np.arange(25), 
            alpha=0.5, 
            label='Cluster 1', 
            align='left')

    plt.xticks(range(24))
    plt.xlabel("Hour of Day")
    plt.ylabel("Count")
    plt.legend()
    plt.show()



def cumulative_flux_map_all_days(days_list, site_id=1, bin_size=20, direction_filter="away"):
    """
    Build a cumulative flux magnitude grid across all days and hours for one site and direction.
    Returns the flux grid and bin edges.
    """
    # grid setup
    x_bins = np.arange(0, 1920 + bin_size, bin_size)
    y_bins = np.arange(0, 1080 + bin_size, bin_size)
    grid = np.zeros((len(y_bins)-1, len(x_bins)-1))

    # loop through all days
    for d in days_list:
        flow_field_data = get_flow_field_data_for_day(day=d, site_id=site_id)

        # loop through 24 hours in each day
        for hour in range(24):
            hourly_data = flow_field_data[hour]   # list of DataFrames for this hour
            if not hourly_data:
                continue

            bin_stats, _ = plot_flux_field_for_hour(
                hourly_data, hour, site_id=site_id, bin_size=bin_size,
                direction_filter=direction_filter, ax=plt.gca(),
                vmin=0, vmax=None, cmap="Reds", show_colorbar=False
            )
            plt.close()

            if bin_stats is None or len(bin_stats) == 0:
                continue

            # accumulate flux into grid
            for _, row in bin_stats.iterrows():
                grid[int(row["y_bin"]), int(row["x_bin"])] += row["flux_mag"]

    return grid, x_bins, y_bins






def skeleton_to_graph(sk):
    """
    Convert boolean skeleton image (sk: 2D bool) to an undirected NetworkX graph.
    Nodes are (x, y) = (col, row) tuples for compatibility with plotting.
    """
    rows, cols = sk.shape
    G = nx.Graph()
    ys, xs = np.nonzero(sk)
    for x, y in zip(xs, ys):
        G.add_node((x, y))
    for x, y in zip(xs, ys):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx_, ny_ = x + dx, y + dy
                if 0 <= nx_ < cols and 0 <= ny_ < rows and sk[ny_, nx_]:
                    G.add_edge((x, y), (nx_, ny_))
    return G


def find_nest_junctions(sk, nest_mask, max_dist=30, n_points=3, min_spacing=10):
    """
    Find skeleton pixels closest to the nest mask, ensuring junctions are at least
    'min_spacing' apart.
    
    Parameters
    ----------
    sk : 2D bool
        Skeletonized binary mask (low-res grid).
    nest_mask : 2D bool or uint8
        Binary nest region mask (high-res or original).
    max_dist : int
        Maximum distance (pixels) from nest mask to consider valid junctions.
    n_points : int
        Maximum number of junctions to return.
    min_spacing : int
        Minimum allowed spacing between junction points (pixels).
    """
    # Resize nest mask to match skeleton grid
    nest_mask = resize(
        nest_mask.astype(float),
        sk.shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False
    ) > 0.5

    # Distance to nearest nest pixel
    dist = distance_transform_edt(~nest_mask)

    # Skeleton coordinates (y, x)
    ys, xs = np.nonzero(sk)
    yx_skel = np.column_stack((ys, xs))
    distances = dist[ys, xs]

    # Sort skeleton pixels by proximity to nest (ascending)
    sorted_idx = np.argsort(distances)
    sorted_coords = yx_skel[sorted_idx]

    # Select distinct junctions
    selected = []
    for p in sorted_coords:
        if dist[p[0], p[1]] > max_dist:
            break  # too far from nest, stop
        if len(selected) == 0:
            selected.append(p)
        else:
            dists = np.linalg.norm(np.array(selected) - p, axis=1)
            if np.all(dists >= min_spacing):
                selected.append(p)
        if len(selected) >= n_points:
            break

    # Convert to (x, y) for plotting/graph use
    close_pts = [(x, y) for (y, x) in selected]
    return close_pts


def extract_all_trails_from_junctions(G, nest_junctions, endpoints, min_length=10, method='all_paths', max_depth=100, max_trails=100):
    """
    Extract trails from nest junctions to endpoints using different strategies.
    
    Parameters:
    -----------
    G : NetworkX graph
        Skeleton graph
    nest_junctions : list
        Starting junction points near nest
    endpoints : list
        Terminal points (degree=1 nodes)
    min_length : int
        Minimum trail length to keep
    method : str
        'all_paths' : Find all simple paths (can be slow for large graphs)
        'shortest_paths' : Use only shortest paths (faster, current approach)
        'dfs_exploration' : Use DFS to explore all possible trails
    max_depth : int
        Maximum depth for DFS exploration (prevents infinite recursion)
    max_trails : int
        Maximum number of trails to find (prevents excessive computation)
    
    Returns:
    --------
    trails : list of numpy arrays
        Trail segments as coordinate arrays
    """
    trails = []
    
    if method == 'shortest_paths':
        # Modified approach: each endpoint gets connected to its closest junction only
        endpoint_to_best_junction = {}  # Maps endpoint -> (junction, path_length)
        
        # First pass: find the shortest path from each junction to each endpoint
        for junc in nest_junctions:
            for end in endpoints:
                try:
                    if nx.has_path(G, junc, end):
                        path = nx.shortest_path(G, junc, end)
                        path_length = len(path)
                        
                        # If this endpoint hasn't been seen before, or this path is shorter
                        if (end not in endpoint_to_best_junction or 
                            path_length < endpoint_to_best_junction[end][1]):
                            endpoint_to_best_junction[end] = (junc, path_length)
                except nx.NetworkXNoPath:
                    continue
        
        # Second pass: create trails for the best junction-endpoint pairs
        for end, (best_junc, path_length) in endpoint_to_best_junction.items():
            if len(trails) >= max_trails:
                break
            try:
                if nx.has_path(G, best_junc, end):
                    path = nx.shortest_path(G, best_junc, end)
                    if len(path) > min_length:
                        trails.append(np.array(path))
            except nx.NetworkXNoPath:
                continue
                    
    elif method == 'all_paths':
        # Find all simple paths between junctions and endpoints
        for junc in nest_junctions:
            for end in endpoints:
                if len(trails) >= max_trails:
                    break
                try:
                    if nx.has_path(G, junc, end):
                        # Get all simple paths (no repeated nodes)
                        paths = list(nx.all_simple_paths(G, junc, end, cutoff=max_depth))  # Limit path length
                        for path in paths:
                            if len(trails) >= max_trails:
                                break
                            if len(path) > min_length:
                                trails.append(np.array(path))
                except (nx.NetworkXNoPath, nx.NetworkXError):
                    continue
            if len(trails) >= max_trails:
                break
                    
    elif method == 'dfs_exploration':
        # Use DFS to systematically explore all trails from each junction
        def dfs_trails(graph, start, max_depth_param=max_depth, max_trails_param=max_trails):
            """DFS to find all trails starting from a junction."""
            visited = set()
            trails_from_start = []
            
            def dfs_recursive(current, path, depth):
                if depth > max_depth_param or current in visited or len(trails_from_start) >= max_trails_param:
                    return
                    
                path.append(current)
                visited.add(current)
                
                # If we hit an endpoint, save this trail
                if graph.degree(current) == 1 and len(path) > min_length:
                    trails_from_start.append(np.array(path))
                
                # Continue exploring neighbors
                for neighbor in graph.neighbors(current):
                    if neighbor not in visited and len(trails_from_start) < max_trails_param:
                        dfs_recursive(neighbor, path.copy(), depth + 1)
                        
                visited.remove(current)
            
            dfs_recursive(start, [], 0)
            return trails_from_start
        
        for junc in nest_junctions:
            if len(trails) >= max_trails:
                break
            remaining_trails = max_trails - len(trails)
            trails.extend(dfs_trails(G, junc, max_depth, remaining_trails))
    
    return trails


def split_trails_from_nest(binary_mask, nest_mask, max_dist=30, n_points=3, min_length=10, min_spacing=10, dilate_kernel_size=5, trail_method='shortest_paths', max_depth=100, max_trails=100, plot=True):
    """Split skeleton into trails that emanate from nest junctions."""
    # Dilate the binary mask to fill gaps before skeletonization
    from skimage.morphology import disk
    kernel = disk(dilate_kernel_size // 2)  # disk kernel for isotropic dilation
    dilated_mask = binary_dilation(binary_mask > 0, kernel)
    
    sk = skeletonize(dilated_mask)
    plt.imshow(sk, cmap="gray", origin="upper")
    plt.show()
    G = skeleton_to_graph(sk)
    degrees = dict(G.degree())
    endpoints = [p for p, d in degrees.items() if d == 1]

    # Find custom junctions near nest
    nest_junctions = find_nest_junctions(sk, nest_mask, max_dist, n_points, min_spacing)

    # Extract trails using specified method
    trails = extract_all_trails_from_junctions(G, nest_junctions, endpoints, min_length, method=trail_method, max_depth=max_depth, max_trails=max_trails)

    if plot:
        plt.figure(figsize=(6,6))
        plt.imshow(sk, cmap="gray", origin="upper")
        plt.scatter([p[0] for p in endpoints], [p[1] for p in endpoints], s=8, c='blue', label="Endpoints")
        plt.scatter([p[0] for p in nest_junctions], [p[1] for p in nest_junctions], s=30, c='red', label="Nest junctions")
        for t in trails:
            plt.plot(t[:,0], t[:,1], lw=1.5)
        plt.legend()
        plt.title(f"Trails from Nest Junctions ({trail_method})")
        plt.show()

    return trails, nest_junctions, endpoints




def expand_segments_to_width(segments, shape, width=2):
    """Dilate skeleton segments into region masks of a given approximate width (in pixels)."""
    struct = disk(width)
    seg_masks = []

    for seg in segments:
        mask = np.zeros(shape, dtype=bool)

        if isinstance(seg, (list, np.ndarray, set)):
            for point in seg:
                if len(point) == 2:
                    x, y = map(int, point)
                    if 0 <= x < shape[1] and 0 <= y < shape[0]:
                        mask[y, x] = True
        # Dilate and store
        mask_dilated = dilation(mask, struct)
        seg_masks.append(mask_dilated)

    return seg_masks





def plot_segmented_masks_grid(seg_masks, background_grid=None, rows=5, cols=5,
                              title_prefix="Trail", mask_color='cyan',
                              alpha=0.6, bin_extent=None):
    """
    Visualize boolean trail masks (seg_masks) in paginated grids of size rows x cols.

    Parameters
    ----------
    seg_masks : list of 2D bool arrays
        Each mask marks pixels belonging to a single trail.
    background_grid : 2D array or None
        Optional heatmap background (e.g., cumulative flux grid). Drawn with origin='upper'.
    rows, cols : int
        Grid layout per page.
    title_prefix : str
        Prefix for subplot titles.
    mask_color : str
        Color to render the mask overlay.
    alpha : float
        Transparency for mask overlay.
    bin_extent : tuple or None
        Optional (x_min, x_max, y_min, y_max) for background extent.
    """
    if not seg_masks:
        print("No segment masks to visualize.")
        return

    per_page = rows * cols
    num_masks = len(seg_masks)
    num_pages = (num_masks + per_page - 1) // per_page

    for page in range(num_pages):
        start = page * per_page
        end = min((page + 1) * per_page, num_masks)
        masks_slice = seg_masks[start:end]

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.0))
        axes = np.atleast_2d(axes)

        for idx in range(rows * cols):
            r = idx // cols
            c = idx % cols
            ax = axes[r, c]

            if idx < len(masks_slice):
                mask = masks_slice[idx]

                # background
                if background_grid is not None:
                    if bin_extent is not None and len(bin_extent) == 4:
                        x_min, x_max, y_min, y_max = bin_extent
                        ax.imshow(background_grid, cmap="hot", origin="upper",
                                  extent=[x_min, x_max, y_min, y_max])
                    else:
                        ax.imshow(background_grid, cmap="hot", origin="upper")

                # overlay mask
                overlay = np.zeros((*mask.shape, 4), dtype=float)
                # set chosen color
                color_map = {'cyan': (0, 1, 1), 'red': (1, 0, 0), 'green': (0, 1, 0), 'blue': (0, 0, 1), 'yellow': (1, 1, 0)}
                rgb = color_map.get(mask_color, (0, 1, 1))
                overlay[mask] = (*rgb, alpha)
                if bin_extent is not None and len(bin_extent) == 4:
                    x_min, x_max, y_min, y_max = bin_extent
                    ax.imshow(overlay, origin='upper', extent=[x_min, x_max, y_min, y_max])
                else:
                    ax.imshow(overlay, origin='upper')

                ax.set_title(f"{title_prefix} {start + idx + 1}")
                ax.set_aspect('equal')
                # Remove the invert_yaxis() call since origin='upper' already handles orientation
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.axis('off')

        plt.tight_layout()
        plt.show()

def remove_subset_trails(trails, tol=2):
    """
    Remove trails that are spatial subsets of longer trails.

    Parameters
    ----------
    trails : list of (N_i, 2) arrays
        List of (x, y) coordinates for each trail.
    tol : float
        Distance tolerance (pixels) to consider points as overlapping.

    Returns
    -------
    filtered : list of (N_i, 2) arrays
        Trails with near-duplicate or fully contained ones removed.
    """

    # Sort by descending length (keep longer trails first)
    trails = sorted(trails, key=lambda t: len(t), reverse=True)
    keep = []

    for i, t1 in enumerate(trails):
        tree1 = cKDTree(t1)
        is_subset = False

        for kept in keep:
            tree2 = cKDTree(kept)
            # Compute nearest neighbor distance from t1 -> kept
            dists, _ = tree2.query(t1, k=1)
            # If all points of t1 are close to kept trail -> subset
            if np.all(dists < tol):
                is_subset = True
                break

        if not is_subset:
            keep.append(t1)

    return keep

def merge_trails_as_graph(trails):
    G = nx.Graph()
    for trail in trails:
        for (x1, y1), (x2, y2) in zip(trail[:-1], trail[1:]):
            G.add_edge((x1, y1), (x2, y2))
    return G

# Then extract unique segments:
def extract_unique_branches(G, nest_junctions, endpoints):
    unique_segments = []
    for junc in nest_junctions:
        for end in endpoints:
            try:
                if nx.has_path(G, junc, end):
                        path = nx.shortest_path(G, junc, end)
                        unique_segments.append(np.array(path))
            except Exception as e:
                print(e)
                continue
    return unique_segments




# Database connection and data loading (copied from plotting_ant_counts_from_db.py)
connection = database_helper.create_connection("localhost", "root", "master", "ant_colony_db")
cursor = connection.cursor()
query = f"""
    select Counts.video_id, Counts.yolo_detection_only_csv, Counts.yolo_tracking_with_direction_csv, Counts.herdnet_detection_only_csv, Counts.herdnet_tracking_with_direction_closest_boundary_method_csv,
    Videos.temperature, Videos.humidity, Videos.LUX, Videos.time_stamp, Videos.site_id from Counts INNER JOIN Videos on Counts.video_id=Videos.video_id;
    """
cursor.execute(query)
table_rows = cursor.fetchall()

site_mapping = {1:'beer', 2:'shack', 3:'rain'}

df = pd.DataFrame(table_rows, columns=cursor.column_names)

# Make sure the time_stamp column is in pandas datetime format
df["time_stamp"] = pd.to_datetime(df["time_stamp"])







## Uncomment the line below to pre-process all velocity data for faster future analysis
## This will create CSV files with per-frame velocity data for each video

## site 1: beer, site 2: shack, site 3: rain#
#site = 1
# days_period = pd.Period('2024-08-01', freq='D')
# num_days = 10
# nest_mask = cv2.imread('/home/tarun/Desktop/masks/beer-tree-08-01-2024_to_08-10-2024_center_only.png',0)
#preprocess_all_velocity_data(days_period, num_days, site)


# days_period = pd.Period('2024-10-22', freq='D')
# num_days = 12
# nest_mask = cv2.imread('/home/tarun/Desktop/masks/beer-10-22-2024_to_11-02-2024_center_only.png',0)
#preprocess_all_velocity_data(days_period, num_days, site)

####### rain
#site = 3
# days_period = pd.Period('2024-08-22', freq='D')
# num_days = 12
# nest_mask = cv2.imread('/home/tarun/Desktop/masks/rain-tree-08-22-2024_to_09-02-2024_center_only.png',0)
#preprocess_all_velocity_data(days_period, num_days, site)

# days_period = pd.Period('2024-10-03', freq='D')
# num_days = 17
# nest_mask = cv2.imread('/home/tarun/Desktop/masks/rain-tree-10-03-2024_to_10-19-2024_center_only.png',0)
# preprocess_all_velocity_data(days_period, num_days, site)

# days_period = pd.Period('2024-11-15', freq='D')
# num_days = 21
# nest_mask = cv2.imread('/home/tarun/Desktop/masks/rain-tree-11-15-2024_to_12-06-2024_center_only.png',0)
# preprocess_all_velocity_data(days_period, num_days, site)

# # ##### shack 
site = 2
days_period = pd.Period('2024-08-26', freq='D')
num_days = 25
#nest_mask = cv2.imread('/home/tarun/Desktop/masks/shack-tree-diffuser-08-01-2024_to_08-26-2024_center_only.png',0)
nest_mask = cv2.imread('/home/tarun/Desktop/masks/shack-tree-diffuser-08-26-2024_to_09-18-2024_center_only.png',0)

#preprocess_all_velocity_data(days_period, num_days, site)

#scatter_plot_velocity_for_day_range(days_period, 23, site)



###############
days_list = [days_period + i for i in range(num_days)]
grid, x_bins, y_bins = cumulative_flux_map_all_days(days_list, site_id=site, bin_size=20, direction_filter="away")

plt.imshow(grid, cmap="hot", origin="upper",
           extent=[x_bins[0], x_bins[-1], y_bins[-1], y_bins[0]])  # flip y for correct orientation
plt.colorbar(label="Cumulative Flux (ant·pixels/s)")
plt.title(f"Cumulative Flux Map (Away, Site {site}, {num_days} Days)")
plt.show()
################ 


pct = 70
binary_mask = grid >= np.percentile(grid[grid>0], pct)
plt.imshow(binary_mask, cmap="gray")
plt.show()

### 2. threshold mask and split skeleton segments

_, nest_mask = cv2.threshold(nest_mask, 127, 255, cv2.THRESH_BINARY)
nest_mask = ~nest_mask
segments, nest_junctions, endpoints = split_trails_from_nest(binary_mask, nest_mask, max_dist=5, n_points=20, min_length=10, min_spacing=3, dilate_kernel_size=0, trail_method='shortest_paths', max_depth=50, max_trails=100, plot=True)


segments = remove_subset_trails(segments, tol=2)


G = merge_trails_as_graph(segments)
filtered_segments = extract_unique_branches(G, nest_junctions, endpoints)

filtered_segments = remove_subset_trails(filtered_segments, tol=2)

print(f"Before: {len(segments)}, After filtering: {len(filtered_segments)}")


# 3. expand to width
seg_masks = expand_segments_to_width(filtered_segments, shape=binary_mask.shape, width=1)

extent = (x_bins[0], x_bins[-1], y_bins[0], y_bins[-1])

plot_segmented_masks_grid(
    seg_masks,
    background_grid=grid,         # or None for mask-only
    rows=5, cols=5,
    title_prefix="Trail",
    mask_color='cyan',
    alpha=0.6,
    bin_extent=extent
)


# 4. overlay
combined_mask = np.zeros_like(binary_mask, dtype=int)
for i, seg in enumerate(seg_masks):
    combined_mask[seg] = i + 1  # label each trail

plt.imshow(combined_mask)
plt.show()


plt.figure(figsize=(8,8))

flux_img = plt.imshow(grid, cmap="hot", origin="upper")
plt.colorbar(flux_img, label="Cumulative Flux (ant·pixels/s)")

# 2️⃣ Overlay — trail mask boundaries (clearer and doesn’t obscure heatmap)
from skimage.segmentation import find_boundaries
boundaries = find_boundaries(combined_mask, mode="outer")

plt.imshow(np.ma.masked_where(~boundaries, boundaries), 
           cmap="cool", alpha=0.9, origin="upper")

# 3️⃣ (Optional) label each trail
from scipy.ndimage import center_of_mass
for i in range(1, combined_mask.max() + 1):
    y, x = center_of_mass(combined_mask == i)
    plt.text(x, y, str(i), color="white", ha="center", va="center", 
             fontsize=9, weight="bold")
    

plt.title("Cumulative Flux Map with Trail Boundaries", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.show()



######## Clustering flux maps ##################
 
#df_clustering = build_flux_dataset(days_period, num_days, site_id=site, direction_filter="away", bin_size=20)
#df_clustering, X_pca, model = cluster_flux_maps(df_clustering, n_components=40, n_clusters=2)
#plot_clusters(df_clustering, X_pca)
#####################################################



# Assuming you have data organized like: flow_field_data[hour] = list of dataframes for that hour
dayA_data = get_flow_field_data_for_day(day=days_period + 3, site_id=site)
dayB_data = get_flow_field_data_for_day(day=days_period + 4, site_id=site)

### direction_filter is either ["away", "toward", "same_day_away_vs_toward"]
sim_matrix = plot_flux_similarity_matrix(dayA_data, dayB_data, site_id=site, 
                                        bin_size=20, direction_filter="away", 
                                        metric="cosine")


# Quantify diagonal strength through shuffling diagnol to create null distribution
# res_perm = quantify_diagonal_strength_extended(sim_matrix,
#                                                n_permutations=1000,
#                                                random_state=42,
#                                                method='permute_pairs',
#                                                plot_title='Row/Col permutation')

res_shuffle = quantify_diagonal_strength_extended(sim_matrix,
                                                  n_permutations=1000,
                                                  random_state=42,
                                                  method='shuffle_entries',
                                                  plot_title='Full entry shuffle')


# print ('method:', res_perm['method'])
# print("Mean diagonal similarity:", res_perm["mean_diag"])
# print("Mean off-diagonal similarity:", res_perm["mean_offdiag"])
# print("Contrast (diag/offdiag):", res_perm["contrast"])
# print("Permutation p-value:", res_perm["p_value"])

print ('method:', res_shuffle['method'])
print("random shuffling full matrix p-value:", res_shuffle["p_value"])


results = band_strength_test(sim_matrix, n_permutations=1000,
                            random_state=42,
                            method="shuffle_entries",
                            plot_hist=True)

print('band contrast method p value', results["p_value"])



#import sys; sys.exit(0)

### Plot daily grids (6x4) of flux maps per hour for a given day and given direction 
for day in range(1,12):
    plot_daily_flux_grid(days_period+day, site_id=site, direction_filter="away", bin_size=20, normalize='per_hour')
    #plot_daily_flux_grid(days_period+2, site_id=site, direction_filter="toward", bin_size=20, normalize='per_hour')


### Plot flow/flux fields for a given hour on a given day. Compare 'away' vs 'toward' flux maps for a given hour.
for n in range(1,9):
    for hour in range(18,21):
        flow_data = get_flow_field_data_for_hour(days_period+n, hour, site_id=site)
        #plot_flow_field_for_hour(flow_data, hour, site_id=site, bin_size=20, direction_filter='away')  # Away direction only
        #plot_flow_field_for_hour(flow_data, hour, site_id=site, bin_size=20, direction_filter='toward')  # Toward direction only
        #plot_flow_field_for_hour(flow_data, hour, site_id=site, bin_size=20, direction_filter='unknown')  # Toward direction only
        #plot_flow_field_for_hour(flow_data, hour, site_id=1, bin_size=30, direction_filter=None)  # All directions
        #plt.tight_layout()
        #plt.show()
        
        plot_flow_field_comparison(flow_data, hour, site_id=site, bin_size=20)  # Side-by-side comparison
        plot_flux_field_comparison(flow_data, hour, site_id=site, bin_size=20)
