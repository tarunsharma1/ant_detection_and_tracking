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
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor



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
        #print(f'Loading velocity data from {velocity_csv_path}')
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
    
    # Get number of frames for normalization
    num_frames = moving_data['frame_number'].nunique() if 'frame_number' in moving_data.columns else 1
    
    # Flux = (average number of ants per frame) × (average velocity per ant) per bin
    bin_stats = moving_data.groupby(['x_bin', 'y_bin']).agg({
        'u': 'mean',           # average u component per ant
        'v': 'mean',            # average v component per ant  
        'velocity': 'count'     # total ant detections across all frames
    }).reset_index().dropna()
    
    # Convert to average ants per frame and average velocity
    bin_stats['avg_ants_per_frame'] = bin_stats['velocity'] / num_frames
    bin_stats['avg_u'] = bin_stats['u']  # already averaged
    bin_stats['avg_v'] = bin_stats['v']   # already averaged
    
    # Flux = average ants per frame × average velocity
    bin_stats['u'] = bin_stats['avg_ants_per_frame'] * bin_stats['avg_u']
    bin_stats['v'] = bin_stats['avg_ants_per_frame'] * bin_stats['avg_v']
    
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
    # Use standard grid size to match plot_flux_field_for_hour
    x_bins = np.arange(0, 1920 + bin_size, bin_size)
    y_bins = np.arange(0, 1080 + bin_size, bin_size)
    
    # Get number of frames for normalization
    num_frames = moving_data['frame_number'].nunique() if 'frame_number' in moving_data.columns else 1
    
    # Bin the data
    moving_data['x_bin'] = pd.cut(moving_data['center_x'], bins=x_bins, labels=False)
    moving_data['y_bin'] = pd.cut(moving_data['center_y'], bins=y_bins, labels=False)
    
    # Compute vector components
    moving_data['u'] = np.cos(moving_data['angle']) * moving_data['velocity']
    moving_data['v'] = -np.sin(moving_data['angle']) * moving_data['velocity']
    
    # Calculate flux as (average ants per frame) × (average velocity)
    bin_flux = moving_data.groupby(['x_bin', 'y_bin']).agg({
        'u': 'mean',           # average u component per ant
        'v': 'mean',           # average v component per ant
        'velocity': 'count'    # total ant detections across all frames
    })
    
    # Remove rows with NaN (bins with no data)
    bin_flux = bin_flux.dropna()
    
    if len(bin_flux) == 0:
        print(f"No valid flux data for hour {hour}")
        return None, None
    
    # Convert to average ants per frame and average velocity
    bin_flux['avg_ants_per_frame'] = bin_flux['velocity'] / num_frames
    bin_flux['avg_u'] = bin_flux['u']  # already averaged
    bin_flux['avg_v'] = bin_flux['v']   # already averaged
    
    # Flux = average ants per frame × average velocity
    bin_flux['u'] = bin_flux['avg_ants_per_frame'] * bin_flux['avg_u']
    bin_flux['v'] = bin_flux['avg_ants_per_frame'] * bin_flux['avg_v']
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
            # compute per-hour flux magnitudes using correct flux calculation
            all_df = pd.concat(flow_data, ignore_index=True)
            
            # Filter by direction if specified (to match what will be plotted)
            if direction_filter:
                all_df = all_df[all_df['direction'] == direction_filter]
                if len(all_df) == 0:
                    continue
            
            moving = all_df[all_df['velocity'] > 0].copy()
            if moving.shape[0] == 0:
                continue
            
            # Get number of frames for normalization
            num_frames = moving['frame_number'].nunique() if 'frame_number' in moving.columns else 1
            
            # Compute vector components
            moving['u'] = np.cos(moving['angle'].astype(float)) * moving['velocity'].astype(float)
            moving['v'] = -np.sin(moving['angle'].astype(float)) * moving['velocity'].astype(float)
            moving['x_bin'] = pd.cut(moving['center_x'], bins=np.arange(0, 1920 + bin_size, bin_size), labels=False)
            moving['y_bin'] = pd.cut(moving['center_y'], bins=np.arange(0, 1080 + bin_size, bin_size), labels=False)
            
            # Flux = (average number of ants per frame) × (average velocity per ant) per bin
            bin_flux = moving.groupby(['x_bin', 'y_bin']).agg({
                'u': 'mean',           # average u component per ant
                'v': 'mean',           # average v component per ant
                'velocity': 'count'    # total ant detections across all frames
            })
            bin_flux = bin_flux.dropna()
            
            if bin_flux.shape[0] > 0:
                # Convert to average ants per frame and average velocity
                bin_flux['avg_ants_per_frame'] = bin_flux['velocity'] / num_frames
                bin_flux['avg_u'] = bin_flux['u']  # already averaged
                bin_flux['avg_v'] = bin_flux['v']  # already averaged
                
                # Flux = average ants per frame × average velocity
                bin_flux['u'] = bin_flux['avg_ants_per_frame'] * bin_flux['avg_u']
                bin_flux['v'] = bin_flux['avg_ants_per_frame'] * bin_flux['avg_v']
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
            # compute per-hour fluxes using correct flux calculation
            all_df = pd.concat(flow_data, ignore_index=True)
            
            # Filter by direction if specified (to match what will be plotted)
            if direction_filter:
                all_df = all_df[all_df['direction'] == direction_filter]
                if len(all_df) == 0:
                    ax.set_title(f"Hour {hour:02d}:00 — no data for direction '{direction_filter}'")
                    ax.axis('off')
                    continue
            
            moving = all_df[all_df['velocity'] > 0].copy()
            if moving.shape[0] == 0:
                ax.set_title(f"Hour {hour:02d}:00 — no movers")
                ax.axis('off')
                continue
            
            # Get number of frames for normalization
            num_frames = moving['frame_number'].nunique() if 'frame_number' in moving.columns else 1
            
            # Compute vector components
            moving['u'] = np.cos(moving['angle'].astype(float)) * moving['velocity'].astype(float)
            moving['v'] = -np.sin(moving['angle'].astype(float)) * moving['velocity'].astype(float)
            moving['x_bin'] = pd.cut(moving['center_x'], bins=np.arange(0, 1920 + bin_size, bin_size), labels=False)
            moving['y_bin'] = pd.cut(moving['center_y'], bins=np.arange(0, 1080 + bin_size, bin_size), labels=False)
            
            # Flux = (average number of ants per frame) × (average velocity per ant) per bin
            bin_flux = moving.groupby(['x_bin', 'y_bin']).agg({
                'u': 'mean',           # average u component per ant
                'v': 'mean',           # average v component per ant
                'velocity': 'count'    # total ant detections across all frames
            })
            bin_flux = bin_flux.dropna()
            
            if bin_flux.shape[0] > 0:
                # Convert to average ants per frame and average velocity
                bin_flux['avg_ants_per_frame'] = bin_flux['velocity'] / num_frames
                bin_flux['avg_u'] = bin_flux['u']  # already averaged
                bin_flux['avg_v'] = bin_flux['v']  # already averaged
                
                # Flux = average ants per frame × average velocity
                bin_flux['u'] = bin_flux['avg_ants_per_frame'] * bin_flux['avg_u']
                bin_flux['v'] = bin_flux['avg_ants_per_frame'] * bin_flux['avg_v']
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
    #                  fmt=".", c='k', ecolor='k', elinewidth=1, label='average speed')

    ## Plot away vs toward velocities
    axes.scatter(x_axis_away, list(itertools.chain.from_iterable(y_list_away)), marker='.', c='g', s=20, alpha=0.3)
    axes.errorbar(range(24) + np.random.uniform(low=0.10, high=0.10, size=24), bootstrapped_means_away, 
                yerr=[y_away_err_lower, y_away_err_upper], fmt=".", c='g', ecolor='k', elinewidth=1, label='away')

    axes.scatter(x_axis_toward, list(itertools.chain.from_iterable(y_list_toward)), marker='.', c='r', s=20, alpha=0.3)
    axes.errorbar(range(24) + np.random.uniform(low=-0.10, high=-0.10, size=24), bootstrapped_means_toward, 
                 yerr=[y_toward_err_lower, y_toward_err_upper], fmt=".", c='r', ecolor='k', elinewidth=1, label='toward')

    axes.scatter(x_axis_toward, list(itertools.chain.from_iterable(y_list_loitering)), marker='.', c='y', s=20, alpha=0.3)
    axes.errorbar(range(24) + np.random.uniform(low=-0.10, high=-0.10, size=24), bootstrapped_means_loitering, 
                 yerr=[y_loitering_err_lower, y_loitering_err_upper], fmt=".", c='y', ecolor='k', elinewidth=1, label='unknown')
    
    axes.legend()
    plt.title('Average ant speed Rain tree 2024-11-15 to 2024-12-06')
    plt.xticks(range(24))
    plt.xlabel('Hour')
    plt.ylabel('Average Speed (pixels/second)')
    plt.show()


def scatter_plot_velocity_by_direction_aggregated(start_day, number_of_days=0, site_id=1):
    """
    Plot (away - toward) velocity difference for every hour across all days.
    X-axis: hours (0-23)
    Y-axis: (away - toward) velocity difference
    Shows all individual values with alpha and bootstrapped mean with 95% CI.
    Similar to scatter_plot_by_direction_aggregated but for velocities instead of counts.
    """
    velocities_away_per_hour_across_days, velocities_toward_per_hour_across_days, velocities_loitering_per_hour_across_days, velocities_total, temperature, humidity, lux = get_velocity_data_from_db(start_day, number_of_days, site_id)
    
    # Save velocities data as pickle file
    import pickle
    site_name = {1: 'beer', 2: 'shack', 3: 'rain'}.get(site_id, f'site_{site_id}')
    end_day = start_day + number_of_days - 1
    bout_name = f"{site_name}_{str(start_day)}_{str(end_day)}"
    
    save_dir = '/home/tarun/Desktop/plots_for_committee_meeting/velocity_analysis'
    os.makedirs(save_dir, exist_ok=True)
    
    pickle_path = os.path.join(save_dir, f"velocities_{bout_name}.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump({
            'velocities_away_per_hour_across_days': velocities_away_per_hour_across_days,
            'velocities_toward_per_hour_across_days': velocities_toward_per_hour_across_days,
            'start_day': start_day,
            'number_of_days': number_of_days,
            'site_id': site_id
        }, f)
    print(f"✅ Saved velocities data to {pickle_path}")
    
    # Calculate (away - toward) velocity difference for each hour across all days
    difference_per_hour = {}  # {hour: [differences]}
    bootstrapped_means = []
    err_lower = []
    err_upper = []
    hours = []
    
    for h in range(24):
        away_values = velocities_away_per_hour_across_days[h]
        toward_values = velocities_toward_per_hour_across_days[h]
        
        # Calculate difference for each day at this hour
        # Ensure both lists have the same length
        min_len = min(len(away_values), len(toward_values))
        differences = [away_values[i] - toward_values[i] for i in range(min_len)]
        
        if len(differences) > 0:
            difference_per_hour[h] = differences
            hours.append(h)
            
            # Bootstrap with 10000 samples
            bootstrapped_data = bootstrap(differences, 10000)
            bootstrapped_mean = np.mean(bootstrapped_data)
            conf_min, conf_max = confidence_interval(bootstrapped_data)
            
            bootstrapped_means.append(bootstrapped_mean)
            err_lower.append(max(0, bootstrapped_mean - conf_min))
            err_upper.append(max(0, conf_max - bootstrapped_mean))
    
    # Create plot
    fig, axes = plt.subplots()
    
    # Plot all individual values with alpha and jitter
    for h in hours:
        differences = difference_per_hour[h]
        jitter = np.random.uniform(low=-0.15, high=0.15, size=len(differences))
        axes.scatter([h] * len(differences) + jitter, differences, 
                    marker='.', c='steelblue', s=20, alpha=0.3)
    
    # Plot bootstrapped means with error bars
    axes.errorbar(hours, bootstrapped_means, 
                 yerr=[err_lower, err_upper], 
                 fmt='o', c='steelblue', ecolor='k', elinewidth=2, capsize=5, capthick=2, 
                 markersize=8, label='Mean ± 95% CI', zorder=10)
    
    # Add horizontal line at y=0 (no difference)
    axes.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Formatting
    axes.set_xticks(range(0, 24, 1))
    axes.set_xlabel('Hour of Day')
    axes.set_ylabel('(Away - Toward) Speed Difference\n(pixels/second)')
    
    # Get site name for title
    site_name = {1: 'beer', 2: 'shack', 3: 'rain'}.get(site_id, f'site_{site_id}')
    axes.set_title(f'{site_name.title()} Tree - (Away - Toward) Velocity Difference by Hour\n{start_day} to {start_day + number_of_days - 1}')
    
    axes.legend(loc='best')
    axes.grid(True, alpha=0.3)
    
    plt.tight_layout()
    #plt.savefig('/home/tarun/Desktop/plots_for_committee_meeting/velocity_analysis/velocity_diffs_plot.png', dpi=300, bbox_inches='tight')
    plt.show()


def linear_regression():
    '''
      All data pooled together, linear regression of ant counts as independent variable, dependent variables being temperature, humidity, time etc.
    '''
    all_ant_velocities = []
    all_temperatures = []
    all_humidity = []
    all_lux = []
    all_hours = []
    all_site_ids = []
    all_months = []

    data_collected = [(pd.Period('2024-08-22', freq='D'), 12, 3), (pd.Period('2024-10-03', freq='D'), 17, 3), (pd.Period('2024-11-15', freq='D'), 22, 3), (pd.Period('2024-08-01', freq='D'), 10, 1), (pd.Period('2024-10-22', freq='D'), 12, 1), (pd.Period('2024-08-01', freq='D'), 26, 2), (pd.Period('2024-08-26', freq='D'), 24, 2)]
    
    for (start_day, number_of_days, site_id) in data_collected:
        velocities_away_per_hour_across_days, velocities_toward_per_hour_across_days, velocities_loitering_per_hour_across_days, velocities_total, temperature, humidity, lux = get_velocity_data_from_db(start_day, number_of_days, site_id)
        
        for hour in range(24):
            all_ant_velocities.extend(velocities_total[hour])
            all_temperatures.extend(temperature[hour])
            all_humidity.extend(humidity[hour])
            all_lux.extend(lux[hour])
            all_hours.extend([hour]*len(velocities_total[hour]))
            ### below are our random effects (categorical variables)
            all_site_ids.extend([site_id]*len(velocities_total[hour]))
            all_months.extend([start_day.month] * len(velocities_total[hour]))


    df_linear_reg = pd.DataFrame({'ant_velocities': all_ant_velocities, 'temperature': all_temperatures, 'humidity': all_humidity, 'lux':all_lux, 'time':all_hours, 'site':all_site_ids, 'month': all_months})
    
    df_linear_reg['site'] = df_linear_reg['site'].astype('category')
    df_linear_reg['month'] = df_linear_reg['month'].astype('category')
    
    df_linear_reg['log_ant_velocities'] = np.log(df_linear_reg['ant_velocities'])

    ## convert time to cyclic variables to avoid jump between 23:00 and 0:00
    df_linear_reg['time_sin'] = np.sin(2 * np.pi * df_linear_reg['time'] / 24)
    df_linear_reg['time_cos'] = np.cos(2 * np.pi * df_linear_reg['time'] / 24)

    
    # model = smf.mixedlm("log_ant_velocities ~ time_sin + time_cos + temperature + humidity + lux", 
    #     df_linear_reg, 
    #     groups=df_linear_reg["site"])
    #     #re_formula="~temperature")

    # result = model.fit(reml=False)
    # print(result.summary())


    ### before fitting lets standardize predictors 
    predictors = ['time_sin', 'time_cos', 'temperature', 'humidity', 'lux']

    # Initialize scaler
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    # Fit scaler and transform predictors
    df_linear_reg_std = df_linear_reg.copy()
    df_linear_reg_std[predictors] = scaler.fit_transform(df_linear_reg_std[predictors])

    # Now fit mixed effects model with standardized predictors
    import statsmodels.formula.api as smf

    model = smf.mixedlm("log_ant_velocities ~ time_sin + time_cos + temperature + humidity + lux",
                            df_linear_reg_std,
                            groups=df_linear_reg_std["site"])
    result = model.fit(reml=False)
    print(result.summary())


    ### calculate AIC/BIC
    log_likelihood = result.llf
    n1 = result.nobs
    k = result.df_modelwc + 1
    aic = 2*k - 2*log_likelihood
    bic = np.log(n1)*k - 2*log_likelihood
    print (f' AIC is {aic} and BIC is {bic}')


    ### plot residuals
    plt.figure(figsize=(8,6))
    fitted = result.fittedvalues
    residuals = result.resid
    sns.scatterplot(x=fitted, y=residuals)
    plt.axhline(0, linestyle='--', color='red')
    plt.show()


    ## check for correlations and multicollinearity
    predictors = df_linear_reg_std[['temperature', 'humidity', 'lux', 'time_sin', 'time_cos']]
    corr = predictors.corr()
    print (corr)
    X = sm.add_constant(predictors)
    # Calculate VIF for each predictor
    vif = pd.DataFrame()
    vif['Variable'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    print(vif)

    ## plot model predictions against actual values
    fitted_values = result.fittedvalues
    actual_values = df_linear_reg_std['log_ant_velocities']
    sns.regplot(x=actual_values, y=fitted_values, scatter_kws={'alpha':0.5}, line_kws={'color':'orange'})
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')
    plt.show()

    import ipdb;ipdb.set_trace()



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
        moving_data['x_bin'] = pd.cut(moving_data['center_x'], bins=x_bins, labels=False)
        moving_data['y_bin'] = pd.cut(moving_data['center_y'], bins=y_bins, labels=False)
        
        # Compute vector components for each ant
        moving_data['u'] = np.cos(moving_data['angle']) * moving_data['velocity']
        moving_data['v'] = -np.sin(moving_data['angle']) * moving_data['velocity']
        
        # Get number of frames for normalization (same as plot_flux_field_for_hour)
        num_frames = moving_data['frame_number'].nunique() if 'frame_number' in moving_data.columns else 1
        
        # Flux = (average number of ants per frame) × (average velocity per ant) per bin
        # This matches the logic in plot_flux_field_for_hour
        # Keep MultiIndex from groupby for efficient reindexing later
        bin_flux = moving_data.groupby(['x_bin', 'y_bin']).agg({
            'u': 'mean',           # average u component per ant
            'v': 'mean',           # average v component per ant
            'velocity': 'count'    # total ant detections across all frames
        })
        
        # Remove rows with NaN (bins with no data)
        bin_flux = bin_flux.dropna()
        
        # Convert to average ants per frame and average velocity
        bin_flux['avg_ants_per_frame'] = bin_flux['velocity'] / num_frames
        bin_flux['avg_u'] = bin_flux['u']  # already averaged
        bin_flux['avg_v'] = bin_flux['v']  # already averaged
        
        # Flux = average ants per frame × average velocity
        bin_flux['u'] = bin_flux['avg_ants_per_frame'] * bin_flux['avg_u']
        bin_flux['v'] = bin_flux['avg_ants_per_frame'] * bin_flux['avg_v']
        bin_flux['flux_mag'] = np.sqrt(bin_flux['u']**2 + bin_flux['v']**2)

        # Flatten into vector - bin_flux already has MultiIndex from groupby
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







def band_strength_test(sim_matrix,
                       offsets=[1, 2],
                       n_permutations=1000,
                       random_state=None,
                       method="permute_pairs",
                       block1_hours=None,
                       block2_hours=None,
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
        'permute_pairs', 'shuffle_entries', or 'shuffle_blockwise'.
    block1_hours : list of int, optional
        Hours for first block (e.g., night time hours). Required if method='shuffle_blockwise'.
    block2_hours : list of int, optional
        Hours for second block (e.g., day time hours). Required if method='shuffle_blockwise'.
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
    
    # Validate block hours for blockwise shuffling
    if method == "shuffle_blockwise":
        if block1_hours is None or block2_hours is None:
            raise ValueError("block1_hours and block2_hours must be provided when method='shuffle_blockwise'")
        if set(block1_hours) & set(block2_hours):
            raise ValueError("block1_hours and block2_hours must be disjoint")
        if n != 24:
            raise ValueError(f"shuffle_blockwise expects 24 hours (n=24), but got n={n}")
    
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
        elif method == "shuffle_blockwise":
            # Identify row/column indices for each block
            block1_indices = np.array(block1_hours, dtype=int)
            block2_indices = np.array(block2_hours, dtype=int)
            
            # Create a copy of the matrix
            mat = sim.copy()
            
            # Shuffle entries within Block1-Block1 (night-night)
            block1_mask = np.zeros((n, n), dtype=bool)
            block1_mask[np.ix_(block1_indices, block1_indices)] = True
            block1_mask_nan = block1_mask & ~np.isnan(mat)
            if np.any(block1_mask_nan):
                block1_values = mat[block1_mask_nan].copy()
                rng.shuffle(block1_values)
                mat[block1_mask_nan] = block1_values
            
            # Shuffle entries within Block2-Block2 (day-day)
            block2_mask = np.zeros((n, n), dtype=bool)
            block2_mask[np.ix_(block2_indices, block2_indices)] = True
            block2_mask_nan = block2_mask & ~np.isnan(mat)
            if np.any(block2_mask_nan):
                block2_values = mat[block2_mask_nan].copy()
                rng.shuffle(block2_values)
                mat[block2_mask_nan] = block2_values
            
            # Shuffle entries in Block1-Block2 and Block2-Block1 (night-day interactions)
            # Combine both off-diagonal blocks for shuffling
            cross_mask = np.zeros((n, n), dtype=bool)
            cross_mask[np.ix_(block1_indices, block2_indices)] = True
            cross_mask[np.ix_(block2_indices, block1_indices)] = True
            cross_mask_nan = cross_mask & ~np.isnan(mat)
            if np.any(cross_mask_nan):
                cross_values = mat[cross_mask_nan].copy()
                rng.shuffle(cross_values)
                mat[cross_mask_nan] = cross_values
        else:
            raise ValueError(f"Unknown method: {method}")
        
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
        method_label = method
        if method == "shuffle_blockwise":
            method_label = f"{method} (blocks: {block1_hours} vs {block2_hours})"
        plt.title(f"Band contrast test - {method_label} (p={p_value:.4f})")
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



def cumulative_flux_map_single_day(day, site_id=1, bin_size=20):
    """
    Build a cumulative flux magnitude grid for a single day by summing away and toward directions.
    Returns the flux grid and bin edges.
    
    Parameters
    ----------
    day : pd.Period
        Single day period
    site_id : int
        Site ID
    bin_size : int
        Bin size for spatial binning
    
    Returns
    -------
    grid : 2D numpy array
        Cumulative flux grid (away + toward)
    x_bins : numpy array
        X-axis bin edges
    y_bins : numpy array
        Y-axis bin edges
    """
    # grid setup
    x_bins = np.arange(0, 1920 + bin_size, bin_size)
    y_bins = np.arange(0, 1080 + bin_size, bin_size)
    grid_away = np.zeros((len(y_bins)-1, len(x_bins)-1))
    grid_toward = np.zeros((len(y_bins)-1, len(x_bins)-1))
    
    flow_field_data = get_flow_field_data_for_day(day=day, site_id=site_id)
    
    # loop through 24 hours in the day
    for hour in range(24):
        hourly_data = flow_field_data[hour]   # list of DataFrames for this hour
        if not hourly_data:
            continue
        
        # Compute flux for away direction
        bin_stats_away, _ = plot_flux_field_for_hour(
            hourly_data, hour, site_id=site_id, bin_size=bin_size,
            direction_filter='away', ax=plt.gca(),
            vmin=0, vmax=None, cmap="Reds", show_colorbar=False
        )
        plt.close()
        
        if bin_stats_away is not None and len(bin_stats_away) > 0:
            for _, row in bin_stats_away.iterrows():
                grid_away[int(row["y_bin"]), int(row["x_bin"])] += row["flux_mag"]
        
        # Compute flux for toward direction
        bin_stats_toward, _ = plot_flux_field_for_hour(
            hourly_data, hour, site_id=site_id, bin_size=bin_size,
            direction_filter='toward', ax=plt.gca(),
            vmin=0, vmax=None, cmap="Reds", show_colorbar=False
        )
        plt.close()
        
        if bin_stats_toward is not None and len(bin_stats_toward) > 0:
            for _, row in bin_stats_toward.iterrows():
                grid_toward[int(row["y_bin"]), int(row["x_bin"])] += row["flux_mag"]
    
    # Combine by summing (away + toward)
    grid = grid_away + grid_toward
    
    return grid, x_bins, y_bins


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


def compute_cumulative_flux_similarity_matrix_across_all_days(days_list, site_id=1, bin_size=20, 
                                             metric="cosine", plot=True):
    """
    Compute cosine similarity matrix between cumulative flux maps across days.
    
    For each day, computes cumulative flux map by summing away and toward flux maps
    (similar to lines 3358-3368). Then computes cosine similarity between all pairs of days.
    
    Parameters
    ----------
    days_list : list of pd.Period
        List of day periods to compare
    site_id : int
        Site ID
    bin_size : int
        Bin size for spatial binning
    metric : str
        Similarity metric ('cosine', 'correlation', 'euclidean')
    plot : bool
        Whether to plot the similarity matrix as a heatmap
    
    Returns
    -------
    sim_matrix : numpy array
        num_days x num_days similarity matrix
    """
    num_days = len(days_list)
    print(f"Computing cumulative flux maps for {num_days} days...")
    
    # Compute cumulative flux map for each day
    daily_flux_maps = []
    for i, day in enumerate(days_list):
        print(f"  Processing day {i+1}/{num_days}: {day}")
        grid, x_bins, y_bins = cumulative_flux_map_single_day(day, site_id=site_id, bin_size=bin_size)
        # Flatten grid to vector
        flux_vector = grid.flatten()
        daily_flux_maps.append(flux_vector)
    
    daily_flux_maps = np.array(daily_flux_maps)
    print(f"✅ Computed flux maps: shape {daily_flux_maps.shape}")
    
    # Compute similarity matrix
    sim_matrix = np.zeros((num_days, num_days))
    
    for i in range(num_days):
        for j in range(num_days):
            vec_i = daily_flux_maps[i]
            vec_j = daily_flux_maps[j]
            
            # Skip if either vector is all zeros
            if np.sum(vec_i) == 0 or np.sum(vec_j) == 0:
                sim_matrix[i, j] = np.nan
                continue
            
            if metric == "cosine":
                sim_matrix[i, j] = cosine_similarity([vec_i], [vec_j])[0, 0]
            elif metric == "correlation":
                corr = np.corrcoef(vec_i, vec_j)[0, 1]
                sim_matrix[i, j] = corr if not np.isnan(corr) else 0.0
            elif metric == "euclidean":
                sim_matrix[i, j] = -np.linalg.norm(vec_i - vec_j)  # negative distance for heatmap
            else:
                raise ValueError(f"Unknown metric: {metric}")
    
    # Plot similarity matrix
    if plot:
        plt.figure(figsize=(10, 8))
        sns.heatmap(sim_matrix, cmap="viridis", annot=False, square=True,
                    xticklabels=[f"Day {i+1}" for i in range(num_days)],
                    yticklabels=[f"Day {i+1}" for i in range(num_days)],
                    cbar_kws={'label': f'{metric.title()} similarity'})
        site_name = {1: 'beer', 2: 'shack', 3: 'rain'}.get(site_id, f'site_{site_id}')
        plt.title(f"Cumulative Flux Map Similarity Across Days - {site_name.title()} Tree\n"
                  f"({metric.title()} similarity, {num_days} days)", fontsize=14, fontweight='bold')
        plt.xlabel("Day")
        plt.ylabel("Day")
        plt.tight_layout()
        plt.show()
    
    return sim_matrix


def compute_cumulative_flux_similarity_matrix_shuffled(days_list, site_id=1, bin_size=20, 
                                                       metric="cosine", plot=True, 
                                                       random_state=None):
    """
    Compute similarity matrix between cumulative flux maps with randomly shuffled values.
    
    This creates a null distribution by shuffling the spatial values in each day's flux map
    before computing similarities. Useful for testing if observed similarities are meaningful.
    
    For each day, computes cumulative flux map, then randomly shuffles the spatial values
    (preserving the distribution but destroying spatial structure). Then computes similarity
    between all pairs of shuffled flux maps.
    
    Parameters
    ----------
    days_list : list of pd.Period
        List of day periods to compare
    site_id : int
        Site ID
    bin_size : int
        Bin size for spatial binning
    metric : str
        Similarity metric ('cosine', 'correlation', 'euclidean')
    plot : bool
        Whether to plot the similarity matrix as a heatmap
    random_state : int or None
        Random seed for reproducibility
    
    Returns
    -------
    sim_matrix : numpy array
        num_days x num_days similarity matrix (with shuffled values)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    num_days = len(days_list)
    print(f"Computing cumulative flux maps (with shuffling) for {num_days} days...")
    
    # Compute cumulative flux map for each day
    daily_flux_maps = []
    for i, day in enumerate(days_list):
        print(f"  Processing day {i+1}/{num_days}: {day}")
        grid, x_bins, y_bins = cumulative_flux_map_single_day(day, site_id=site_id, bin_size=bin_size)
        # Flatten grid to vector
        flux_vector = grid.flatten()
        
        # Shuffle the values randomly (preserves distribution, destroys spatial structure)
        shuffled_vector = flux_vector.copy()
        np.random.shuffle(shuffled_vector)
        
        daily_flux_maps.append(shuffled_vector)
    
    daily_flux_maps = np.array(daily_flux_maps)
    print(f"✅ Computed shuffled flux maps: shape {daily_flux_maps.shape}")
    
    # Compute similarity matrix
    sim_matrix = np.zeros((num_days, num_days))
    
    for i in range(num_days):
        for j in range(num_days):
            vec_i = daily_flux_maps[i]
            vec_j = daily_flux_maps[j]
            
            # Skip if either vector is all zeros
            if np.sum(vec_i) == 0 or np.sum(vec_j) == 0:
                sim_matrix[i, j] = np.nan
                continue
            
            if metric == "cosine":
                sim_matrix[i, j] = cosine_similarity([vec_i], [vec_j])[0, 0]
            elif metric == "correlation":
                corr = np.corrcoef(vec_i, vec_j)[0, 1]
                sim_matrix[i, j] = corr if not np.isnan(corr) else 0.0
            elif metric == "euclidean":
                sim_matrix[i, j] = -np.linalg.norm(vec_i - vec_j)  # negative distance for heatmap
            else:
                raise ValueError(f"Unknown metric: {metric}")
    
    # Plot similarity matrix
    if plot:
        plt.figure(figsize=(10, 8))
        sns.heatmap(sim_matrix, cmap="viridis", annot=False, square=True,
                    xticklabels=[f"Day {i+1}" for i in range(num_days)],
                    yticklabels=[f"Day {i+1}" for i in range(num_days)],
                    cbar_kws={'label': f'{metric.title()} similarity (shuffled)'})
        site_name = {1: 'beer', 2: 'shack', 3: 'rain'}.get(site_id, f'site_{site_id}')
        plt.title(f"Cumulative Flux Map Similarity (Shuffled) - {site_name.title()} Tree\n"
                  f"({metric.title()} similarity, {num_days} days, spatial structure destroyed)", 
                  fontsize=14, fontweight='bold')
        plt.xlabel("Day")
        plt.ylabel("Day")
        plt.tight_layout()
        plt.show()
    
    return sim_matrix






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
    Extract trails from nest junctions (starting points) to endpoints (ending points) using different strategies.
    
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

    # Find custom junctions near nest (these are actually starting points for the trails)
    nest_junctions = find_nest_junctions(sk, nest_mask, max_dist, n_points, min_spacing)

    # Extract trails using specified method
    trails = extract_all_trails_from_junctions(G, nest_junctions, endpoints, min_length, method=trail_method, max_depth=max_depth, max_trails=max_trails)

    if plot:
        plt.figure(figsize=(6,6))
        plt.imshow(sk, cmap="gray", origin="upper")
        plt.scatter([p[0] for p in endpoints], [p[1] for p in endpoints], s=8, c='blue', label="Endpoints")
        plt.scatter([p[0] for p in nest_junctions], [p[1] for p in nest_junctions], s=30, c='red', label="Start points")
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

def remove_subset_trails(trails, tol=2, overlap_threshold=1.0):
    """
    Remove trails that are spatial subsets of longer trails.

    Parameters
    ----------
    trails : list of (N_i, 2) arrays
        List of (x, y) coordinates for each trail.
    tol : float
        Distance tolerance (pixels) to consider points as overlapping.
    overlap_threshold : float, default=1.0
        Fraction of points that must overlap (0.0 to 1.0).
        - 1.0 (100%): Remove trail only if ALL points are within tolerance (strict)
        - 0.95 (95%): Remove trail if 95% or more points are within tolerance
        - 0.5 (50%): Remove trail if 50% or more points are within tolerance (lenient)

    Returns
    -------
    filtered : list of (N_i, 2) arrays
        Trails with near-duplicate or overlapping ones removed.
    """

    # Sort by descending length (keep longer trails first)
    trails = sorted(trails, key=lambda t: len(t), reverse=True)
    keep = []

    for i, t1 in enumerate(trails):
        is_subset = False

        for kept in keep:
            tree2 = cKDTree(kept)
            # Compute nearest neighbor distance from t1 -> kept
            dists, _ = tree2.query(t1, k=1)
            # Calculate fraction of points within tolerance
            points_within_tol = np.sum(dists < tol)
            overlap_fraction = points_within_tol / len(t1) if len(t1) > 0 else 0.0
            
            # If overlap fraction meets threshold -> subset
            if overlap_fraction >= overlap_threshold:
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

def plot_trail_ant_counts_timeseries(seg_masks, days_period, num_days, site_id=1, 
                                   direction_filter=None, bin_size=20, 
                                   title_suffix="", figsize=(12, 8)):
    """
    Plot time series of ant counts per trail segment over multiple days.
    Counts are normalized by trail length (number of bins) to enable comparison across trails.
    This gives ant density (ants per frame per bin) rather than absolute counts.
    
    Parameters
    ----------
    seg_masks : list of 2D bool arrays
        Each mask marks pixels belonging to a single trail segment.
    days_period : pd.Period
        Starting day period.
    num_days : int
        Number of days to analyze.
    site_id : int
        Site ID (1=beer, 2=shack, 3=rain).
    direction_filter : str or None
        Direction filter: 'away', 'toward', 'unknown', or None for all directions.
    bin_size : int
        Bin size used for the flux map (should match seg_masks grid).
    title_suffix : str
        Additional text for plot title.
    figsize : tuple
        Figure size (width, height).
    
    Returns
    -------
    tuple
        (trail_counts, trail_counts_raw) where each is a dict:
        {trail_id: {hour: [values_per_day]}}
        - trail_counts contains density values (ants per frame per bin, length-normalized)
        - trail_counts_raw contains raw counts (average ants per frame, NOT normalized by trail length)
    """
    if not seg_masks:
        print("No trail segments provided.")
        return {}, {}
    
    # Validate seg_masks shape matches expected grid size
    # Expected shape based on bin_size: (1080/bin_size, 1920/bin_size)
    expected_shape = (1080 // bin_size, 1920 // bin_size)
    
    # Calculate trail lengths (number of bins/pixels in each trail) for normalization
    trail_lengths = [np.sum(trail_mask) for trail_mask in seg_masks]
    print(f"Trail lengths (number of bins): {trail_lengths}")
    
    # Initialize data structure: trail_id -> hour -> list of counts
    trail_counts = {}  # Density values (normalized by trail length)
    trail_counts_raw = {}  # Raw counts (average ants per frame, NOT normalized)
    for trail_id in range(len(seg_masks)):
        trail_counts[trail_id] = {hour: [] for hour in range(24)}
        trail_counts_raw[trail_id] = {hour: [] for hour in range(24)}
    
    # Get data for each day
    for day_offset in range(num_days):
        current_day = days_period + day_offset
        print(f"Processing day {current_day}")
        
        # Get flow field data for the entire day
        flow_field_data = get_flow_field_data_for_day(current_day, site_id=site_id)
        
        # Process each hour
        for hour in range(24):
            hourly_data = flow_field_data[hour]
            if not hourly_data:
                # No data for this hour, add zeros for all trails
                print(f"  Hour {hour}: No data - adding zeros")
                for trail_id in range(len(seg_masks)):
                    trail_counts[trail_id][hour].append(0)
                    trail_counts_raw[trail_id][hour].append(0)
                continue
            
            # Combine all data for this hour
            all_velocity_data = pd.concat(hourly_data, ignore_index=True)
            
            # Filter by direction if specified
            if direction_filter:
                all_velocity_data = all_velocity_data[all_velocity_data['direction'] == direction_filter]
                if len(all_velocity_data) == 0:
                    # No data for this direction, add zeros for all trails
                    print(f"  Hour {hour}: No data after direction filter '{direction_filter}' - adding zeros")
                    for trail_id in range(len(seg_masks)):
                        trail_counts[trail_id][hour].append(0)
                        trail_counts_raw[trail_id][hour].append(0)
                    continue
            
            # Calculate average ants per frame (not unique ants per hour)
            # We want to count all detections but normalize by number of frames
            num_frames = all_velocity_data['frame_number'].nunique() if 'frame_number' in all_velocity_data.columns else 1
            if num_frames == 0:
                num_frames = 1  # Prevent division by zero
            print(f"  Total detections: {len(all_velocity_data)}, Frames: {num_frames}")
            
            # Convert coordinates to bin indices (same as flux calculation)
            x_bins = np.arange(0, 1920 + bin_size, bin_size)
            y_bins = np.arange(0, 1080 + bin_size, bin_size)
            
            # Bin the ant positions
            all_velocity_data['x_bin'] = pd.cut(all_velocity_data['center_x'], bins=x_bins, labels=False)
            all_velocity_data['y_bin'] = pd.cut(all_velocity_data['center_y'], bins=y_bins, labels=False)
            
            # Create a binned count map for this hour (much more efficient)
            # Initialize count grid
            count_grid = np.zeros((len(y_bins)-1, len(x_bins)-1), dtype=int)
            
            # Count ants in each bin using vectorized operations
            valid_data = all_velocity_data.dropna(subset=['x_bin', 'y_bin'])
            if len(valid_data) > 0:
                # Convert to integer indices
                x_indices = valid_data['x_bin'].astype(int)
                y_indices = valid_data['y_bin'].astype(int)
                
                # Filter out indices that are out of bounds
                valid_mask = ((x_indices >= 0) & (x_indices < count_grid.shape[1]) & 
                             (y_indices >= 0) & (y_indices < count_grid.shape[0]))
                
                if valid_mask.any():
                    x_valid = x_indices[valid_mask]
                    y_valid = y_indices[valid_mask]
                    
                    # Count occurrences in each bin using numpy.bincount
                    # Flatten 2D coordinates to 1D for bincount
                    flat_indices = y_valid * count_grid.shape[1] + x_valid
                    bin_counts = np.bincount(flat_indices, minlength=count_grid.size)
                    count_grid = bin_counts.reshape(count_grid.shape)
                    
                    # DEBUG: Print total counts
                    total_ants_in_grid = np.sum(count_grid)
                    print(f"  Total ants in count grid per frame at hour {hour} for day {day_offset} : {total_ants_in_grid/num_frames}")
                else:
                    print(f"DEBUG: Hour {hour}, Day {day_offset} - No valid detections after binning")
            else:
                print(f"DEBUG: Hour {hour}, Day {day_offset} - No valid data after dropping NaN")
            
            # Count ants in each trail segment using vectorized mask multiplication
            # Normalize by number of frames to get average ants per frame
            # Then normalize by trail length to get density (ants per frame per bin)
            for trail_id, trail_mask in enumerate(seg_masks):
                # Multiply the count grid with the trail mask and sum
                trail_count_total = np.sum(count_grid * trail_mask)
                # Convert to average ants per frame (raw count, not normalized by trail length)
                trail_count_avg = trail_count_total / num_frames
                
                # Normalize by trail length (number of bins in trail) to get density
                trail_length = trail_lengths[trail_id]
                if trail_length > 0:
                    trail_count_density = trail_count_avg / trail_length
                else:
                    trail_count_density = 0  # Avoid division by zero
                
                # Store directly in dictionaries
                trail_counts[trail_id][hour].append(trail_count_density)
                trail_counts_raw[trail_id][hour].append(trail_count_avg)
    
    # Create the plot for COUNTS (density values)
    plt.figure(figsize=figsize)
    
    # Generate colors for each trail
    colors = plt.cm.tab20(np.linspace(0, 1, len(seg_masks)))
    
    # Formatting variables
    direction_text = f" ({direction_filter})" if direction_filter else " (all directions)"
    site_name = {1: 'beer', 2: 'shack', 3: 'rain'}.get(site_id, f'site_{site_id}')
    
    # Plot each trail using counts
    for trail_id in range(len(seg_masks)):
        trail_data = trail_counts[trail_id]
        
        # Calculate bootstrapped mean and 95% confidence interval for each hour across all days
        hours = []
        bootstrapped_means = []
        err_lower = []
        err_upper = []
        
        for hour in range(24):
            counts = trail_data[hour]
            if counts and len(counts) > 0:  # Only if we have data
                hours.append(hour)
                # Bootstrap with 10000 samples
                bootstrapped_data = bootstrap(counts, 10000)
                bootstrapped_mean = np.mean(bootstrapped_data)
                bootstrapped_means.append(bootstrapped_mean)
                
                # Get 95% confidence interval
                conf_min, conf_max = confidence_interval(bootstrapped_data)
                err_lower.append(bootstrapped_mean - conf_min)
                err_upper.append(conf_max - bootstrapped_mean)
            else:
                hours.append(hour)
                bootstrapped_means.append(0)
                err_lower.append(0)
                err_upper.append(0)
        
        # Plot with error bars (asymmetric CI)
        plt.errorbar(hours, bootstrapped_means, yerr=[err_lower, err_upper], 
                   color=colors[trail_id], marker='o', 
                   label=f'Trail {trail_id + 1}', 
                   capsize=3, capthick=1, alpha=0.8)
    
    # Formatting for counts plot
    plt.xlabel('Hour of Day')
    plt.ylabel('Ant Density (Average Ants per Frame per Bin)')
    plt.title(f'Ant Density per Trail Segment (Length-Normalized) - {site_name.title()} Tree{direction_text}{title_suffix}')
    plt.xticks(range(0, 24, 2))
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    plt.close()  # Close the figure to free memory
    
    
    return trail_counts, trail_counts_raw


def plot_trail_density_difference(trail_counts_away, trail_counts_toward, seg_masks, 
                                   site_id=1, title_suffix="", figsize=(12, 8)):
    """
    Plot the Directional Bias Ratio (DBR) = (away - toward) / (away + toward) in ant densities 
    for each trail and hour. This shows the normalized directional bias in ant movement on each trail.
    DBR ranges from -1 (all toward) to +1 (all away), with 0 indicating no bias.
    
    Parameters
    ----------
    trail_counts_away : dict
        Dictionary from plot_trail_ant_counts_timeseries with direction_filter='away'
        Format: {trail_id: {hour: [density_values_per_day]}}
    trail_counts_toward : dict
        Dictionary from plot_trail_ant_counts_timeseries with direction_filter='toward'
        Format: {trail_id: {hour: [density_values_per_day]}}
    seg_masks : list of 2D bool arrays
        Each mask marks pixels belonging to a single trail segment.
    site_id : int
        Site ID (1=beer, 2=shack, 3=rain).
    title_suffix : str
        Additional text for plot title.
    figsize : tuple
        Figure size (width, height).
    
    Returns
    -------
    dict
        {trail_id: {hour: [dbr_values_per_day]}} where DBR = (away - toward) / (away + toward)
    """
    # Compute DBR (away - toward) / (away + toward) for each trail, hour, and day
    trail_dbr = {}
    for trail_id in trail_counts_away.keys():
        trail_dbr[trail_id] = {}
        for hour in range(24):
            away_values = trail_counts_away[trail_id][hour]
            toward_values = trail_counts_toward[trail_id][hour]
            
            # Ensure both lists have the same length (should be num_days)
            min_len = min(len(away_values), len(toward_values))
            if len(away_values) != len(toward_values):
                print(f"Warning: Mismatch in data length for trail {trail_id}, hour {hour}. "
                      f"Away: {len(away_values)}, Toward: {len(toward_values)}")
            
            # Compute DBR: (away - toward) / (away + toward)
            # Handle division by zero (when away + toward = 0, set DBR to 0)
            dbr_values = []
            for i in range(min_len):
                away = away_values[i]
                toward = toward_values[i]
                total = away + toward
                if total > 0:
                    dbr = (away - toward) / total
                else:
                    dbr = 0.0  # No ants in either direction
                dbr_values.append(dbr)
            trail_dbr[trail_id][hour] = dbr_values
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Generate colors for each trail
    colors = plt.cm.tab20(np.linspace(0, 1, len(seg_masks)))
    
    # Plot each trail
    for trail_id in range(len(seg_masks)):
        trail_data = trail_dbr[trail_id]
        
        # Calculate bootstrapped mean and 95% confidence interval for each hour across all days
        hours = []
        bootstrapped_means = []
        err_lower = []
        err_upper = []
        
        for hour in range(24):
            dbr_values = trail_data[hour]
            if dbr_values and len(dbr_values) > 0:  # Only if we have data
                hours.append(hour)
                # Bootstrap with 10000 samples
                bootstrapped_data = bootstrap(dbr_values, 10000)
                bootstrapped_mean = np.mean(bootstrapped_data)
                bootstrapped_means.append(bootstrapped_mean)
                
                # Get 95% confidence interval
                conf_min, conf_max = confidence_interval(bootstrapped_data)
                err_lower.append(bootstrapped_mean - conf_min)
                err_upper.append(conf_max - bootstrapped_mean)
            else:
                hours.append(hour)
                bootstrapped_means.append(0)
                err_lower.append(0)
                err_upper.append(0)
        
        # Plot with error bars (asymmetric CI)
        plt.errorbar(hours, bootstrapped_means, yerr=[err_lower, err_upper], 
                   color=colors[trail_id], marker='o', 
                   label=f'Trail {trail_id + 1}', 
                   capsize=3, capthick=1, alpha=0.8)
    
    # Formatting
    site_name = {1: 'beer', 2: 'shack', 3: 'rain'}.get(site_id, f'site_{site_id}')
    
    plt.xlabel('Hour of Day')
    plt.ylabel('Directional Bias Ratio (DBR)')
    plt.title(f'Trail Directional Bias Ratio (Away - Toward) / (Away + Toward) - {site_name.title()} Tree{title_suffix}')
    plt.xticks(range(0, 24, 2))
    plt.ylim(-1.1, 1.1)  # DBR ranges from -1 to +1
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)  # Zero line
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save plot
    import os
    save_dir = '/home/tarun/Desktop/plots_for_committee_meeting/trail_separation/'
    os.makedirs(save_dir, exist_ok=True)
    
    filename = f"trail_dbr_{site_name}{title_suffix.replace(' ', '_').replace('-', '_')}.png"
    filepath = os.path.join(save_dir, filename)
    plt.show()
    #plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"DBR plot saved to: {filepath}")
    plt.close()  # Close the figure to free memory
    
    return trail_dbr


def plot_trail_usage_across_days(trail_fractions_toward, trail_fractions_away, 
                                days_period, num_days, site_id=1, 
                                figsize=(20, 8)):
    """
    Create two separate plots showing trail fractions across all days.
    One plot for 'toward' direction, one plot for 'away' direction.
    Each plot shows all trails as separate lines.
    
    Parameters
    ----------
    trail_fractions_toward : dict
        Dictionary from plot_trail_ant_counts_timeseries with direction_filter='toward'
        Format: {trail_id: {hour: [fractions_per_day]}}
    trail_fractions_away : dict
        Dictionary from plot_trail_ant_counts_timeseries with direction_filter='away'
        Format: {trail_id: {hour: [fractions_per_day]}}
    days_period : pd.Period
        Starting day period
    num_days : int
        Number of days analyzed
    site_id : int
        Site ID for naming
    figsize : tuple
        Figure size (width, height)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Validate inputs
    if not trail_fractions_toward or not trail_fractions_away:
        raise ValueError("trail_fractions_toward and trail_fractions_away must not be empty")
    
    # Get all trail IDs and validate they match
    trail_ids_toward = set(trail_fractions_toward.keys())
    trail_ids_away = set(trail_fractions_away.keys())
    
    if trail_ids_toward != trail_ids_away:
        missing_in_away = trail_ids_toward - trail_ids_away
        missing_in_toward = trail_ids_away - trail_ids_toward
        raise ValueError(
            f"Trail IDs don't match between toward and away data. "
            f"Missing in away: {missing_in_away}, Missing in toward: {missing_in_toward}"
        )
    
    trail_ids = sorted(list(trail_ids_toward))  # Sort for consistent ordering
    n_trails = len(trail_ids)
    
    # Validate that all hours (0-23) exist for at least one trail
    sample_trail_id = trail_ids[0]
    expected_hours = set(range(24))
    available_hours_toward = set(trail_fractions_toward[sample_trail_id].keys())
    available_hours_away = set(trail_fractions_away[sample_trail_id].keys())
    
    if available_hours_toward != expected_hours:
        missing_hours = expected_hours - available_hours_toward
        print(f"Warning: Missing hours in toward data: {missing_hours}")
    if available_hours_away != expected_hours:
        missing_hours = expected_hours - available_hours_away
        print(f"Warning: Missing hours in away data: {missing_hours}")
    
    # Create flattened time points (hours x days)
    time_points = []
    time_labels = []
    for day_idx in range(num_days):
        for hour in range(24):
            time_points.append(day_idx * 24 + hour)
            if hour % 6 == 0:  # Label every 6 hours
                time_labels.append(f'D{day_idx+1}H{hour:02d}')
            else:
                time_labels.append('')
    
    # Generate colors for each trail
    colors = plt.cm.tab20(np.linspace(0, 1, n_trails))
    
    # Create two separate plots
    site_name = {1: 'beer', 2: 'shack', 3: 'rain'}.get(site_id, f'site_{site_id}')
    
    # Plot 1: TOWARD direction
    plt.figure(figsize=figsize)
    
    for i, trail_id in enumerate(trail_ids):
        # Get data for this trail
        toward_data = trail_fractions_toward[trail_id]
        
        # Flatten the data for plotting
        toward_values = []
        
        for day_idx in range(num_days):
            for hour in range(24):
                # Get the fraction for this day (day_idx) and hour
                # Handle missing hours or insufficient data gracefully
                if hour not in toward_data:
                    toward_values.append(0)
                elif day_idx < len(toward_data[hour]):
                    toward_values.append(toward_data[hour][day_idx])
                else:
                    toward_values.append(0)
        
        # Plot this trail
        plt.plot(time_points, toward_values, color=colors[i], linewidth=2, alpha=0.8, 
                label=f'Trail {trail_id + 1}', marker='o', markersize=3)
    
    # Formatting for TOWARD plot
    plt.title(f'Trail Usage - Toward Nest - {site_name.title()} Tree\n'
              f'({num_days} days, {24*num_days} time points)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Time (Days × Hours)')
    plt.ylabel('Fraction of Total Ants per Trail')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set x-axis labels for every 6 hours
    plt.xticks(range(0, len(time_points), 6), 
               [time_labels[j] for j in range(0, len(time_points), 6)], 
               rotation=45, ha='right')
    
    # Add vertical lines to separate days
    for day_idx in range(1, num_days):
        plt.axvline(x=day_idx * 24 - 0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save TOWARD plot
    import os
    save_dir = '/home/tarun/Desktop/plots_for_committee_meeting/trail_separation/'
    os.makedirs(save_dir, exist_ok=True)
    
    filename_toward = f"trail_usage_toward_{site_name}_{num_days}days.png"
    filepath_toward = os.path.join(save_dir, filename_toward)
    plt.show()
    #plt.savefig(filepath_toward, dpi=300, bbox_inches='tight')
    print(f"Trail usage TOWARD plot saved to: {filepath_toward}")
    plt.close()
    
    # Plot 2: AWAY direction
    plt.figure(figsize=figsize)
    
    for i, trail_id in enumerate(trail_ids):
        # Get data for this trail
        away_data = trail_fractions_away[trail_id]
        
        # Flatten the data for plotting
        away_values = []
        
        for day_idx in range(num_days):
            for hour in range(24):
                # Get the fraction for this day (day_idx) and hour
                # Handle missing hours or insufficient data gracefully
                if hour not in away_data:
                    away_values.append(0)
                elif day_idx < len(away_data[hour]):
                    away_values.append(away_data[hour][day_idx])
                else:
                    away_values.append(0)
        
        # Plot this trail
        plt.plot(time_points, away_values, color=colors[i], linewidth=2, alpha=0.8, 
                label=f'Trail {trail_id + 1}', marker='s', markersize=3)
    
    # Formatting for AWAY plot
    plt.title(f'Trail Usage - Away from Nest - {site_name.title()} Tree\n'
              f'({num_days} days, {24*num_days} time points)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Time (Days × Hours)')
    plt.ylabel('Fraction of Total Ants per Trail')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set x-axis labels for every 6 hours
    plt.xticks(range(0, len(time_points), 6), 
               [time_labels[j] for j in range(0, len(time_points), 6)], 
               rotation=45, ha='right')
    
    # Add vertical lines to separate days
    for day_idx in range(1, num_days):
        plt.axvline(x=day_idx * 24 - 0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save AWAY plot
    filename_away = f"trail_usage_away_{site_name}_{num_days}days.png"
    filepath_away = os.path.join(save_dir, filename_away)
    plt.show()
    #plt.savefig(filepath_away, dpi=300, bbox_inches='tight')
    print(f"Trail usage AWAY plot saved to: {filepath_away}")
    plt.close()
    
    return None


def plot_daily_average_trail_usage(trail_fractions_toward, trail_fractions_away, 
                                  days_period, num_days, site_id=1, 
                                  figsize=(15, 8)):
    """
    Create plots showing daily average trail usage (averaged over 24 hours per day).
    One plot for 'toward' direction, one plot for 'away' direction.
    Each plot shows all trails as separate lines.
    
    Parameters
    ----------
    trail_fractions_toward : dict
        Dictionary from plot_trail_ant_counts_timeseries with direction_filter='toward'
        Format: {trail_id: {hour: [fractions_per_day]}}
    trail_fractions_away : dict
        Dictionary from plot_trail_ant_counts_timeseries with direction_filter='away'
        Format: {trail_id: {hour: [fractions_per_day]}}
    days_period : pd.Period
        Starting day period
    num_days : int
        Number of days analyzed
    site_id : int
        Site ID for naming
    figsize : tuple
        Figure size (width, height)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Get all trail IDs
    trail_ids = list(trail_fractions_toward.keys())
    n_trails = len(trail_ids)
    
    # Create day labels
    day_labels = []
    for day_idx in range(num_days):
        current_day = days_period + day_idx
        day_labels.append(f'Day {day_idx + 1}\n({current_day})')
    
    # Generate colors for each trail
    colors = plt.cm.tab20(np.linspace(0, 1, n_trails))
    
    # Create two separate plots
    site_name = {1: 'beer', 2: 'shack', 3: 'rain'}.get(site_id, f'site_{site_id}')
    
    # Plot 1: TOWARD direction - Daily averages
    plt.figure(figsize=figsize)
    
    for i, trail_id in enumerate(trail_ids):
        # Get data for this trail
        toward_data = trail_fractions_toward[trail_id]
        
        # Calculate daily averages
        daily_averages = []
        
        for day_idx in range(num_days):
            # Average over all 24 hours for this day
            day_fractions = []
            for hour in range(24):
                if day_idx < len(toward_data[hour]):
                    day_fractions.append(toward_data[hour][day_idx])
                else:
                    day_fractions.append(0)
            
            # Calculate mean for this day
            daily_avg = np.mean(day_fractions)
            daily_averages.append(daily_avg)
        
        # Plot this trail
        plt.plot(range(num_days), daily_averages, color=colors[i], linewidth=3, alpha=0.8, 
                label=f'Trail {trail_id + 1}', marker='o', markersize=6)
    
    # Formatting for TOWARD plot
    plt.title(f'Daily Average Trail Usage - Toward Nest - {site_name.title()} Tree\n'
              f'(Averaged over 24 hours per day)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Day')
    plt.ylabel('Average Fraction of Total Ants per Trail')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set x-axis labels
    plt.xticks(range(num_days), day_labels, rotation=45, ha='right')
    plt.ylim(0, 1)  # Fraction range
    
    plt.tight_layout()
    
    # Save TOWARD plot
    import os
    save_dir = '/home/tarun/Desktop/plots_for_committee_meeting/trail_separation/'
    os.makedirs(save_dir, exist_ok=True)
    
    filename_toward = f"daily_average_trail_usage_toward_{site_name}_{num_days}days.png"
    filepath_toward = os.path.join(save_dir, filename_toward)
    
    plt.savefig(filepath_toward, dpi=300, bbox_inches='tight')
    print(f"Daily average TOWARD plot saved to: {filepath_toward}")
    plt.close()
    
    # Plot 2: AWAY direction - Daily averages
    plt.figure(figsize=figsize)
    
    for i, trail_id in enumerate(trail_ids):
        # Get data for this trail
        away_data = trail_fractions_away[trail_id]
        
        # Calculate daily averages
        daily_averages = []
        
        for day_idx in range(num_days):
            # Average over all 24 hours for this day
            day_fractions = []
            for hour in range(24):
                if day_idx < len(away_data[hour]):
                    day_fractions.append(away_data[hour][day_idx])
                else:
                    day_fractions.append(0)
            
            # Calculate mean for this day
            daily_avg = np.mean(day_fractions)
            daily_averages.append(daily_avg)
        
        # Plot this trail
        plt.plot(range(num_days), daily_averages, color=colors[i], linewidth=3, alpha=0.8, 
                label=f'Trail {trail_id + 1}', marker='s', markersize=6)
    
    # Formatting for AWAY plot
    plt.title(f'Daily Average Trail Usage - Away from Nest - {site_name.title()} Tree\n'
              f'(Averaged over 24 hours per day)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Day')
    plt.ylabel('Average Fraction of Total Ants per Trail')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set x-axis labels
    plt.xticks(range(num_days), day_labels, rotation=45, ha='right')
    plt.ylim(0, 1)  # Fraction range
    
    plt.tight_layout()
    
    # Save AWAY plot
    filename_away = f"daily_average_trail_usage_away_{site_name}_{num_days}days.png"
    filepath_away = os.path.join(save_dir, filename_away)
    
    plt.savefig(filepath_away, dpi=300, bbox_inches='tight')
    print(f"Daily average AWAY plot saved to: {filepath_away}")
    plt.close()
    
    return None


def test_trail_bidirectionality(trail_counts_away, trail_counts_toward, 
                                num_days, alpha=0.05):
    """
    Test bidirectionality for each trail and hour by computing density difference
    (away density - toward density) and using bootstrapped mean with 95% confidence intervals.
    
    Density difference = A - T, where:
    - A = density of ants moving 'away' on that trail (ants per frame per bin, normalized by trail length)
    - T = density of ants moving 'toward' on that trail (ants per frame per bin, normalized by trail length)
    
    For each trail and hour:
    1. Calculate density difference (A - T) for each day at that hour
    2. Bootstrap the difference values (10000 samples) to get mean and 95% CI
    3. Check if 95% CI excludes 0 (significantly different from zero)
    
    Parameters
    ----------
    trail_counts_away : dict
        Dictionary from plot_trail_ant_counts_timeseries with direction_filter='away'
        Format: {trail_id: {hour: [density_values_per_day]}}
        Note: Uses DENSITY values (trail_counts_away), normalized by trail length.
        Density = average ants per frame per bin (length-normalized).
    trail_counts_toward : dict
        Dictionary from plot_trail_ant_counts_timeseries with direction_filter='toward'
        Format: {trail_id: {hour: [density_values_per_day]}}
        Note: Uses DENSITY values (trail_counts_toward), normalized by trail length.
        Density = average ants per frame per bin (length-normalized).
    num_days : int
        Number of days analyzed
    alpha : float, default=0.05
        Significance level (not used directly, but kept for compatibility)
    
    Returns
    -------
    dict
        Results for each trail and hour:
        {
          trail_id: {
            hour: {
              'dbr_values': list of density difference values (A - T) for this hour across days,
              'bootstrapped_mean': float,
              'ci_lower': float (2.5th percentile),
              'ci_upper': float (97.5th percentile),
              'is_significant': bool (True if CI excludes 0),
              'direction': str ('away' or 'toward' or 'balanced'),
              'n_samples': int
            }
          }
        }
    """
    from scipy import stats
    import numpy as np
    
    # Validate inputs
    if not trail_counts_away or not trail_counts_toward:
        raise ValueError("trail_counts_away and trail_counts_toward must not be empty")
    
    # Get all trail IDs
    trail_ids = sorted(list(trail_counts_away.keys()))
    results = {}
    significant_trail_hours = []  # List of (trail_id, hour) tuples
    
    print("\n" + "="*80)
    print("TRAIL BIDIRECTIONALITY TEST (Density Difference Analysis) - Per Hour")
    print("="*80)
    print(f"Density Difference = (Away Density - Toward Density)")
    print(f"Testing H0: Difference = 0 (no directional bias) vs H1: Difference ≠ 0")
    print(f"Using bootstrapped mean and 95% confidence interval")
    print(f"Significance: 95% CI does not include 0")
    print("="*80)
    
    for trail_id in trail_ids:
        away_data = trail_counts_away[trail_id]
        toward_data = trail_counts_toward[trail_id]
        
        trail_results = {}
        
        # Test each hour separately
        for hour in range(24):
            if hour not in away_data or hour not in toward_data:
                continue
            
            # Collect density difference values for this hour across all days
            density_diff_values = []
            
            for day_idx in range(num_days):
                if day_idx < len(away_data[hour]) and day_idx < len(toward_data[hour]):
                    A = away_data[hour][day_idx]  # Away density
                    T = toward_data[hour][day_idx]  # Toward density
                    
                    # Calculate density difference: (Away Density - Toward Density)
                    density_diff_values.append(A - T)
            
            if len(density_diff_values) == 0:
                continue
            
            # Bootstrap with 10000 samples
            bootstrapped_data = bootstrap(density_diff_values, 10000)
            bootstrapped_mean = np.mean(bootstrapped_data)
            
            # Get 95% confidence interval
            conf_min, conf_max = confidence_interval(bootstrapped_data)
            
            # Check if significantly different from zero (CI does not include 0)
            is_significant = (conf_min > 0) or (conf_max < 0)
            
            # Determine direction of bias
            if is_significant:
                if bootstrapped_mean > 0:
                    direction = 'away'
                else:
                    direction = 'toward'
            else:
                direction = 'balanced'
            
            # Store results for this hour
            trail_results[hour] = {
                'dbr_values': density_diff_values,  # Kept as 'dbr_values' for backward compatibility
                'bootstrapped_mean': bootstrapped_mean,
                'ci_lower': conf_min,
                'ci_upper': conf_max,
                'is_significant': is_significant,
                'direction': direction,
                'n_samples': len(density_diff_values)
            }
            
            # Track significant trail-hour combinations
            if is_significant:
                significant_trail_hours.append((trail_id, hour))
        
        results[trail_id] = trail_results
    
    # Print significant trail-hour combinations
    print("\n" + "="*80)
    print("SIGNIFICANT TRAIL-HOUR COMBINATIONS (Density difference significantly different from 0):")
    print("="*80)
    if len(significant_trail_hours) == 0:
        print("No trail-hour combinations with significant density difference (95% CI excludes 0)")
    else:
        print(f"{'Trail':<8} {'Hour':<8} {'Mean Diff':<12} {'95% CI':<25} {'Direction':<12}")
        print("-"*80)
        for trail_id, hour in sorted(significant_trail_hours):
            r = results[trail_id][hour]
            ci_str = f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]"
            print(f"Trail {trail_id+1:<6} {hour:<8} {r['bootstrapped_mean']:>10.4f}  {ci_str:<25} {r['direction']:>11}")
    print("="*80 + "\n")
    
    return results


def plot_dbr_timeseries_per_trail(dbr_test_results, site_id=1, 
                                   output_dir=None, figsize=(16, 10)):
    """
    Plot DBR (Directional Bias Ratio) aggregated over days for each trail and hour.
    
    Creates a subplot for each trail showing bootstrapped mean DBR with 95% confidence
    intervals across 24 hours. Uses pre-computed results from test_trail_bidirectionality().
    
    Parameters
    ----------
    dbr_test_results : dict
        Results from test_trail_bidirectionality() function.
        Format: {trail_id: {hour: {'bootstrapped_mean': float, 'ci_lower': float, 
                                   'ci_upper': float, ...}}}
    site_id : int, default=1
        Site ID for naming output file
    output_dir : str, optional
        Directory to save the plot. If None, uses default plots directory.
    figsize : tuple, default=(16, 10)
        Figure size (width, height)
    
    Returns
    -------
    None
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    # Validate inputs
    if not dbr_test_results:
        raise ValueError("dbr_test_results must not be empty")
    
    trail_ids = sorted(list(dbr_test_results.keys()))
    num_trails = len(trail_ids)
    
    # Create subplots - arrange in a grid
    n_cols = min(3, num_trails)
    n_rows = (num_trails + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=False)
    if num_trails == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # X-axis: hours 0-23
    hours = np.arange(24)
    
    for idx, trail_id in enumerate(trail_ids):
        ax = axes[idx]
        # Clear any existing plots on this axis (safety measure)
        ax.clear()
        
        trail_results = dbr_test_results[trail_id]  # {hour: {results}}
        
        # Extract bootstrapped means and confidence intervals for each hour
        bootstrapped_means = []
        err_lower = []
        err_upper = []
        plot_hours = []
        
        for hour in range(24):
            if hour in trail_results:
                hour_data = trail_results[hour]
                bootstrapped_mean = hour_data['bootstrapped_mean']
                ci_lower = hour_data['ci_lower']
                ci_upper = hour_data['ci_upper']
                
                plot_hours.append(hour)
                bootstrapped_means.append(bootstrapped_mean)
                err_lower.append(max(0, bootstrapped_mean - ci_lower))
                err_upper.append(max(0, ci_upper - bootstrapped_mean))

        
        # Plot with error bars (asymmetric CI)
        if len(plot_hours) > 0:
            ax.errorbar(plot_hours, bootstrapped_means, yerr=[err_lower, err_upper],
                       marker='o', markersize=5, capsize=3, capthick=1.5,
                       linewidth=1.5, alpha=0.8, color='steelblue', label='DBR')
        
        # Add horizontal line at DBR = 0 (no bias)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Hour of Day', fontsize=10)
        ax.set_ylabel('Away - Toward densities', fontsize=10)
        ax.set_title(f'Trail {trail_id + 1}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, 23.5)
        #ax.set_ylim(-0.1, 0.1)  # DBR ranges from -1 to +1
        
        # Set x-axis ticks: show every hour
        ax.set_xticks(range(0, 24, 1))
        ax.set_xticklabels(range(0, 24, 1), fontsize=9)
    
    # Hide unused subplots
    for idx in range(num_trails, len(axes)):
        axes[idx].set_visible(False)
    
    # Add overall title
    site_name = {1: 'beer', 2: 'shack', 3: 'rain'}.get(site_id, f'site_{site_id}')
    fig.suptitle(f'Directional density difference per Trail (Aggregated over Days) - {site_name.title()} Tree\n'
                 f'(Density away - Density toward) | Bootstrapped Mean ± 95% CI', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def lagged_correlation_analysis(trail_fractions_toward, trail_fractions_away, 
                               max_lag_hours=12, min_correlation_samples=5,
                               plot_results=True, figsize=(15, 10)):
    """
    Analyze lagged correlations between ants moving toward vs away from nest.
    
    Parameters
    ----------
    trail_fractions_toward : dict
        Dictionary from plot_trail_ant_counts_timeseries with direction_filter='toward'
        Format: {trail_id: {hour: [fractions_per_day]}} - fractions of total ants
    trail_fractions_away : dict
        Dictionary from plot_trail_ant_counts_timeseries with direction_filter='away'
        Format: {trail_id: {hour: [fractions_per_day]}} - fractions of total ants
    max_lag_hours : int
        Maximum lag to test (in hours)
    min_correlation_samples : int
        Minimum number of data points required for correlation calculation
    plot_results : bool
        Whether to create visualization plots
    figsize : tuple
        Figure size for plots
    
    Returns
    -------
    dict
        Results containing correlations, lags, and statistics for each trail
    """
    import scipy.stats as stats
    from scipy.stats import pearsonr
    
    results = {}
    
    # Get all trail IDs (should be the same for both directions)
    trail_ids = list(trail_fractions_toward.keys())
    
    if plot_results:
        # Create subplots for each trail
        n_trails = len(trail_ids)
        n_cols = min(3, n_trails)
        n_rows = (n_trails + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_trails == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
    
    
    for i, trail_id in enumerate(trail_ids):
        print(f"Analyzing trail {trail_id + 1}...")
        
        
        # Get data for this trail
        toward_data = trail_fractions_toward[trail_id]
        away_data = trail_fractions_away[trail_id]
        
        
        # Create time series by flattening data day by day, then hour by hour within each day
        # This ensures lag of 1 means comparing across hours, not across days
        toward_series = []
        away_series = []
        
        # Get the number of days (assuming all hours have the same number of days)
        num_days = len(toward_data[0])  # Number of days for hour 0
        
        for day_idx in range(num_days):
            for hour in range(24):
                toward_series.append(toward_data[hour][day_idx])
                away_series.append(away_data[hour][day_idx])
        
        toward_series = np.array(toward_series)
        away_series = np.array(away_series)
        
        # Calculate correlations for different lags
        correlations = []
        lags = []
        p_values = []
        
        for lag in range(max_lag_hours + 1):
            if lag == 0:
                # No lag: toward(t) vs away(t)
                x = toward_series
                y = away_series
            else:
                # Lag: toward(t) vs away(t+lag)
                if len(toward_series) > lag:
                    x = toward_series[:-lag]
                    y = away_series[lag:]
                else:
                    continue
            
            # Only calculate correlation if we have enough data points
            if len(x) >= min_correlation_samples and len(y) >= min_correlation_samples:
                # Remove any NaN values
                mask = ~(np.isnan(x) | np.isnan(y))
                if np.sum(mask) >= min_correlation_samples:
                    x_clean = x[mask]
                    y_clean = y[mask]
                    
                    if len(x_clean) > 1 and np.std(x_clean) > 0 and np.std(y_clean) > 0:
                        corr, p_val = pearsonr(x_clean, y_clean)
                        correlations.append(corr)
                        lags.append(lag)
                        p_values.append(p_val)
                    else:
                        correlations.append(np.nan)
                        lags.append(lag)
                        p_values.append(np.nan)
                else:
                    correlations.append(np.nan)
                    lags.append(lag)
                    p_values.append(np.nan)
            else:
                correlations.append(np.nan)
                lags.append(lag)
                p_values.append(np.nan)
        
        # Find best lag (highest absolute correlation)
        valid_corrs = np.array(correlations)
        valid_mask = ~np.isnan(valid_corrs)
        
        if np.any(valid_mask):
            # Get indices of valid correlations
            valid_indices = np.where(valid_mask)[0]
            valid_corr_values = valid_corrs[valid_mask]
            
            # Find the index of the best correlation among valid ones
            best_valid_idx = np.argmax(np.abs(valid_corr_values))
            best_original_idx = valid_indices[best_valid_idx]
            
            best_lag = lags[best_original_idx]
            best_corr = valid_corrs[best_original_idx]
            best_p_val = p_values[best_original_idx]
        else:
            best_lag = 0
            best_corr = np.nan
            best_p_val = np.nan
        
        # Store results
        results[trail_id] = {
            'correlations': correlations,
            'lags': lags,
            'p_values': p_values,
            'best_lag': best_lag,
            'best_correlation': best_corr,
            'best_p_value': best_p_val,
            'toward_series': toward_series,
            'away_series': away_series
        }
        
        # Plot for this trail
        if plot_results and i < len(axes):
            ax = axes[i]
            
            # Plot correlation vs lag
            valid_lags = [l for l, c in zip(lags, correlations) if not np.isnan(c)]
            valid_corrs = [c for c in correlations if not np.isnan(c)]
            
            ax.plot(valid_lags, valid_corrs, 'o-', linewidth=2, markersize=6)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.axvline(x=best_lag, color='red', linestyle='--', alpha=0.7, 
                      label=f'Best lag: {best_lag}h')
            
            # Add significance markers
            for lag, corr, p_val in zip(valid_lags, valid_corrs, 
                                       [p for p in p_values if not np.isnan(p)]):
                if p_val < 0.05:
                    ax.scatter(lag, corr, color='red', s=100, marker='*', 
                             zorder=5, label='p < 0.05' if lag == valid_lags[0] else "")
            
            ax.set_xlabel('Lag (hours)')
            ax.set_ylabel('Correlation')
            ax.set_title(f'Trail {trail_id + 1}\nBest: r={best_corr:.3f}, lag={best_lag}h, p={best_p_val:.3f}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            print(f"Trail {trail_id + 1}: Best correlation r={best_corr:.3f} at lag {best_lag}h (p={best_p_val:.3f})")
    
    # Hide unused subplots
    if plot_results:
        for i in range(len(trail_ids), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Lagged Correlation Analysis: Toward(t) vs Away(t+lag)', fontsize=16)
        plt.tight_layout()
        
        # Save the plot
        import os
        save_dir = '/home/tarun/Desktop/plots_for_committee_meeting/trail_separation/'
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, 'lagged_correlation_analysis.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Lagged correlation plot saved to: {filepath}")
        plt.close()
    
    return results


def plot_lagged_correlation_summary(results, figsize=(12, 8)):
    """
    Create a summary plot of lagged correlation results.
    
    Parameters
    ----------
    results : dict
        Results from lagged_correlation_analysis()
    figsize : tuple
        Figure size
    """
    trail_ids = list(results.keys())
    
    # Extract summary statistics
    best_correlations = [results[tid]['best_correlation'] for tid in trail_ids]
    best_lags = [results[tid]['best_lag'] for tid in trail_ids]
    best_p_values = [results[tid]['best_p_value'] for tid in trail_ids]
    
    # Create summary plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Best correlations
    colors = ['red' if p < 0.05 else 'blue' for p in best_p_values]
    bars = ax1.bar(range(len(trail_ids)), best_correlations, color=colors, alpha=0.7)
    ax1.set_xlabel('Trail ID')
    ax1.set_ylabel('Best Correlation')
    ax1.set_title('Best Correlations by Trail')
    ax1.set_xticks(range(len(trail_ids)))
    ax1.set_xticklabels([f'T{i+1}' for i in trail_ids])
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add significance annotation
    ax1.text(0.02, 0.98, 'Red: p < 0.05\nBlue: p ≥ 0.05', 
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Best lags
    ax2.bar(range(len(trail_ids)), best_lags, color='green', alpha=0.7)
    ax2.set_xlabel('Trail ID')
    ax2.set_ylabel('Best Lag (hours)')
    ax2.set_title('Best Lags by Trail')
    ax2.set_xticks(range(len(trail_ids)))
    ax2.set_xticklabels([f'T{i+1}' for i in trail_ids])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the summary plot
    import os
    save_dir = '/home/tarun/Desktop/plots_for_committee_meeting/trail_separation/'
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, 'lagged_correlation_summary.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Lagged correlation summary saved to: {filepath}")
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("LAGGED CORRELATION ANALYSIS SUMMARY")
    print("="*60)
    print(f"{'Trail':<8} {'Best r':<8} {'Best Lag':<10} {'p-value':<10} {'Significant':<12}")
    print("-"*60)
    
    for i, tid in enumerate(trail_ids):
        r = best_correlations[i]
        lag = best_lags[i]
        p = best_p_values[i]
        sig = "Yes" if p < 0.05 else "No"
        print(f"T{tid+1:<7} {r:<8.3f} {lag:<10} {p:<10.3f} {sig:<12}")
    
    print("-"*60)
    print(f"Mean best correlation: {np.nanmean(best_correlations):.3f}")
    print(f"Mean best lag: {np.nanmean(best_lags):.1f} hours")
    print(f"Significant correlations: {sum(1 for p in best_p_values if p < 0.05)}/{len(best_p_values)}")
    print("="*60)




parameters = {'axes.labelsize':8,'axes.titlesize':8, 'xtick.labelsize':8, 'font.family':"sans-serif", 'font.sans-serif':['Arial'], 'font.size':8, 'svg.fonttype':'none'}
plt.rcParams.update(parameters)

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

#days_period = pd.Period('2024-10-03', freq='D')
#num_days = 17
#nest_mask = cv2.imread('/home/tarun/Desktop/masks/rain-tree-10-03-2024_to_10-19-2024_center_only.png',0)
# preprocess_all_velocity_data(days_period, num_days, site)

# days_period = pd.Period('2024-11-15', freq='D')
# num_days = 21
# nest_mask = cv2.imread('/home/tarun/Desktop/masks/rain-tree-11-15-2024_to_12-06-2024_center_only.png',0)
# preprocess_all_velocity_data(days_period, num_days, site)

# # ##### shack 
site = 2
# days_period = pd.Period('2024-08-01', freq='D')
# num_days = 26
# nest_mask = cv2.imread('/home/tarun/Desktop/masks/shack-tree-diffuser-08-01-2024_to_08-26-2024_center_only.png',0)

days_period = pd.Period('2024-08-26', freq='D')
num_days = 24
nest_mask = cv2.imread('/home/tarun/Desktop/masks/shack-tree-diffuser-08-26-2024_to_09-18-2024_center_only.png',0)

#preprocess_all_velocity_data(days_period, num_days, site)


############# analysis 1: linear regression and scatter plot #############

linear_regression()
import sys;sys.exit(0)
#scatter_plot_velocity_for_day_range(days_period, num_days, site)
#scatter_plot_velocity_by_direction_aggregated(days_period, num_days, site)




############# analysis 2: flux maps and similarity matrices between days/hours #############

### Plot flow/flux fields for a given hour on a given day. Compare 'away' vs 'toward' flux maps for a given hour.
# for n in range(1,2):
#     for hour in range(18,21):
#         flow_data = get_flow_field_data_for_hour(days_period+n, hour, site_id=site)
#         #plot_flow_field_for_hour(flow_data, hour, site_id=site, bin_size=20, direction_filter='away')  # Away direction only
#         #plot_flow_field_for_hour(flow_data, hour, site_id=site, bin_size=20, direction_filter='toward')  # Toward direction only
#         #plot_flow_field_for_hour(flow_data, hour, site_id=site, bin_size=20, direction_filter='unknown')  # Toward direction only
#         #plot_flow_field_for_hour(flow_data, hour, site_id=1, bin_size=20, direction_filter=None)  # All directions
#         #plt.tight_layout()
#         #plt.show()
        
#         #plot_flow_field_comparison(flow_data, hour, site_id=site, bin_size=20)  # Side-by-side comparison
#         plot_flux_field_comparison(flow_data, hour, site_id=site, bin_size=20)







# ### band contrast test for same day away vs toward flux similarity matrix for each day to test for global hourly bidirectionality ###
# for d in range(num_days):
#     dayA_data = get_flow_field_data_for_day(day=days_period + d, site_id=site)
#     dayB_data = get_flow_field_data_for_day(day=days_period + d, site_id=site)
#     sim_matrix = plot_flux_similarity_matrix(dayA_data, dayA_data, site_id=site, 
#                                             bin_size=20, direction_filter="same_day_away_vs_toward", 
#                                             metric="cosine")
#     results = band_strength_test(sim_matrix, n_permutations=1000,
#                                 random_state=42,
#                                 method="shuffle_blockwise",
#                                 block1_hours=[7,8,9,10,11,12,13,14,15,16,17,18,19],
#                                 block2_hours=[0,1,2,3,4,5,6,20,21,22,23],
#                                 plot_hist=False)
#     print('band contrast method p value for day ', d, ' is ', results["p_value"])




### Plot daily grids (6x4) of flux maps per hour for a given day and given direction 
# for day in range(1,2):
#     plot_daily_flux_grid(days_period+day, site_id=site, direction_filter="toward", bin_size=20, normalize='shared')
#     #plot_daily_flux_grid(days_period+2, site_id=site, direction_filter="toward", bin_size=20, normalize='per_hour')









############### analysis 3: cumulative flux map and trail separation #############
days_list = [days_period + i for i in range(num_days)]


sim_matrix = compute_cumulative_flux_similarity_matrix_across_all_days(
    days_list, 
    site_id=site, 
    bin_size=20, 
    metric="cosine", 
    plot=True
)
sim_matrix_shuffled = compute_cumulative_flux_similarity_matrix_shuffled(
    days_list, site_id=site, bin_size=20, metric="cosine", plot=True, random_state=42
)

# Print statistics for both matrices
print("\n" + "="*60)
print("SIMILARITY MATRIX STATISTICS")
print("="*60)

# Original matrix statistics (excluding diagonal and NaN values)
mask_original = ~np.isnan(sim_matrix)
# Exclude diagonal (self-similarity, always 1.0)
for i in range(len(sim_matrix)):
    mask_original[i, i] = False
values_original = sim_matrix[mask_original]

print(f"\nOriginal Matrix (excluding diagonal):")
print(f"  Minimum:  {np.min(values_original):.4f}")
print(f"  Median:   {np.median(values_original):.4f}")
print(f"  Maximum:  {np.max(values_original):.4f}")
print(f"  Mean:     {np.mean(values_original):.4f}")

# Shuffled matrix statistics (excluding diagonal and NaN values)
mask_shuffled = ~np.isnan(sim_matrix_shuffled)
# Exclude diagonal
for i in range(len(sim_matrix_shuffled)):
    mask_shuffled[i, i] = False
values_shuffled = sim_matrix_shuffled[mask_shuffled]

print(f"\nShuffled Matrix (excluding diagonal):")
print(f"  Minimum:  {np.min(values_shuffled):.4f}")
print(f"  Median:   {np.median(values_shuffled):.4f}")
print(f"  Maximum:  {np.max(values_shuffled):.4f}")
print(f"  Mean:     {np.mean(values_shuffled):.4f}")

print("="*60 + "\n")

#import sys;sys.exit(0)

# Compute cumulative flux maps separately for away and toward directions
# This avoids cancellation from opposite directions when combining
# Each direction's flux is computed independently, preserving intensity from both
grid_away, x_bins, y_bins = cumulative_flux_map_all_days(days_list, site_id=site, bin_size=20, direction_filter="away")
grid_toward, _, _ = cumulative_flux_map_all_days(days_list, site_id=site, bin_size=20, direction_filter="toward")
# Combine by summing (preserves total intensity from both directions)
# Alternative: use (grid_away + grid_toward) / 2 for averaging instead
grid = grid_away + grid_toward

plt.imshow(grid, cmap="hot", origin="upper",
           extent=[x_bins[0], x_bins[-1], y_bins[-1], y_bins[0]])  # flip y for correct orientation
plt.colorbar(label="Cumulative Flux (ant·pixels/s)")
plt.title(f"Cumulative Flux Map (All Directions, Site {site}, {num_days} Days)")
plt.show()




pct = 70
binary_mask = grid >= np.percentile(grid[grid>0], pct)
plt.imshow(binary_mask, cmap="gray")
plt.show()

### 2. threshold mask and split skeleton segments

_, nest_mask = cv2.threshold(nest_mask, 127, 255, cv2.THRESH_BINARY)
nest_mask = ~nest_mask
segments, nest_junctions, endpoints = split_trails_from_nest(binary_mask, nest_mask, max_dist=5, n_points=20, min_length=10, min_spacing=3, dilate_kernel_size=0, trail_method='shortest_paths', max_depth=50, max_trails=100, plot=True)


filtered_segments = remove_subset_trails(segments, tol=2, overlap_threshold=0.90)


# G = merge_trails_as_graph(segments)
# filtered_segments = extract_unique_branches(G, nest_junctions, endpoints)

# filtered_segments = remove_subset_trails(filtered_segments, tol=2)

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



########## analysis 4: analysis of separated trails - timeseries ##########################

trail_counts, trail_counts_raw = plot_trail_ant_counts_timeseries(
    seg_masks, 
    days_period, 
    num_days, 
    site_id=site,
    direction_filter=None,  # All directions
    bin_size=20
)



# First, get the trail counts for both directions
trail_counts_toward, trail_counts_raw_toward = plot_trail_ant_counts_timeseries(
    seg_masks, days_period, num_days, site_id=site,
    direction_filter='toward', bin_size=20
)

trail_counts_away, trail_counts_raw_away = plot_trail_ant_counts_timeseries(
    seg_masks, days_period, num_days, site_id=site,
    direction_filter='away', bin_size=20
)


#plot_trail_density_difference(trail_counts_away, trail_counts_toward, seg_masks, 
#                                   site_id=site, title_suffix="", figsize=(12, 8))


### test for bidirectional analysis (using density values)
dbr_test_results = test_trail_bidirectionality(trail_counts_away, trail_counts_toward, num_days=num_days)

plot_dbr_timeseries_per_trail(dbr_test_results, site_id=site)
import sys;sys.exit(0)


# # Create trail usage across days plot
# plot_trail_usage_across_days(
#     trail_fractions_toward, 
#     trail_fractions_away,
#     days_period, 
#     num_days, 
#     site_id=site
# )


# # Create daily average trail usage plot
# plot_daily_average_trail_usage(
#     trail_fractions_toward, 
#     trail_fractions_away,
#     days_period, 
#     num_days, 
#     site_id=site
# )

# # Perform lagged correlation analysis
# correlation_results = lagged_correlation_analysis(
#     trail_fractions_toward, 
#     trail_fractions_away,
#     max_lag_hours=4,  # Test up to 12 hours
#     min_correlation_samples=5
# )

# # Create summary plots
# plot_lagged_correlation_summary(correlation_results)


import sys; sys.exit(0)

######## analysis 5: clustering flux maps - not doing this for now ##################
 
#df_clustering = build_flux_dataset(days_period, num_days, site_id=site, direction_filter="away", bin_size=20)
#df_clustering, X_pca, model = cluster_flux_maps(df_clustering, n_components=40, n_clusters=2)
#plot_clusters(df_clustering, X_pca)
#####################################################


