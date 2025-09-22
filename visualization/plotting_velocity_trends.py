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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_flow_field_for_hour(flow_field_data, hour, site_id=1, bin_size=20, 
                             direction_filter=None, ax=None, 
                             vmin=None, vmax=None, cmap="Reds"):
    """
    Plot flow field for a given hour using velocity data.
    """
    if not flow_field_data:
        print(f"No flow field data available for hour {hour}")
        return None
    
    all_velocity_data = pd.concat(flow_field_data, ignore_index=True)
    
    # Filter by direction
    if direction_filter:
        all_velocity_data = all_velocity_data[all_velocity_data['direction'] == direction_filter]
        if len(all_velocity_data) == 0:
            print(f"No data for direction '{direction_filter}' in hour {hour}")
            return None
    
    # Exclude stationary ants
    moving_data = all_velocity_data[all_velocity_data['velocity'] > 0].copy()
    if len(moving_data) == 0:
        print(f"No moving ants found for hour {hour}")
        return None
    
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
        return None
    
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

# Create a pandas period object for the analysis
## beer
#days_period = pd.Period('2024-08-01', freq='D')

## rain
#days_period = pd.Period('2024-08-22', freq='D')
#days_period = pd.Period('2024-10-03', freq='D')
#days_period = pd.Period('2024-11-15', freq='D')



# Get the sites where we have data for the period we are interested in
#sites = set(df.loc[(df.time_stamp.dt.day == days_period.start_time.day) & (df.time_stamp.dt.month == days_period.start_time.month)].site_id)
#sites = list(sites)
sites = [1]  # site 1: beer, site 2: shack, site 3: rain

# Run the velocity analysis
for site in sites:
    

    # Uncomment the line below to pre-process all velocity data for faster future analysis
    # This will create CSV files with per-frame velocity data for each video
    
    #preprocess_all_velocity_data(days_period, 10, site)
    days_period = pd.Period('2024-08-01', freq='D')

    #days_period = pd.Period('2024-10-22', freq='D')
    # preprocess_all_velocity_data(days_period, 12, site)

    # ### rain
    #site = 3
    #days_period = pd.Period('2024-08-22', freq='D')
    #preprocess_all_velocity_data(days_period, 12, site)

    #days_period = pd.Period('2024-10-03', freq='D')
    # preprocess_all_velocity_data(days_period, 17, site)

    #days_period = pd.Period('2024-11-15', freq='D')
    # preprocess_all_velocity_data(days_period, 23, site)

    # ## shack 
    #site = 2
    #days_period = pd.Period('2024-08-01', freq='D')
    #preprocess_all_velocity_data(days_period, 50, site)

    #scatter_plot_velocity_for_day_range(days_period, 23, site)

    
    # Example of how to plot flow fields:
    for hour in range(12,24):
        flow_data = get_flow_field_data_for_hour(days_period+3, hour, site_id=site)
        #plot_flow_field_for_hour(flow_data, hour, site_id=site, bin_size=20, direction_filter='away')  # Away direction only
        #plot_flow_field_for_hour(flow_data, hour, site_id=site, bin_size=20, direction_filter='toward')  # Toward direction only
        #plot_flow_field_for_hour(flow_data, hour, site_id=site, bin_size=20, direction_filter='unknown')  # Toward direction only
        #plt.tight_layout()
        #plt.show()
        #plot_flow_field_for_hour(flow_data, hour, site_id=1, bin_size=20, direction_filter='toward')  # Toward direction only
        
        #plot_flow_field_for_hour(flow_data, hour, site_id=1, bin_size=30, direction_filter=None)  # All directions
        plot_flow_field_comparison(flow_data, hour, site_id=site, bin_size=20)  # Side-by-side comparison