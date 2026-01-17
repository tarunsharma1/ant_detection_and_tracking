"""
Trajectory Clustering for Ant Behavior Analysis

This module implements trajectory clustering to identify:
1. Different behaviors (excavation loops, foraging, exploring)
2. Different trails used by ants

The approach uses multiple clustering strategies:
- Trajectory shape clustering (using DTW, Hausdorff distance)
- Spatial clustering (identifying trail segments)
- Behavioral clustering (based on movement patterns)
"""

import sys
sys.path.append('../')
from mysql_dataset import database_helper
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.optimize import linear_sum_assignment
import cv2
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize, binary_dilation, disk
from skimage.segmentation import find_boundaries
import networkx as nx
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Additional imports for advanced trajectory analysis
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.stats import entropy
import os

class TrajectoryClusterer:
    """
    Main class for clustering ant trajectories to identify behaviors and trails.
    """
    
    def __init__(self, site_id=1, nest_mask_path=None):
        self.site_id = site_id
        self.trajectories = []
        self.trajectory_features = []
        self.cluster_labels = None
        self.behavior_clusters = None
        self.trail_clusters = None
        self.nest_mask = None
        self.nest_mask_path = nest_mask_path
        
    def load_trajectory_data(self, days_period, num_days, direction_filter=None, 
                           smooth_trajectories=True, detect_id_swaps=True):
        """
        Load trajectory data from database for specified time period.
        
        Parameters
        ----------
        days_period : pd.Period
            Starting day period
        num_days : int
            Number of days to analyze
        direction_filter : str or None
            Filter by direction ('away', 'toward', 'unknown', or None for all)
        smooth_trajectories : bool
            Whether to apply smoothing to trajectories
        detect_id_swaps : bool
            Whether to detect and handle ID swaps
        """
        print(f"Loading trajectory data for site {self.site_id}, {num_days} days...")
        
        # Connect to database
        connection = database_helper.create_connection("localhost", "root", "master", "ant_colony_db")
        cursor = connection.cursor()
        
        query = """
        SELECT Counts.video_id, Counts.herdnet_tracking_with_direction_closest_boundary_method_csv,
        Videos.temperature, Videos.humidity, Videos.LUX, Videos.time_stamp, Videos.site_id 
        FROM Counts INNER JOIN Videos ON Counts.video_id=Videos.video_id
        WHERE Videos.site_id = %s
        """
        cursor.execute(query, (self.site_id,))
        table_rows = cursor.fetchall()
        
        df = pd.DataFrame(table_rows, columns=cursor.column_names)
        df["time_stamp"] = pd.to_datetime(df["time_stamp"])
        
        # Filter by date range
        df_per_site = df.loc[df.site_id == self.site_id]
        trajectories = []
        
        for i in range(num_days):
            current_day = days_period + i
            df_per_day = df_per_site.loc[
                (df_per_site.time_stamp.dt.day == current_day.start_time.day) & 
                (df_per_site.time_stamp.dt.month == current_day.start_time.month)
            ]
            
            for _, video in df_per_day.iterrows():
                tracking_csv = video['herdnet_tracking_with_direction_closest_boundary_method_csv']
                try:
                    data = pd.read_csv(tracking_csv)
                    if len(data) == 0:
                        continue
                        
                    # Filter by direction if specified
                    if direction_filter:
                        data = data[data['direction'] == direction_filter]
                        if len(data) == 0:
                            continue
                    
                    # Extract trajectories for each ant
                    for ant_id in data['ant_id'].unique():
                        ant_track = data[data['ant_id'] == ant_id].copy()
                        if len(ant_track) < 20:  # Skip very short tracks
                            continue

                        # Calculate center points
                        ant_track['center_x'] = (ant_track['x1'] + ant_track['x2']) / 2
                        ant_track['center_y'] = (ant_track['y1'] + ant_track['y2']) / 2
                        
                        # Add metadata
                        ant_track['video_id'] = video['video_id']
                        ant_track['hour'] = video['time_stamp'].hour
                        ant_track['day'] = i
                        
                        trajectories.append(ant_track)
                        
                except Exception as e:
                    print(f"Error loading {tracking_csv}: {e}")
                    continue
        
        print(f"Loaded {len(trajectories)} raw trajectories")
        
        # Apply preprocessing
        if smooth_trajectories:
            print("Applying trajectory smoothing...")
            trajectories = self._smooth_trajectories(trajectories)
        
        if detect_id_swaps:
            print("Detecting and handling ID swaps...")
            trajectories = self._detect_and_fix_id_swaps(trajectories)
        
        self.trajectories = trajectories
        print(f"Final dataset: {len(trajectories)} trajectories")
        
        # Load nest mask if provided
        if self.nest_mask_path and os.path.exists(self.nest_mask_path):
            self._load_nest_mask()
        
        return trajectories
    
    def load_hourly_trajectory_data(self, target_date, target_hour, direction_filter=None,
                                  smooth_trajectories=True, detect_id_swaps=True):
        """
        Load trajectory data for a specific hour from a specific day.
        
        Parameters
        ----------
        target_date : str or pd.Timestamp
            Target date (e.g., '2024-08-26' or pd.Timestamp('2024-08-26'))
        target_hour : int
            Target hour (0-23)
        direction_filter : str or None
            Filter by direction ('away', 'toward', 'unknown', or None for all)
        smooth_trajectories : bool
            Whether to apply smoothing to trajectories
        detect_id_swaps : bool
            Whether to detect and handle ID swaps
        """
        print(f"Loading trajectory data for site {self.site_id}, {target_date} at hour {target_hour}...")
        
        # Convert target_date to datetime if string
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
        
        # Connect to database
        connection = database_helper.create_connection("localhost", "root", "master", "ant_colony_db")
        cursor = connection.cursor()
        
        query = """
        SELECT Counts.video_id, Counts.herdnet_tracking_with_direction_closest_boundary_method_csv,
        Videos.temperature, Videos.humidity, Videos.LUX, Videos.time_stamp, Videos.site_id 
        FROM Counts INNER JOIN Videos ON Counts.video_id=Videos.video_id
        WHERE Videos.site_id = %s
        """
        cursor.execute(query, (self.site_id,))
        table_rows = cursor.fetchall()
        
        df = pd.DataFrame(table_rows, columns=cursor.column_names)
        df["time_stamp"] = pd.to_datetime(df["time_stamp"])
        
        # Filter by site, date, and hour
        df_filtered = df.loc[
            (df.site_id == self.site_id) &
            (df.time_stamp.dt.date == target_date.date()) &
            (df.time_stamp.dt.hour == target_hour)
        ]
        
        if len(df_filtered) == 0:
            print(f"No videos found for site {self.site_id} on {target_date} at hour {target_hour}")
            return []
        
        print(f"Found {len(df_filtered)} videos for the specified hour")
        
        trajectories = []
        
        for _, video in df_filtered.iterrows():
            tracking_csv = video['herdnet_tracking_with_direction_closest_boundary_method_csv']
            try:
                data = pd.read_csv(tracking_csv)
                if len(data) == 0:
                    continue
                    
                # Filter by direction if specified
                if direction_filter:
                    data = data[data['direction'] == direction_filter]
                    if len(data) == 0:
                        continue
                
                # Extract trajectories for each ant
                for ant_id in data['ant_id'].unique():
                    ant_track = data[data['ant_id'] == ant_id].copy()
                    if len(ant_track) < 20:  # Skip very short tracks
                        continue
                        
                    # Calculate center points
                    ant_track['center_x'] = (ant_track['x1'] + ant_track['x2']) / 2
                    ant_track['center_y'] = (ant_track['y1'] + ant_track['y2']) / 2
                    
                    # Add metadata
                    ant_track['video_id'] = video['video_id']
                    ant_track['hour'] = video['time_stamp'].hour
                    ant_track['day'] = target_date.day
                    ant_track['date'] = target_date.date()
                    
                    trajectories.append(ant_track)
                    
            except Exception as e:
                print(f"Error loading {tracking_csv}: {e}")
                continue
        
        print(f"Loaded {len(trajectories)} raw trajectories for hour {target_hour}")
        
        # Apply preprocessing
        if smooth_trajectories:
            print("Applying trajectory smoothing...")
            trajectories = self._smooth_trajectories(trajectories)
        
        if detect_id_swaps:
            print("Detecting and handling ID swaps...")
            trajectories = self._detect_and_fix_id_swaps(trajectories)
        
        self.trajectories = trajectories
        print(f"Final dataset: {len(trajectories)} trajectories for hour {target_hour}")
        
        # Load nest mask if provided
        if self.nest_mask_path and os.path.exists(self.nest_mask_path):
            self._load_nest_mask()
        
        return trajectories
    
    
    
    def _smooth_trajectories(self, trajectories, window_size=3, method='savgol'):
        """
        Apply smoothing to trajectories to reduce noise.
        
        Parameters
        ----------
        trajectories : list
            List of trajectory DataFrames
        window_size : int
            Size of smoothing window
        method : str
            Smoothing method ('moving_average', 'gaussian', 'savgol')
        """
        smoothed_trajectories = []
        
        for traj in trajectories:
            if len(traj) < window_size:
                smoothed_trajectories.append(traj)
                continue
            
            # Sort by frame number
            traj = traj.sort_values('frame_number').reset_index(drop=True)
            
            # Apply smoothing to x and y coordinates
            if method == 'moving_average':
                traj['center_x_smooth'] = traj['center_x'].rolling(window=window_size, center=True).mean()
                traj['center_y_smooth'] = traj['center_y'].rolling(window=window_size, center=True).mean()
            elif method == 'gaussian':
                from scipy.ndimage import gaussian_filter1d
                traj['center_x_smooth'] = gaussian_filter1d(traj['center_x'], sigma=1.0)
                traj['center_y_smooth'] = gaussian_filter1d(traj['center_y'], sigma=1.0)
            elif method == 'savgol':
                from scipy.signal import savgol_filter
                if len(traj) >= window_size:
                    traj['center_x_smooth'] = savgol_filter(traj['center_x'], window_size, 2)
                    traj['center_y_smooth'] = savgol_filter(traj['center_y'], window_size, 2)
                else:
                    traj['center_x_smooth'] = traj['center_x']
                    traj['center_y_smooth'] = traj['center_y']
            
            # Fill NaN values at edges
            traj['center_x_smooth'] = traj['center_x_smooth'].fillna(traj['center_x'])
            traj['center_y_smooth'] = traj['center_y_smooth'].fillna(traj['center_y'])
            
            # Update center coordinates
            traj['center_x'] = traj['center_x_smooth']
            traj['center_y'] = traj['center_y_smooth']
            
            # Drop temporary columns
            traj = traj.drop(['center_x_smooth', 'center_y_smooth'], axis=1)
            
            smoothed_trajectories.append(traj)
        
        print(f"Applied {method} smoothing with window size {window_size}")
        return smoothed_trajectories
    
    def _detect_and_fix_id_swaps(self, trajectories, max_jump_distance=100, 
                                 min_trajectory_length=20):
        """
        Detect and fix ID swaps in trajectories.
        
        Parameters
        ----------
        trajectories : list
            List of trajectory DataFrames
        max_jump_distance : float
            Maximum distance for a valid movement between consecutive frames
        min_trajectory_length : int
            Minimum length for a trajectory to be considered
        """
        print("Detecting ID swaps...")
        
        # Group trajectories by video and hour for ID swap detection
        video_trajectories = defaultdict(list)
        for traj in trajectories:
            video_id = traj['video_id'].iloc[0]
            hour = traj['hour'].iloc[0]
            key = (video_id, hour)
            video_trajectories[key].append(traj)
        
        cleaned_trajectories = []
        total_swaps_detected = 0
        
        for (video_id, hour), video_trajs in video_trajectories.items():
            if len(video_trajs) < 2:
                cleaned_trajectories.extend(video_trajs)
                continue
            
            # Sort trajectories by ant_id and frame_number
            for traj in video_trajs:
                traj = traj.sort_values('frame_number').reset_index(drop=True)
            
            # Detect potential ID swaps
            cleaned_video_trajs = self._fix_id_swaps_in_video(video_trajs, max_jump_distance)
            cleaned_trajectories.extend(cleaned_video_trajs)
            
            # Count detected swaps
            original_ids = set(traj['ant_id'].iloc[0] for traj in video_trajs)
            cleaned_ids = set(traj['ant_id'].iloc[0] for traj in cleaned_video_trajs)
            swaps_in_video = len(original_ids) - len(cleaned_ids)
            total_swaps_detected += swaps_in_video
        
        print(f"Detected and fixed {total_swaps_detected} potential ID swaps")
        
        # Filter out very short trajectories after cleaning
        final_trajectories = [traj for traj in cleaned_trajectories 
                            if len(traj) >= min_trajectory_length]
        
        print(f"Filtered to {len(final_trajectories)} trajectories after ID swap detection")
        return final_trajectories
    
    def _fix_id_swaps_in_video(self, video_trajectories, max_jump_distance):
        """
        Fix ID swaps within a single video.
        """
        if len(video_trajectories) < 2:
            return video_trajectories
        
        # Create a list of all detections with their metadata
        all_detections = []
        for traj_idx, traj in enumerate(video_trajectories):
            for _, row in traj.iterrows():
                all_detections.append({
                    'frame_number': row['frame_number'],
                    'x': row['center_x'],
                    'y': row['center_y'],
                    'original_ant_id': row['ant_id'],
                    'trajectory_idx': traj_idx,
                    'row_data': row
                })
        
        # Sort by frame number
        all_detections.sort(key=lambda x: x['frame_number'])
        
        # Group by frame
        frame_groups = defaultdict(list)
        for det in all_detections:
            frame_groups[det['frame_number']].append(det)
        
        # Detect potential swaps by looking for large jumps
        potential_swaps = []
        for frame_num in sorted(frame_groups.keys())[1:]:  # Skip first frame
            prev_frame = frame_num - 1
            if prev_frame not in frame_groups:
                continue
            
            current_detections = frame_groups[frame_num]
            prev_detections = frame_groups[prev_frame]
            
            # For each current detection, find the closest previous detection
            for curr_det in current_detections:
                min_distance = float('inf')
                best_prev_det = None
                
                for prev_det in prev_detections:
                    distance = np.sqrt((curr_det['x'] - prev_det['x'])**2 + 
                                     (curr_det['y'] - prev_det['y'])**2)
                    if distance < min_distance:
                        min_distance = distance
                        best_prev_det = prev_det
                
                # If the closest previous detection has a different ant_id,
                # and the distance is reasonable, it might be a swap
                if (best_prev_det and 
                    curr_det['original_ant_id'] != best_prev_det['original_ant_id'] and
                    min_distance < max_jump_distance):
                    
                    potential_swaps.append({
                        'frame': frame_num,
                        'current': curr_det,
                        'previous': best_prev_det,
                        'distance': min_distance
                    })
        
        # Apply swap corrections
        corrected_trajectories = []
        for traj_idx, traj in enumerate(video_trajectories):
            corrected_traj = traj.copy()
            
            # Apply corrections for this trajectory
            for swap in potential_swaps:
                if swap['current']['trajectory_idx'] == traj_idx:
                    # This trajectory should get the ant_id from the previous detection
                    frame_mask = corrected_traj['frame_number'] == swap['frame']
                    if frame_mask.any():
                        corrected_traj.loc[frame_mask, 'ant_id'] = swap['previous']['original_ant_id']
            
            corrected_trajectories.append(corrected_traj)
        
        return corrected_trajectories
    
    def _load_nest_mask(self):
        """Load nest mask from file."""
        try:
            import cv2
            self.nest_mask = cv2.imread(self.nest_mask_path, cv2.IMREAD_GRAYSCALE)
            
            if self.nest_mask is not None:
                # Convert to binary mask (0 = background, 255 = nest)
                _, self.nest_mask = cv2.threshold(self.nest_mask, 127, 255, cv2.THRESH_BINARY)
                
                print(f"✅ Loaded nest mask from {self.nest_mask_path}")
                print(f"   Nest mask shape: {self.nest_mask.shape}")
            else:
                print(f"❌ Failed to load nest mask from {self.nest_mask_path}")
        except Exception as e:
            print(f"❌ Error loading nest mask: {e}")
            self.nest_mask = None
    
    def _calculate_nest_proximity(self, coords):
        """
        Calculate proximity to nest entrance for trajectory start point only.
        
        Parameters
        ----------
        coords : array
            Array of (x, y) coordinates
            
        Returns
        -------
        float
            Distance from trajectory start to nest entrance (pixels)
        """
        if self.nest_mask is None:
            return float('inf')  # No nest mask available
        
        try:
            from scipy.ndimage import distance_transform_edt
            
            # Calculate distance to nearest nest pixel
            # Invert mask so nest pixels are 0 (background) and non-nest pixels are 1
            inverted_mask = self.nest_mask.astype(float)
            dist_transform = distance_transform_edt(inverted_mask)
            
            # Only check the start point of the trajectory
            start_x, start_y = coords[0]
            
            if 0 <= int(start_y) < dist_transform.shape[0] and 0 <= int(start_x) < dist_transform.shape[1]:
                return dist_transform[int(start_y), int(start_x)]
            else:
                return float('inf')
            
        except Exception as e:
            print(f"⚠️ Error calculating nest proximity: {e}")
            return float('inf')
    
    def _detect_outliers(self, trajectories, outlier_threshold=3.0):
        """
        Detect and optionally remove outlier points in trajectories.
        
        Parameters
        ----------
        trajectories : list
            List of trajectory DataFrames
        outlier_threshold : float
            Z-score threshold for outlier detection
        """
        cleaned_trajectories = []
        total_outliers_removed = 0
        
        for traj in trajectories:
            if len(traj) < 5:
                cleaned_trajectories.append(traj)
                continue
            
            # Calculate distances between consecutive points
            traj = traj.sort_values('frame_number').reset_index(drop=True)
            distances = []
            
            for i in range(1, len(traj)):
                dist = np.sqrt((traj.iloc[i]['center_x'] - traj.iloc[i-1]['center_x'])**2 + 
                              (traj.iloc[i]['center_y'] - traj.iloc[i-1]['center_y'])**2)
                distances.append(dist)
            
            if len(distances) < 3:
                cleaned_trajectories.append(traj)
                continue
            
            # Detect outliers using Z-score
            distances = np.array(distances)
            z_scores = np.abs((distances - np.mean(distances)) / (np.std(distances) + 1e-6))
            outlier_mask = z_scores > outlier_threshold
            
            if np.any(outlier_mask):
                # Remove outlier points
                outlier_indices = np.where(outlier_mask)[0] + 1  # +1 because distances are between points
                traj_cleaned = traj.drop(traj.index[outlier_indices]).reset_index(drop=True)
                total_outliers_removed += len(outlier_indices)
                cleaned_trajectories.append(traj_cleaned)
            else:
                cleaned_trajectories.append(traj)
        
        print(f"Removed {total_outliers_removed} outlier points")
        return cleaned_trajectories
    
    
    
    def extract_trajectory_features(self, trajectories=None):
        """
        Extract features from trajectories for clustering - OPTIMIZED VERSION.
        
        Parameters
        ----------
        trajectories : list or None
            List of trajectory DataFrames
        
        """
        if trajectories is None:
            trajectories = self.trajectories
            
        print(f"Extracting trajectory features for {len(trajectories)} trajectories...")
        
        # Pre-filter trajectories by length
        valid_trajectories = [traj for traj in trajectories if len(traj) >= 20]
        print(f"Processing {len(valid_trajectories)} valid trajectories (>=20 points)")
        
        if len(valid_trajectories) == 0:
            print("No valid trajectories found")
            return np.array([])
        
        # Pre-allocate feature array
        n_trajectories = len(valid_trajectories)
        features = np.zeros((n_trajectories, 4))  # 4 features
        
        # Batch process trajectories
        print("Computing features in batches...")
        batch_size = 1000
        
        for batch_start in range(0, n_trajectories, batch_size):
            batch_end = min(batch_start + batch_size, n_trajectories)
            batch_trajectories = valid_trajectories[batch_start:batch_end]
            
            # Process batch
            batch_features = self._extract_features_batch(batch_trajectories)
            features[batch_start:batch_end] = batch_features
            
            if batch_start % 5000 == 0:
                print(f"  Processed {batch_end}/{n_trajectories} trajectories")
        
        self.trajectory_features = features
        print(f"✅ Extracted features for {len(features)} trajectories")
        return self.trajectory_features
    
    def _extract_features_batch(self, trajectories):
        """
        Extract features for a batch of trajectories efficiently.
        """
        batch_size = len(trajectories)
        features = np.zeros((batch_size, 4))
        
        for i, traj in enumerate(trajectories):
            # Sort by frame number
            traj = traj.sort_values('frame_number').reset_index(drop=True)
            coords = traj[['center_x', 'center_y']].values
            
            # 1. Trajectory length (vectorized)
            if len(coords) > 1:
                diffs = np.diff(coords, axis=0)
                distances = np.linalg.norm(diffs, axis=1)
                total_length = np.sum(distances)
            else:
                total_length = 0
            
            # 2. Efficiency (straight-line distance / total length)
            if len(coords) > 0:
                straight_distance = np.linalg.norm(coords[-1] - coords[0])
                efficiency = straight_distance / (total_length + 1e-6)
            else:
                efficiency = 0
            
            # 3. Loop strength (how close start/end are)
            # if len(coords) > 0:
            #     loop_distance = np.linalg.norm(coords[-1] - coords[0])
            #     loop_strength = 1.0 / (loop_distance + 1)
            # else:
            #     loop_strength = 0
            
            # # 4. Circularity (how circular the trajectory is)
            # if len(coords) > 10:
            #     center = np.mean(coords, axis=0)
            #     distances_from_center = np.linalg.norm(coords - center, axis=1)
            #     radius_std = np.std(distances_from_center)
            #     radius_mean = np.mean(distances_from_center)
            #     circularity = 1.0 / (radius_std / (radius_mean + 1e-6) + 1)
            # else:
            #     circularity = 0
            
            # 5. Average absolute angle changes (using tracking data angle column)
            if len(traj) > 1 and 'angle' in traj.columns:
                angles = traj['angle'].values
                angle_changes = np.abs(np.diff(angles))
                # Handle angle wrapping (e.g., 359° to 1° should be 2°, not 358°)
                angle_changes = np.minimum(angle_changes, 360 - angle_changes)
                avg_angle_change = np.mean(angle_changes)
            else:
                avg_angle_change = 0
            
            # # 6. Speed and pause ratio (vectorized)
            # if len(coords) > 1:
            #     frame_diff = traj['frame_number'].diff().values[1:]
            #     speeds = distances / (frame_diff + 1e-6) * 20  # 20 fps
            #     avg_speed = np.mean(speeds)
            #     pause_ratio = np.sum(speeds < 1.0) / len(speeds)
            # else:
            #     avg_speed = 0
            #     pause_ratio = 0
            
            # # 7. Spatial compactness (vectorized)
            # if len(coords) > 0:
            #     bbox_area = (coords[:, 0].max() - coords[:, 0].min()) * (coords[:, 1].max() - coords[:, 1].min())
            #     spatial_compactness = 1.0 / (bbox_area + 1)
            # else:
            #     spatial_compactness = 0
            
            # Store features
            features[i] = [
                efficiency,
                straight_distance,
                avg_angle_change,
                total_length
            ]
        
        return features
    
    def _sample_representative_trajectories(self, trajectories, max_samples):
        """
        Sample representative trajectories based on length and complexity.
        """
        if len(trajectories) <= max_samples:
            return trajectories
        
        # Calculate trajectory lengths
        lengths = [len(traj) for traj in trajectories]
        
        # Sample from different length quartiles
        quartiles = np.percentile(lengths, [25, 50, 75])
        samples_per_quartile = max_samples // 4
        
        selected = []
        for i, traj in enumerate(trajectories):
            length = lengths[i]
            if len(selected) < max_samples:
                if length <= quartiles[0] and len([t for t in selected if lengths[trajectories.index(t)] <= quartiles[0]]) < samples_per_quartile:
                    selected.append(traj)
                elif quartiles[0] < length <= quartiles[1] and len([t for t in selected if quartiles[0] < lengths[trajectories.index(t)] <= quartiles[1]]) < samples_per_quartile:
                    selected.append(traj)
                elif quartiles[1] < length <= quartiles[2] and len([t for t in selected if quartiles[1] < lengths[trajectories.index(t)] <= quartiles[2]]) < samples_per_quartile:
                    selected.append(traj)
                elif length > quartiles[2] and len([t for t in selected if lengths[trajectories.index(t)] > quartiles[2]]) < samples_per_quartile:
                    selected.append(traj)
        
        return selected[:max_samples]
    
    def _sample_diverse_trajectories(self, trajectories, max_samples):
        """
        Sample diverse trajectories using clustering-based selection.
        """
        if len(trajectories) <= max_samples:
            return trajectories
        
        # Extract simple features for diversity
        features = []
        for traj in trajectories:
            coords = traj[['center_x', 'center_y']].values
            if len(coords) > 1:
                length = np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1))
                straight_dist = np.linalg.norm(coords[-1] - coords[0])
                efficiency = straight_dist / (length + 1e-6)
                features.append([length, efficiency])
            else:
                features.append([0, 0])
        
        features = np.array(features)
        
        # Simple clustering to find diverse samples
        from sklearn.cluster import KMeans
        n_clusters = min(max_samples, len(trajectories))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features)
        
        # Select one trajectory from each cluster
        selected = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) > 0:
                # Select the trajectory closest to cluster center
                center = kmeans.cluster_centers_[cluster_id]
                distances = np.linalg.norm(features[cluster_indices] - center, axis=1)
                best_idx = cluster_indices[np.argmin(distances)]
                selected.append(trajectories[best_idx])
        
        return selected[:max_samples]
    
    

    
    
    def cluster_behaviors(self, n_clusters=None, method='kmeans', eps=2.0, min_samples=3, 
                         min_cluster_size=5, min_samples_hdbscan=3, use_scaling=True):
        """
        Cluster trajectories based on behavioral features.
        
        Parameters
        ----------
        n_clusters : int or None
            Number of clusters. If None, will use elbow method to determine.
        method : str
            Clustering method ('kmeans', 'dbscan', 'hdbscan', 'agglomerative')
        eps : float
            DBSCAN epsilon parameter (distance threshold).
        min_samples : int
            DBSCAN minimum samples parameter.
        min_cluster_size : int
            HDBSCAN minimum cluster size.
        min_samples_hdbscan : int
            HDBSCAN minimum samples parameter.
        use_scaling : bool
            Whether to scale features before clustering.
        """
        if self.trajectory_features is None:
            raise ValueError("Must extract features first")
        
        # Scale features if requested
        if use_scaling:
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(self.trajectory_features)
            print(f"✅ Features scaled: {features_scaled.shape}")
        else:
            features_scaled = self.trajectory_features
        
        if method == 'kmeans':
            if n_clusters is None:
                # Use elbow method to determine optimal clusters
                inertias = []
                K_range = range(2, min(11, len(features_scaled)//2))
                for k in K_range:
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    kmeans.fit(features_scaled)
                    inertias.append(kmeans.inertia_)
                
                # Find elbow point
                diffs = np.diff(inertias)
                second_diffs = np.diff(diffs)
                elbow_idx = np.argmax(second_diffs) + 2
                n_clusters = K_range[elbow_idx]
                print(f"Optimal number of clusters: {n_clusters}")
            
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            
        
        
        self.cluster_labels = clusterer.fit_predict(features_scaled)
        
        # Report clustering results
        n_clusters_found = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        n_noise = list(self.cluster_labels).count(-1)
        
        print(f"🔍 {method.upper()} Results:")
        print(f"   Clusters found: {n_clusters_found}")
        print(f"   Noise points: {n_noise}")
        print(f"   Noise ratio: {n_noise/len(self.cluster_labels):.2%}")
        
        # Calculate clustering metrics
        if len(np.unique(self.cluster_labels)) > 1:
            silhouette = silhouette_score(features_scaled, self.cluster_labels)
            calinski = calinski_harabasz_score(features_scaled, self.cluster_labels)
            print(f"Silhouette score: {silhouette:.3f}")
            print(f"Calinski-Harabasz score: {calinski:.3f}")
        
        
        
        
        return self.cluster_labels
    
    
    
    
    
    def _analyze_feature_separation(self, features):
        """
        Analyze how well-separated the features are for clustering.
        """
        if len(features) == 0:
            print("No features to analyze.")
            return
        
        print(f"Feature Analysis for {len(features)} trajectories:")
        print(f"Feature dimensions: {features.shape}")
        
        # Check feature ranges and variance
        print("\nFeature Statistics:")
        
        feature_names = [
            'efficiency',
            'straight_distance',                    # Low efficiency = complex paths               # How close start/end are
            'avg_angle_change',              # Average absolute angle changes
            'total_length'
        ]
        
        for i, name in enumerate(feature_names):
            if i < features.shape[1]:
                values = features[:, i]
                print(f"  {name:20s}: mean={np.mean(values):.3f}, std={np.std(values):.3f}, "
                      f"min={np.min(values):.3f}, max={np.max(values):.3f}")
        
        # Check for constant features (no variance)
        constant_features = []
        for i in range(features.shape[1]):
            if np.std(features[:, i]) < 1e-6:
                constant_features.append(i)
        
        if constant_features:
            print(f"\n⚠️  Warning: {len(constant_features)} features have no variance:")
            for i in constant_features:
                if i < len(feature_names):
                    print(f"   - {feature_names[i]}")
        
        # Check feature correlations
        print(f"\nFeature Correlations (top 5 highest):")
        corr_matrix = np.corrcoef(features.T)
        correlations = []
        for i in range(len(feature_names)):
            for j in range(i+1, len(feature_names)):
                if i < corr_matrix.shape[0] and j < corr_matrix.shape[1]:
                    corr = abs(corr_matrix[i, j])
                    correlations.append((corr, feature_names[i], feature_names[j]))
        
        correlations.sort(reverse=True)
        for corr, feat1, feat2 in correlations[:5]:
            print(f"   {feat1} ↔ {feat2}: {corr:.3f}")
        
        # Check if features are well-distributed
        print(f"\nFeature Distribution Analysis:")
        for i, name in enumerate(feature_names):
            if i < features.shape[1]:
                values = features[:, i]
                # Check if values are concentrated in a small range
                value_range = np.max(values) - np.min(values)
                if value_range < 0.1:
                    print(f"   ⚠️  {name}: Very small range ({value_range:.3f}) - may not be discriminative")
                elif np.std(values) < 0.1:
                    print(f"   ⚠️  {name}: Low variance ({np.std(values):.3f}) - may not be discriminative")
                else:
                    print(f"   ✅ {name}: Good range ({value_range:.3f}) and variance ({np.std(values):.3f})")
        
        # Suggest improvements
        print(f"\n💡 Suggestions for better clustering:")
        if constant_features:
            print(f"   - Remove constant features: {[feature_names[i] for i in constant_features if i < len(feature_names)]}")
        
        high_corr_pairs = [pair for pair in correlations[:3] if pair[0] > 0.8]
        if high_corr_pairs:
            print(f"   - Consider removing highly correlated features:")
            for corr, feat1, feat2 in high_corr_pairs:
                print(f"     {feat1} ↔ {feat2} (correlation: {corr:.3f})")
        
        # Check if we have enough variation for clustering
        n_good_features = 0
        for i in range(features.shape[1]):
            if np.std(features[:, i]) > 0.1 and (np.max(features[:, i]) - np.min(features[:, i])) > 0.1:
                n_good_features += 1
        
        print(f"\n📊 Clustering Readiness:")
        print(f"   Good features: {n_good_features}/{features.shape[1]}")
        if n_good_features < 3:
            print(f"   ⚠️  Warning: Only {n_good_features} good features - clustering may be difficult")
            print(f"   💡 Consider: Adding more discriminative features or using different feature engineering")
        else:
            print(f"   ✅ Sufficient features for clustering")
    
    
    
    def identify_trails(self, trajectories=None, min_trail_length=100, eps=50):
        """
        Identify trails by clustering trajectory start/end points and common paths.
        
        Parameters
        ----------
        trajectories : list or None
            List of trajectory DataFrames
        min_trail_length : int
            Minimum length for a trail to be considered
        eps : float
            DBSCAN epsilon parameter for spatial clustering
        """
        if trajectories is None:
            trajectories = self.trajectories
        
        print("Identifying trails...")
        
        # Extract start and end points
        start_points = []
        end_points = []
        trajectory_info = []
        
        for i, traj in enumerate(trajectories):
            if len(traj) < 5:
                continue
                
            traj = traj.sort_values('frame_number').reset_index(drop=True)
            coords = traj[['center_x', 'center_y']].values
            
            start_points.append(coords[0])
            end_points.append(coords[-1])
            trajectory_info.append({
                'index': i,
                'start': coords[0],
                'end': coords[-1],
                'trajectory': coords,
                'length': len(coords)
            })
        
        start_points = np.array(start_points)
        end_points = np.array(end_points)
        
        # Cluster start points to identify trail origins
        start_clusterer = DBSCAN(eps=eps, min_samples=3)
        start_clusters = start_clusterer.fit_predict(start_points)
        
        # Cluster end points to identify trail destinations
        end_clusterer = DBSCAN(eps=eps, min_samples=3)
        end_clusters = end_clusterer.fit_predict(end_points)
        
        # Group trajectories by start-end cluster pairs
        trail_groups = defaultdict(list)
        
        for i, (start_cluster, end_cluster) in enumerate(zip(start_clusters, end_clusters)):
            if start_cluster != -1 and end_cluster != -1:  # Skip noise
                trail_key = (start_cluster, end_cluster)
                trail_groups[trail_key].append(trajectory_info[i])
        
        # Filter trails by minimum length and number of trajectories
        self.trail_clusters = {}
        trail_id = 0
        
        for trail_key, trajectories_in_trail in trail_groups.items():
            if len(trajectories_in_trail) >= 3:  # At least 3 trajectories
                avg_length = np.mean([t['length'] for t in trajectories_in_trail])
                if avg_length >= min_trail_length:
                    self.trail_clusters[trail_id] = {
                        'start_cluster': trail_key[0],
                        'end_cluster': trail_key[1],
                        'trajectories': trajectories_in_trail,
                        'avg_length': avg_length,
                        'count': len(trajectories_in_trail)
                    }
                    trail_id += 1
        
        print(f"Identified {len(self.trail_clusters)} trails")
        
        # Analyze each trail
        for trail_id, trail_data in self.trail_clusters.items():
            print(f"\nTrail {trail_id}:")
            print(f"  Trajectories: {trail_data['count']}")
            print(f"  Avg Length: {trail_data['avg_length']:.1f}")
            
            # Calculate trail direction
            start_positions = np.array([t['start'] for t in trail_data['trajectories']])
            end_positions = np.array([t['end'] for t in trail_data['trajectories']])
            
            avg_start = np.mean(start_positions, axis=0)
            avg_end = np.mean(end_positions, axis=0)
            direction = avg_end - avg_start
            
            print(f"  Direction: {direction}")
            print(f"  Start: ({avg_start[0]:.1f}, {avg_start[1]:.1f})")
            print(f"  End: ({avg_end[0]:.1f}, {avg_end[1]:.1f})")
        
        return self.trail_clusters
    
    
    
    def save_visualization_data(self, output_dir='/home/tarun/Desktop/plots_for_committee_meeting/trajectory_clustering/'):
        """
        Save trajectory data and cluster labels needed for visualization.
        This allows visualize_clusters() to be called independently without re-running clustering.
        """
        import pickle
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if self.trajectories is None or self.cluster_labels is None:
            print("❌ No trajectories or cluster labels to save")
            return
        
        # Extract essential data for visualization
        visualization_data = {
            'trajectories': [],
            'cluster_labels': self.cluster_labels.tolist(),
            'ant_ids': []
        }
        
        print(f"💾 Saving visualization data for {len(self.trajectories)} trajectories...")
        
        for i, traj in enumerate(self.trajectories):
            if i < len(self.cluster_labels):
                # Save only coordinates and ant_id (minimal data needed for visualization)
                traj_sorted = traj.sort_values('frame_number')
                coords = traj_sorted[['center_x', 'center_y']].values
                ant_id = traj_sorted['ant_id'].iloc[0]
                
                visualization_data['trajectories'].append(coords)
                visualization_data['ant_ids'].append(ant_id)
        
        # Save to pickle file
        output_file = os.path.join(output_dir, 'visualization_data.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(visualization_data, f)
        
        print(f"✅ Visualization data saved to {output_file}")
        print(f"   Trajectories: {len(visualization_data['trajectories'])}")
        print(f"   Clusters: {len(set(self.cluster_labels))}")
    
    def load_visualization_data(self, input_file='/home/tarun/Desktop/plots_for_committee_meeting/trajectory_clustering/visualization_data.pkl'):
        """
        Load trajectory data and cluster labels for visualization.
        
        Parameters
        ----------
        input_file : str
            Path to saved visualization data pickle file
        """
        import pickle
        import os
        
        if not os.path.exists(input_file):
            print(f"❌ Visualization data file not found: {input_file}")
            return False
        
        print(f"📂 Loading visualization data from {input_file}...")
        
        try:
            with open(input_file, 'rb') as f:
                visualization_data = pickle.load(f)
            
            # Reconstruct trajectories as DataFrames (minimal structure for visualization)
            self.trajectories = []
            for coords, ant_id in zip(visualization_data['trajectories'], visualization_data['ant_ids']):
                traj_df = pd.DataFrame(coords, columns=['center_x', 'center_y'])
                traj_df['ant_id'] = ant_id
                traj_df['frame_number'] = range(len(coords))  # Dummy frame numbers
                self.trajectories.append(traj_df)
            
            self.cluster_labels = np.array(visualization_data['cluster_labels'])
            
            print(f"✅ Loaded visualization data:")
            print(f"   Trajectories: {len(self.trajectories)}")
            print(f"   Clusters: {len(set(self.cluster_labels))}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading visualization data: {e}")
            return False
    
    def visualize_clusters(self, max_trajectories_per_cluster=500, figsize=(15, 10), 
                          sampling_method='random', show_density=True, 
                          load_from_file=None):
        """
        Visualize trajectory clusters with excavating ants highlighted.
        
        Parameters
        ----------
        max_trajectories_per_cluster : int
            Maximum number of trajectories to plot per cluster
        figsize : tuple
            Figure size
        sampling_method : str
            Sampling method ('random', 'representative', 'diverse')
        show_density : bool
            Whether to show trajectory density heatmap
        load_from_file : str or None
            If provided, load visualization data from this file instead of using self.trajectories
        """
        # Load data from file if specified
        if load_from_file:
            if not self.load_visualization_data(load_from_file):
                return
        
        if self.cluster_labels is None or self.trajectories is None:
            print("No clusters found. Run cluster_behaviors() first or load visualization data.")
            return
        
        # Load excavating ant IDs from annotation file
        excavating_ant_ids = []
        try:
            excavating_file = '/home/tarun/Desktop/plots_for_committee_meeting/excavation_analysis/excavating_annotation_beer-tree-08-01-2024_to_08-10-2024.csv'
            ann_df = pd.read_csv(excavating_file)
            if 'ant_id' in ann_df.columns and 'excavating' in ann_df.columns:
                excavating_ant_ids = (
                    ann_df[ann_df['excavating'].astype(str).str.strip().str.lower() == 'yes']['ant_id']
                    .dropna()
                    .astype(int)
                    .unique()
                    .tolist()
                )
                print(f"✅ Loaded {len(excavating_ant_ids)} excavating ant IDs from annotations")
            else:
                print("⚠️ Annotation file missing required columns 'ant_id' or 'excavating'. Using empty list.")
        except Exception as e:
            print(f"⚠️ Could not load excavating annotations: {e}. Using empty list.")
        
        # Get unique cluster IDs (excluding noise)
        unique_clusters = [c for c in np.unique(self.cluster_labels) if c != -1]
        n_clusters = len(unique_clusters)
        
        if n_clusters == 0:
            print("No valid clusters to visualize")
            return
        
        n_cols = min(3, n_clusters)
        n_rows = (n_clusters + n_cols - 1) // n_cols
        
        # Create subplots with shared axes for consistent sizing
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, 
                                 sharex=True, sharey=True, squeeze=False)
        axes = axes.flatten()
        
        # Calculate global axis limits from all trajectories
        all_x = []
        all_y = []
        for traj in self.trajectories:
            coords = traj[['center_x', 'center_y']].values
            all_x.extend(coords[:, 0])
            all_y.extend(coords[:, 1])
        
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        x_margin = (x_max - x_min) * 0.05
        y_margin = (y_max - y_min) * 0.05
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        
        for plot_idx, cluster_id in enumerate(unique_clusters):
            if plot_idx >= len(axes):
                break
                
            ax = axes[plot_idx]
            
            mask = self.cluster_labels == cluster_id
            cluster_trajectories = [self.trajectories[i] for i in np.where(mask)[0]]
            
            # Separate excavating and non-excavating trajectories
            excavating_trajectories = []
            other_trajectories = []
            
            for traj in cluster_trajectories:
                ant_id = traj['ant_id'].iloc[0]
                if ant_id in excavating_ant_ids:
                    excavating_trajectories.append(traj)
                else:
                    other_trajectories.append(traj)
            
            # Intelligent sampling of trajectories
            if sampling_method == 'representative':
                # Sample trajectories with different lengths and patterns
                other_trajectories = self._sample_representative_trajectories(other_trajectories, max_trajectories_per_cluster)
            elif sampling_method == 'diverse':
                # Sample trajectories with diverse characteristics
                other_trajectories = self._sample_diverse_trajectories(other_trajectories, max_trajectories_per_cluster)
            else:  # 'random'
                # Random sampling
                np.random.seed(42)
                if len(other_trajectories) > max_trajectories_per_cluster:
                    other_trajectories = np.random.choice(other_trajectories, max_trajectories_per_cluster, replace=False).tolist()
            
            # Plot non-excavating trajectories first (in light colors)
            n_to_plot = len(other_trajectories)
            for i, traj in enumerate(other_trajectories):
                traj = traj.sort_values('frame_number')
                coords = traj[['center_x', 'center_y']].values
                
                ax.plot(coords[:, 0], coords[:, 1], 
                       color='k', alpha=1, linewidth=1)
            
            # Plot excavating trajectories with special highlighting
            for i, traj in enumerate(excavating_trajectories):
                traj = traj.sort_values('frame_number')
                coords = traj[['center_x', 'center_y']].values
                ant_id = traj['ant_id'].iloc[0]
                
                # Use bright, distinct colors for excavating ants
                excavating_colors = {
                    141690: (1, 0, 0),      # Red
                }
                
                color = excavating_colors.get(ant_id, (1, 0, 0))
                
                # Draw thick, bright trajectory
                ax.plot(coords[:, 0], coords[:, 1], 
                       color=color, alpha=0.9, linewidth=4)
                
                # Mark start and end points with special markers
                ax.scatter(coords[0, 0], coords[0, 1], 
                         color=color, s=100, marker='o', 
                         edgecolor='black', linewidth=2, zorder=5)
                ax.scatter(coords[-1, 0], coords[-1, 1], 
                         color=color, s=100, marker='s', 
                         edgecolor='black', linewidth=2, zorder=5)
            
            # Add title with accurate trajectory counts
            total_trajectories = len(cluster_trajectories)
            plotted_trajectories = min(max_trajectories_per_cluster, len(other_trajectories)) + len(excavating_trajectories)
            
            title = f'Behavior Cluster {cluster_id}\n({total_trajectories} total, {plotted_trajectories} plotted)'
            if excavating_trajectories:
                title += f'\n🎯 {len(excavating_trajectories)} excavating ants'
            if plotted_trajectories < total_trajectories:
                title += f'\n⚠️ Showing {plotted_trajectories}/{total_trajectories} trajectories'
            ax.set_title(title)
            
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # Set consistent axis limits for all subplots
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
        
        # Hide unused subplots
        for i in range(n_clusters, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Behavior Clusters with Excavating Ants Highlighted', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    
    
    def save_results(self, output_dir='/home/tarun/Desktop/plots_for_committee_meeting/trajectory_clustering/', 
                     save_visualization_data=True):
        """
        Save comprehensive clustering results including features and metadata.
        
        Parameters
        ----------
        output_dir : str
            Output directory for saving results
        save_visualization_data : bool
            Whether to also save visualization data for quick loading
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save visualization data if requested
        if save_visualization_data:
            self.save_visualization_data(output_dir)
        
        # Save comprehensive results with features and metadata
        if self.cluster_labels is not None and self.trajectory_features is not None:
            print("💾 Saving comprehensive trajectory data...")
            
            # Create comprehensive dataframe
            comprehensive_data = []
            
            for i, traj in enumerate(self.trajectories):
                if i < len(self.cluster_labels) and i < len(self.trajectory_features):
                    # Extract metadata
                    ant_id = traj['ant_id'].iloc[0]
                    video_id = traj['video_id'].iloc[0]
                    hour = traj['hour'].iloc[0] if 'hour' in traj.columns else None
                    day = traj['day'].iloc[0] if 'day' in traj.columns else None
                    date = traj['date'].iloc[0] if 'date' in traj.columns else None
                    
                    # Extract trajectory coordinates
                    coords = traj[['center_x', 'center_y']].values
                    trajectory_length = len(coords)
                    
                    # Get features
                    features = self.trajectory_features[i]
                    
                    # Get cluster label
                    cluster_label = self.cluster_labels[i]
                    
                    # Calculate nest proximity
                    nest_proximity = self._calculate_nest_proximity(coords)
                    
                    comprehensive_data.append({
                        'trajectory_index': i,
                        'ant_id': ant_id,
                        'video_id': video_id,
                        'hour': hour,
                        'day': day,
                        'date': date,
                        'trajectory_length': trajectory_length,
                        'cluster_label': cluster_label,
                        'efficiency': features[0],
                        'straight_distance': features[1],
                        'avg_angle_change': features[2],
                        'total_length': features[3],
                        'nest_proximity': nest_proximity
                    })
            
            # Create and save comprehensive dataframe
            comprehensive_df = pd.DataFrame(comprehensive_data)
            
            # Save as compressed CSV for efficiency
            output_file = os.path.join(output_dir, 'comprehensive_trajectory_analysis.csv.gz')
            comprehensive_df.to_csv(output_file, index=False, compression='gzip')
            
            print(f"✅ Comprehensive data saved to {output_file}")
            print(f"   Records: {len(comprehensive_df):,}")
            print(f"   File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
            
            # Save summary statistics
            summary_stats = {
                'total_trajectories': len(comprehensive_df),
                'unique_ants': comprehensive_df['ant_id'].nunique(),
                'unique_videos': comprehensive_df['video_id'].nunique(),
                'date_range': f"{comprehensive_df['date'].min()} to {comprehensive_df['date'].max()}" if 'date' in comprehensive_df.columns else "N/A",
                'cluster_distribution': comprehensive_df['cluster_label'].value_counts().to_dict(),
                'avg_trajectory_length': comprehensive_df['trajectory_length'].mean(),
                'feature_columns': ['efficiency', 'straight_distance', 'avg_angle_change', 'total_length']
            }
            
            # Save summary as JSON
            import json
            summary_file = os.path.join(output_dir, 'analysis_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(summary_stats, f, indent=2, default=str)
            
            print(f"📊 Summary statistics saved to {summary_file}")
            
            # Save cluster-specific data
            for cluster_id in comprehensive_df['cluster_label'].unique():
                cluster_data = comprehensive_df[comprehensive_df['cluster_label'] == cluster_id]
                cluster_file = os.path.join(output_dir, f'cluster_{cluster_id}_data.csv.gz')
                cluster_data.to_csv(cluster_file, index=False, compression='gzip')
            
            print(f"📁 Cluster-specific files saved (cluster_0_data.csv.gz, etc.)")
        
        # Save cluster labels (legacy format)
        if self.cluster_labels is not None:
            results_df = pd.DataFrame({
                'trajectory_index': range(len(self.cluster_labels)),
                'cluster_label': self.cluster_labels
            })
            results_df.to_csv(os.path.join(output_dir, 'behavior_clusters.csv'), index=False)
        
        
        
        # Save all ant IDs from cluster 0 with nest proximity to a text file
        # if self.cluster_labels is not None:
        #     cluster_0_ants_with_proximity = []
        #     for i, traj in enumerate(self.trajectories):
        #         if i < len(self.cluster_labels) and self.cluster_labels[i] == 0:
        #             ant_id = traj['ant_id'].iloc[0]
        #             # Calculate nest proximity for this trajectory
        #             coords = traj[['center_x', 'center_y']].values
        #             nest_proximity = self._calculate_nest_proximity(coords)
        #             cluster_0_ants_with_proximity.append((ant_id, nest_proximity))
            
        #     if cluster_0_ants_with_proximity:
        #         cluster_0_file = os.path.join(output_dir, 'cluster_0_ant_ids.txt')
        #         with open(cluster_0_file, 'w') as f:
        #             for ant_id, nest_proximity in cluster_0_ants_with_proximity:
        #                 f.write(f"{ant_id},{nest_proximity:.2f}\n")
        #         print(f"💾 Cluster 0 ant IDs with nest proximity saved to {cluster_0_file}")
        #         print(f"   Found {len(cluster_0_ants_with_proximity)} ants in cluster 0")
                
        #         # Show proximity statistics
        #         proximities = [prox for _, prox in cluster_0_ants_with_proximity]
        #         print(f"   Nest proximity range: {min(proximities):.2f} - {max(proximities):.2f}")
        #         print(f"   Average proximity: {np.mean(proximities):.2f}")
        #         near_nest_count = sum(1 for prox in proximities if prox < 50)
        #         print(f"   Ants near nest (<50 pixels): {near_nest_count}/{len(proximities)}")
        
        # Save trail information
        


def main_hourly():
    """
    Main function for hourly trajectory clustering analysis with detailed visualizations.
    """
    
    # Initialize clusterer with nest mask
    nest_mask_path = '/home/tarun/Desktop/masks/beer-tree-08-01-2024_to_08-10-2024_center_only.png'
    clusterer = TrajectoryClusterer(site_id=1, nest_mask_path=nest_mask_path)

    # clusterer.visualize_clusters(
    # load_from_file='/home/tarun/Desktop/plots_for_committee_meeting/trajectory_clustering/visualization_data.pkl'
    # )
    
    # Define target date and hour
    target_date = '2024-08-01'
    target_hour = 21
    #print(f"Analyzing trajectories for {target_date} at hour {target_hour}")
    
    num_days = 10
    target_date_period = pd.Period(target_date, freq='D')
    trajectories = clusterer.load_trajectory_data(
        target_date_period, num_days, direction_filter=None,
        smooth_trajectories=False, detect_id_swaps=False
    )


    # # Load hourly data
    # trajectories = clusterer.load_hourly_trajectory_data(
    #     target_date, target_hour, direction_filter=None,
    #     smooth_trajectories=False, detect_id_swaps=False
    # )
    
    if len(trajectories) == 0:
        print(f"❌ No trajectories found for {target_date} at hour {target_hour}")
        return
    
    print(f"✅ Loaded {len(trajectories)} trajectories")
    

    
    # 1. Extract features and cluster behaviors
    print("\n1. Extracting features and clustering behaviors...")

    features = clusterer.extract_trajectory_features(trajectories)
    
    
    
    # Debug feature separation
    print("\n🔍 Analyzing feature separation...")
    clusterer._analyze_feature_separation(features)
    
    # Test different clustering methods
    print("\n🔍 Testing different clustering methods...")
    methods_to_test = ['kmeans']
    
    
    best_method = None
    best_score = -1
    
    for method in methods_to_test:
        print(f"\n--- Testing {method.upper()} ---")
        try:
            if method == 'kmeans':
                labels = clusterer.cluster_behaviors(method=method, n_clusters=None, use_scaling=True)
            
            
            # Evaluate clustering quality
            if labels is not None:
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                noise_ratio = n_noise / len(labels)
                
                print(f"   Result: {n_clusters} clusters, {n_noise} noise ({noise_ratio:.1%})")
                
                # Score based on having multiple clusters and reasonable noise
                if n_clusters > 1 and noise_ratio < 0.8:
                    score = n_clusters * (1 - noise_ratio)
                    if score > best_score:
                        best_score = score
                        best_method = method
                        behavior_labels = labels
                        print(f"   ✅ New best method! Score: {score:.2f}")
                    else:
                        print(f"   Score: {score:.2f}")
                else:
                    print(f"   ❌ Poor clustering (too few clusters or too much noise)")
                    
        except Exception as e:
            print(f"   ❌ Error with {method}: {e}")
    
    
        
        # Visualize behavior clusters in detail
        print("\n2. Visualizing behavior clusters...")
        clusterer.visualize_clusters(max_trajectories_per_cluster=100, sampling_method='diverse')
    
    
    
        # # 7. Save results
    print("\n7. Saving results...")
    clusterer.save_results()  # This will also save visualization data by default
    
    print("\n🎉 Hourly analysis complete!")
    print(f"\n📊 Summary for {target_date} hour {target_hour}:")
    print(f"   - Trajectories: {len(trajectories)}")
    if behavior_labels is not None:
        print(f"   - Behavior clusters: {len(np.unique(behavior_labels))}")
    
    # Example: To visualize clusters later without re-running clustering:
    # clusterer = TrajectoryClusterer(site_id=1)
    # clusterer.visualize_clusters(load_from_file='/home/tarun/Desktop/plots_for_committee_meeting/trajectory_clustering/visualization_data.pkl')
    
    


if __name__ == "__main__":
    main_hourly()
