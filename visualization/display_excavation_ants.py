#!/usr/bin/env python3
"""
Simple display tool to show video frames with ant IDs overlaid for excavation identification.

This script displays video frames with ant IDs overlaid using OpenCV imshow,
allowing you to visually identify which ants are performing excavation behavior.

Usage:
    python display_excavation_ants.py
"""

import sys
import os
sys.path.append('../')
import pandas as pd
import numpy as np
import cv2
from mysql_dataset import database_helper

def load_video_and_tracking_data(site_id, target_date, target_hour):
    """
    Load video and tracking data for a specific hour.
    """
    print(f"Loading data for site {site_id}, {target_date} at hour {target_hour}...")
    
    # Connect to database
    connection = database_helper.create_connection("localhost", "root", "master", "ant_colony_db")
    cursor = connection.cursor()
    
    query = """
    SELECT Counts.video_id, Counts.herdnet_tracking_with_direction_closest_boundary_method_csv,
    Videos.temperature, Videos.humidity, Videos.LUX, Videos.time_stamp, Videos.site_id
    FROM Counts INNER JOIN Videos ON Counts.video_id=Videos.video_id
    WHERE Videos.site_id = %s
    """
    cursor.execute(query, (site_id,))
    table_rows = cursor.fetchall()
    
    df = pd.DataFrame(table_rows, columns=cursor.column_names)
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    
    # Filter by site, date, and hour
    df_filtered = df.loc[
        (df.site_id == site_id) &
        (df.time_stamp.dt.date == pd.to_datetime(target_date).date()) &
        (df.time_stamp.dt.hour == target_hour)
    ]
    
    if len(df_filtered) == 0:
        print(f"No videos found for site {site_id} on {target_date} at hour {target_hour}")
        return None, None
    
    # Get the first video
    video_info = df_filtered.iloc[0]
    tracking_csv = video_info['herdnet_tracking_with_direction_closest_boundary_method_csv']
    video_path = video_info['video_id']  # video_id is the full path
    
    print(f"Found video: {video_path}")
    print(f"Tracking data: {tracking_csv}")
    
    # Load tracking data
    try:
        tracking_data = pd.read_csv(tracking_csv)
        print(f"Loaded {len(tracking_data)} tracking records")
        return video_path, tracking_data
    except Exception as e:
        print(f"Error loading tracking data: {e}")
        return None, None

def display_frames_with_ant_ids(video_path, tracking_data, excavating_ant_ids, frame_interval=30, save_crops=True, crop_output_dir='ant_crops'):
    """
    Display video frames with ant IDs overlaid, highlighting excavating ants and their trajectories.
    Optionally save crops of ants during first 20% of their trajectories for visual confirmation.
    
    Parameters
    ----------
    video_path : str
        Path to video file
    tracking_data : pd.DataFrame
        Tracking data with ant_id, frame_number, x1, y1, x2, y2
    excavating_ant_ids : list
        List of ant IDs that are excavating
    frame_interval : int
        Interval between frames to display
    save_crops : bool
        Whether to save crops of ants for visual confirmation
    crop_output_dir : str
        Directory to save crop images
    """
    print(f"Displaying frames from: {video_path}")
    print(f"Frame interval: {frame_interval}")
    print(f"Excavating ants: {excavating_ant_ids}")
    print("Press 'q' to quit, 'n' for next frame, 'p' for previous frame")
    
    # Create output directory for crops if saving
    if save_crops:
        import os
        os.makedirs(crop_output_dir, exist_ok=True)
        print(f"📁 Crop images will be saved to: {crop_output_dir}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} frames at {fps:.1f} FPS")
    
    # Initialize trajectory tracking for excavating ants
    excavating_trajectories = {ant_id: [] for ant_id in excavating_ant_ids}
    
    # Track trajectory lengths for crop saving (all frames)
    trajectory_lengths = {ant_id: 0 for ant_id in excavating_ant_ids}
    frames_saved_per_ant = {ant_id: 0 for ant_id in excavating_ant_ids}
    
    current_frame = 0
    
    while cap.isOpened():
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        
        if not ret:
            print("End of video reached")
            break
        
        # Get tracking data for this frame
        frame_data = tracking_data[tracking_data['frame_number'] == current_frame]
        
        if len(frame_data) > 0:
            # Create a copy of the frame for drawing
            display_frame = frame.copy()
            
            # Get unique ants in this frame
            unique_ants = frame_data['ant_id'].unique()
            
            # Draw all ants first (in gray) - just trajectories, no boxes
            for ant_id in unique_ants:
                ant_data = frame_data[frame_data['ant_id'] == ant_id]
                
                if len(ant_data) > 0:
                    # Get bounding box coordinates
                    x1 = int(ant_data['x1'].iloc[0])
                    y1 = int(ant_data['y1'].iloc[0])
                    x2 = int(ant_data['x2'].iloc[0])
                    y2 = int(ant_data['y2'].iloc[0])
                    
                    # Calculate center
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # For non-excavating ants, just add to trajectory (no visual drawing)
                    if ant_id not in excavating_ant_ids:
                        # We'll handle trajectory drawing for all ants later
                        pass
            
            # Generate colors for cluster 1 ants dynamically
            def generate_color(index):
                """Generate distinct colors for each ant."""
                colors = [
                    (0, 0, 255),      # Red
                    (0, 255, 0),      # Green
                    (255, 0, 0),      # Blue
                    (0, 255, 255),    # Yellow
                    (255, 0, 255),    # Magenta
                    (255, 255, 0),    # Cyan
                    (255, 128, 0),    # Orange
                    (128, 0, 255),    # Purple
                    (0, 128, 255),    # Light Blue
                    (255, 192, 203), # Pink
                    (128, 255, 0),    # Lime
                    (255, 0, 128),    # Hot Pink
                ]
                return colors[index % len(colors)]
            
            # Create color mapping for cluster 1 ants
            cluster_1_colors = {}
            for i, ant_id in enumerate(excavating_ant_ids):
                cluster_1_colors[ant_id] = generate_color(i)
            
            for ant_id in excavating_ant_ids:
                if ant_id in unique_ants:
                    ant_data = frame_data[frame_data['ant_id'] == ant_id]
                    
                    if len(ant_data) > 0:
                        # Get bounding box coordinates
                        x1 = int(ant_data['x1'].iloc[0])
                        y1 = int(ant_data['y1'].iloc[0])
                        x2 = int(ant_data['x2'].iloc[0])
                        y2 = int(ant_data['y2'].iloc[0])
                        
                        # Calculate center
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        # Add to trajectory
                        excavating_trajectories[ant_id].append((center_x, center_y))
                        trajectory_lengths[ant_id] = len(excavating_trajectories[ant_id])
                        
                        # Save crop for all frames of trajectory for visual confirmation
                        if save_crops:
                            # Get bounding box with some padding
                            padding = 10
                            x1_padded = max(0, x1 - padding)
                            y1_padded = max(0, y1 - padding)
                            x2_padded = min(frame.shape[1], x2 + padding)
                            y2_padded = min(frame.shape[0], y2 + padding)
                            
                            # Crop the ant
                            ant_crop = frame[y1_padded:y2_padded, x1_padded:x2_padded]
                            
                            if ant_crop.size > 0:
                                # Create subfolder for this ant
                                ant_folder = os.path.join(crop_output_dir, f"ant_{ant_id}")
                                os.makedirs(ant_folder, exist_ok=True)
                                
                                # Save crop image
                                crop_filename = f"frame_{current_frame:04d}_crop.png"
                                crop_path = os.path.join(ant_folder, crop_filename)
                                cv2.imwrite(crop_path, ant_crop)
                                
                                # Increment counter for this ant
                                frames_saved_per_ant[ant_id] += 1
                                
                                # Only print progress every 50 frames to avoid spam
                                if frames_saved_per_ant[ant_id] % 50 == 0 or frames_saved_per_ant[ant_id] <= 5:
                                    print(f"📸 Saved crop: ant_{ant_id}/frame_{current_frame:04d}_crop.png (total: {frames_saved_per_ant[ant_id]})")
                        
                        # Draw trajectory only (no bounding box)
                        color = cluster_1_colors.get(ant_id, (255, 255, 255))
                        if len(excavating_trajectories[ant_id]) > 1:
                            for i in range(1, len(excavating_trajectories[ant_id])):
                                pt1 = excavating_trajectories[ant_id][i-1]
                                pt2 = excavating_trajectories[ant_id][i]
                                cv2.line(display_frame, pt1, pt2, color, 3)  # Thicker lines for visibility
            
            # Add frame info
            cv2.putText(display_frame, f'Frame: {current_frame}/{total_frames}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(display_frame, f'Total Ants: {len(unique_ants)}', 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Count cluster 0 ants in this frame
            cluster_0_in_frame = sum(1 for ant_id in excavating_ant_ids if ant_id in unique_ants)
            cv2.putText(display_frame, f'Cluster 0 (Near Nest): {cluster_0_in_frame}', 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Cluster 0 Ants Near Nest with Trajectories', display_frame)
            
            print(f"Frame {current_frame}: {len(unique_ants)} total ants, {cluster_0_in_frame} cluster 0 (near nest)")
        
        # Handle key presses
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break
        
        current_frame += 1
        if current_frame >= total_frames:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print trajectory summary
    print("\n📊 Trajectory Summary for Cluster 0 Ants (Near Nest):")
    for ant_id in excavating_ant_ids:
        trajectory_length = len(excavating_trajectories[ant_id])
        print(f"  Ant {ant_id}: {trajectory_length} trajectory points")

def load_cluster_0_ant_ids_with_proximity(cluster_file_path, proximity_threshold=50):
    """
    Load ant IDs from cluster 0 with nest proximity filtering.
    
    Parameters
    ----------
    cluster_file_path : str
        Path to cluster 0 file with ant_id,nest_proximity format
    proximity_threshold : float
        Maximum nest proximity to include (pixels)
    
    Returns
    -------
    list
        List of ant IDs that are in cluster 0 and near the nest
    """
    try:
        with open(cluster_file_path, 'r') as f:
            ant_ids = []
            all_ants = []
            near_nest_ants = []
            
            for line in f.readlines():
                if line.strip():
                    try:
                        parts = line.strip().split(',')
                        if len(parts) == 2:
                            ant_id = int(float(parts[0]))
                            nest_proximity = float(parts[1])
                            
                            all_ants.append((ant_id, nest_proximity))
                            
                            # Filter by nest proximity threshold
                            if nest_proximity < proximity_threshold:
                                ant_ids.append(ant_id)
                                near_nest_ants.append((ant_id, nest_proximity))
                        else:
                            print(f"⚠️ Skipping malformed line: {line.strip()}")
                            continue
                    except ValueError as e:
                        print(f"⚠️ Skipping invalid line: {line.strip()} - {e}")
                        continue
        
        print(f"✅ Loaded cluster 0 data:")
        print(f"   Total ants in cluster 0: {len(all_ants)}")
        print(f"   Ants near nest (<{proximity_threshold} pixels): {len(near_nest_ants)}")
        
        if all_ants:
            all_proximities = [prox for _, prox in all_ants]
            near_proximities = [prox for _, prox in near_nest_ants]
            
            print(f"   All ants proximity range: {min(all_proximities):.2f} - {max(all_proximities):.2f}")
            if near_nest_ants:
                print(f"   Near nest proximity range: {min(near_proximities):.2f} - {max(near_proximities):.2f}")
                print(f"   Average near nest proximity: {np.mean(near_proximities):.2f}")
        
        return ant_ids
        
    except FileNotFoundError:
        print(f"❌ Cluster file not found: {cluster_file_path}")
        return []
    except Exception as e:
        print(f"❌ Error loading cluster file: {e}")
        return []

def main():
    """
    Main function to display video frames with cluster 0 ants near nest.
    """
    print("🔍 CLUSTER 0 ANT TRAJECTORY VISUALIZATION (NEAR NEST)")
    print("=" * 60)
    
    # Configuration
    site_id = 1
    target_date = '2024-08-02'
    target_hour = 13
    proximity_threshold = 100  # pixels
    
    # Path to cluster 0 ant IDs file
    cluster_file_path = '/home/tarun/Desktop/plots_for_committee_meeting/trajectory_clustering/cluster_0_ant_ids.txt'
    
    print(f"Loading cluster 0 ants near nest for site {site_id} on {target_date} at hour {target_hour}")
    print(f"Nest proximity threshold: {proximity_threshold} pixels")
    
    # Load cluster 0 ant IDs with proximity filtering
    cluster_0_ant_ids = load_cluster_0_ant_ids_with_proximity(cluster_file_path, proximity_threshold)
    
    if not cluster_0_ant_ids:
        print("❌ No cluster 0 ants near nest found. Run clustering first.")
        return
    
    print(f"Cluster 0 ants near nest: {cluster_0_ant_ids}")
    
    # Load video and tracking data
    video_path, tracking_data = load_video_and_tracking_data(
        site_id, target_date, target_hour
    )
    
    if video_path is None or tracking_data is None:
        print("❌ Failed to load video or tracking data")
        return
    
    # Display frames with cluster 0 ant trajectories
    display_frames_with_ant_ids(
        video_path, tracking_data, cluster_0_ant_ids,
        frame_interval=30,  # Show every 30th frame
        save_crops=False,    # Save crops for visual confirmation
        crop_output_dir='cluster_0_ant_crops'  # Output directory for crops
    )
    
    print("\n🎉 Display complete!")
    print("\nTrajectory visualization shows:")
    print("1. Cluster 0 ants near nest (<100 pixels) highlighted in bright colors")
    print("2. Their movement trajectories drawn as colored lines")
    print("3. Real-time trajectory building as video plays")
    print("4. Crop images saved for visual confirmation of excavation behavior")
    print(f"5. Check 'cluster_0_ant_crops' folder for individual ant subfolders")
    print("6. Each ant has its own subfolder with ALL frames of trajectory")
    
    # Print summary of saved crops
    if save_crops:
        print(f"\n📁 Crop Summary:")
        for ant_id in excavating_ant_ids:
            frames_saved = frames_saved_per_ant.get(ant_id, 0)
            print(f"   Ant {ant_id}: {frames_saved} frames saved")

if __name__ == "__main__":
    main()
