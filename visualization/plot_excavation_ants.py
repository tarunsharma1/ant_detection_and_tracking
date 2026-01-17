#!/usr/bin/env python3
"""
Visualization tool to identify excavating ants by plotting ant IDs on video frames.

This script helps identify which ant IDs are performing excavation behavior
by overlaying ant IDs on the raw video frames, allowing manual annotation
of excavating vs non-excavating ants.

Usage:
    python plot_excavation_ants.py
"""

import sys
import os
sys.path.append('../')
import pandas as pd
import numpy as np
import cv2

def load_video_and_tracking_data(site_id, target_date, target_hour):
    """
    Load video and tracking data for a specific hour.
    
    Parameters
    ----------
    site_id : int
        Site ID to analyze
    target_date : str
        Target date (e.g., '2024-08-02')
    target_hour : int
        Target hour (0-23)
    
    Returns
    -------
    tuple
        (video_path, tracking_data, video_info)
    """
    print(f"Loading data for site {site_id}, {target_date} at hour {target_hour}...")
    
    # Connect to database
    from mysql_dataset import database_helper
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
        return None, None, None
    
    # Get the first video (you can modify this to select specific videos)
    video_info = df_filtered.iloc[0]
    tracking_csv = video_info['herdnet_tracking_with_direction_closest_boundary_method_csv']
    
    # video_id is the full path to the video
    video_path = video_info['video_id']
    
    print(f"Found video: {video_path}")
    print(f"Tracking data: {tracking_csv}")
    
    # Load tracking data
    try:
        tracking_data = pd.read_csv(tracking_csv)
        print(f"Loaded {len(tracking_data)} tracking records")
        return video_path, tracking_data, video_info
    except Exception as e:
        print(f"Error loading tracking data: {e}")
        return None, None, None

def load_high_score_ant_ids(predictions_csv_path, target_video_id, score_threshold=0.9):
    """
    Load ant IDs with classifier scores above a threshold for a specific video.
    
    Parameters
    ----------
    predictions_csv_path : str
        Path to the predictions CSV file containing classifier scores
    target_video_id : str
        Video ID to filter for
    score_threshold : float
        Minimum excavation score to include an ant
    
    Returns
    -------
    set
        Set of ant IDs that exceed the threshold in the target video
    """
    print(f"Loading classifier scores from: {predictions_csv_path}")
    
    try:
        predictions_df = pd.read_csv(predictions_csv_path)
        print(f"Loaded {len(predictions_df)} total trajectories from predictions file")
        
        if 'video_id' not in predictions_df.columns or 'ant_id' not in predictions_df.columns:
            print("❌ Predictions file missing required columns 'video_id' and/or 'ant_id'")
            return set()
        
        # Determine score column (default expected name)
        if 'excavating_score' in predictions_df.columns:
            score_column = 'excavating_score'
        else:
            # Use the last column as a fallback
            score_column = predictions_df.columns[-1]
            print(f"⚠️ Using '{score_column}' as score column (last column in file)")
        
        # Filter for the specific video
        predictions_df['video_id'] = predictions_df['video_id'].astype(str)
        filtered_df = predictions_df[predictions_df['video_id'] == str(target_video_id)]
        print(f"Found {len(filtered_df)} trajectories for video: {target_video_id}")
        
        if len(filtered_df) == 0:
            return set()
        
        # Filter by score threshold
        high_score_df = filtered_df[filtered_df[score_column] >= score_threshold]
        print(f"Found {len(high_score_df)} trajectories with score >= {score_threshold}")
        
        ant_ids = set(high_score_df['ant_id'].astype(int).unique())
        print(f"Unique high-score ant IDs: {len(ant_ids)}")
        print(f"Ant IDs: {sorted(ant_ids)}")
        
        return ant_ids
    except Exception as e:
        print(f"Error loading predictions data: {e}")
        return set()

def display_ant_ids_on_frames(
    video_path,
    tracking_data,
    highlight_ant_ids=None,
    frame_interval=30,
    score_threshold=None,
    output_video_path=None,
):
    """
    Display ant IDs on video frames using OpenCV for visual identification.
    
    Parameters
    ----------
    video_path : str
        Path to video file
    tracking_data : pd.DataFrame
        Tracking data with ant_id, frame_number, x1, y1, x2, y2
    highlight_ant_ids : set, optional
        Set of ant IDs to highlight. If None, randomly select 50%.
    frame_interval : int
        Interval between frames to display
    """
    print(f"Displaying video: {video_path}")
    print("Press 'q' to quit, 'n' for next frame, 'p' for previous frame")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} frames at {fps:.1f} FPS")
    
    # Determine frame size (fallback to reading one frame if metadata missing)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_width == 0 or frame_height == 0:
        ret_probe, frame_probe = cap.read()
        if not ret_probe:
            print("❌ Could not read any frames from the video.")
            cap.release()
            return
        frame_height, frame_width = frame_probe.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print(f"Frame size: {frame_width}x{frame_height}")

    video_writer = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer_fps = fps if fps and not np.isnan(fps) and fps > 0 else 20.0
        video_writer = cv2.VideoWriter(output_video_path, fourcc, writer_fps, (frame_width, frame_height))
        if not video_writer.isOpened():
            print(f"⚠️ Could not open VideoWriter at {output_video_path}. Video will not be saved.")
            video_writer = None
        else:
            print(f"💾 Saving annotated video to {output_video_path}")
    
    # Determine which ant IDs to display
    tracking_data = tracking_data.copy()
    if 'ant_id' in tracking_data.columns:
        tracking_data['ant_id'] = tracking_data['ant_id'].astype(int)
    if 'frame_number' in tracking_data.columns:
        tracking_data['frame_number'] = tracking_data['frame_number'].astype(int)
    all_ant_ids = tracking_data['ant_id'].unique()
    
    if highlight_ant_ids is not None:
        # Use provided ant IDs (e.g., high-score ants)
        selected_ant_ids = {int(ant_id) for ant_id in highlight_ant_ids}
        if score_threshold is not None:
            selection_mode = f"Score ≥ {score_threshold:.2f}"
        else:
            selection_mode = "Selected Ants"
        print(f"Total ants in video: {len(all_ant_ids)}")
        print(f"Showing selected ants: {len(selected_ant_ids)} ants")
        print(f"Selected ant IDs: {sorted(selected_ant_ids)}")
    else:
        # Random 50% selection (original behavior)
        np.random.seed(42)
        selected_ant_ids = set(np.random.choice(all_ant_ids, size=len(all_ant_ids)//2, replace=False))
        selection_mode = "Random 50%"
        print(f"Total ants in video: {len(all_ant_ids)}")
        print(f"Showing random 50%: {len(selected_ant_ids)} ants")
        print(f"Selected ant IDs: {sorted(selected_ant_ids)}")
    
    current_frame = 0
    window_name = f'Ant IDs for Excavation Identification ({selection_mode})'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Keep aspect ratio but shrink to fit on screen (max 1280x720)
    max_width, max_height = 1280, 720
    scale = min(max_width / frame_width, max_height / frame_height, 1.0)
    display_width = int(frame_width * scale)
    display_height = int(frame_height * scale)
    cv2.resizeWindow(window_name, display_width, display_height)
    
    while cap.isOpened():
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        
        if not ret:
            print("End of video reached")
            break
        
        display_frame = frame.copy()
        
        if display_frame is None or display_frame.size == 0:
            print(f"⚠️ Empty frame encountered at index {current_frame}, skipping")
            current_frame += 1
            continue
        
        # Get tracking data for this frame
        frame_data = tracking_data[tracking_data['frame_number'] == current_frame]
        unique_ants = frame_data['ant_id'].unique() if len(frame_data) > 0 else []
        selected_ants_in_frame = []
        
        if len(frame_data) > 0:
            # Ensure consistent typing for ant IDs
            frame_data = frame_data.copy()
            frame_data['ant_id'] = frame_data['ant_id'].astype(int)
            unique_ants = frame_data['ant_id'].unique()
            selected_ants_in_frame = [ant_id for ant_id in unique_ants if ant_id in selected_ant_ids]
            
            # Use a single color for bounding boxes (red)
            box_color = (0, 0, 255)  # Red in BGR
            
            # Draw bounding boxes for each selected ant
            for ant_id in selected_ants_in_frame:
                ant_data = frame_data[frame_data['ant_id'] == ant_id]
                
                if len(ant_data) > 0:
                    # Get bounding box coordinates
                    x1 = int(round(ant_data['x1'].iloc[0]))
                    y1 = int(round(ant_data['y1'].iloc[0]))
                    x2 = int(round(ant_data['x2'].iloc[0]))
                    y2 = int(round(ant_data['y2'].iloc[0]))
                    
                    # Calculate center
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Draw bounding box around the ant
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
        
        # Add frame info overlay
        cv2.putText(display_frame, f'Frame: {current_frame}/{total_frames}',
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if len(frame_data) > 0:
            cv2.putText(display_frame, f'Showing: {len(selected_ants_in_frame)}/{len(unique_ants)} ants',
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            cv2.putText(display_frame, 'No tracking data in this frame',
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display the frame
        cv2.imshow(window_name, display_frame)

        # Save frame if requested
        if video_writer is not None:
            video_writer.write(display_frame)
        
        
        # Handle key presses
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('n'):
            current_frame = min(current_frame + frame_interval, total_frames - 1)
        elif key == ord('p'):
            current_frame = max(current_frame - frame_interval, 0)
        else:
            current_frame += 1
            if current_frame >= total_frames:
                break
    
    cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f"✅ Saved video to {output_video_path}")
    cv2.destroyAllWindows()
    print("Display complete!")
    


def main():
    """
    Main function to create excavation ant identification tool.
    """
    print("🔍 EXCAVATION ANT IDENTIFICATION TOOL")
    print("=" * 60)
    
    # Configuration
    site_id = 1
    target_date = '2024-08-05'
    target_hour = 20
    predictions_csv_path = '/home/tarun/Desktop/plots_for_committee_meeting/excavation_analysis/models/trajectory_classifier_predictions.csv'
    target_video_id = '/media/tarun/Backup5TB/all_ant_data/beer-tree-08-01-2024_to_08-10-2024/2024-08-05_20_01_02/2024-08-05_20_01_02.mp4'
    score_threshold = 0.8
    output_video_path = '/home/tarun/Desktop/plots_for_committee_meeting/excavation_analysis/models/2024-08-05_20_high_score_overlay.mp4'
    
    print(f"Analyzing site {site_id} on {target_date} at hour {target_hour}")
    
    # Load video and tracking data
    video_path, tracking_data, video_info = load_video_and_tracking_data(
        site_id, target_date, target_hour
    )
    
    if video_path is None or tracking_data is None:
        print("❌ Failed to load video or tracking data")
        return
    
    # Load high-score ant IDs for filtering
    print("\n📊 Loading high-score ant IDs from classifier...")
    high_score_ant_ids = load_high_score_ant_ids(
        predictions_csv_path, target_video_id, score_threshold=score_threshold
    )
    
    if len(high_score_ant_ids) == 0:
        print("⚠️  No high-score ant IDs found, will use random 50% selection")
        high_score_ant_ids = None
    
    # Display ant IDs on frames for visual identification
    print("\n📸 Displaying ant IDs on video frames...")
    print("Look for ants carrying bark pieces in their mouths!")
    print("Note down the ant IDs that appear to be excavating.")
    
    display_ant_ids_on_frames(
        video_path, tracking_data,
        highlight_ant_ids=high_score_ant_ids,
        frame_interval=30,
        score_threshold=score_threshold,
        output_video_path=output_video_path
    )
    
    
    

if __name__ == "__main__":
    main()
