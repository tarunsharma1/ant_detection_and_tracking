'''
Created : Mar 12 2024.
Code for plotting and displaying tracked ant ids on a video. This is to compare how the standard SORT tracking visually compares to tracking + interpolation 
'''

import cv2
import numpy as np
import pandas as pd


def interpolate_tracks(df, ant_id):
    """
    Interpolates missing bounding box positions for a given track.
    """
    track_df = df[df["ant_id"] == ant_id].copy()
    
    # Set frame as index
    track_df = track_df.set_index("frame_number")

    # Define interpolation method (linear, cubic, etc.)
    track_df_interp = track_df.reindex(range(track_df.index.min(), track_df.index.max() + 1))
    track_df_interp[["x1", "y1", "x2", "y2"]] = track_df_interp[["x1", "y1", "x2", "y2"]].interpolate(method="linear")

    return track_df_interp.reset_index()



def interpolate(df):
	df["frame_diff"] = df.groupby("ant_id")["frame_number"].diff()
	df_interp = pd.concat([interpolate_tracks(df, ant_id) for ant_id in df["ant_id"].unique()], ignore_index=True)
	return df_interp



path = '/media/tarun/Backup5TB/all_ant_data/beer-tree-08-01-2024_to_08-10-2024/2024-08-01_19_01_00'
video = path + '/' + path.split('/')[-1] + '.mp4'

tracking_csv = path + '/' + path.split('/')[-1] + '_yolo_tracking_with_direction.csv'

cap = cv2.VideoCapture(video)

df = pd.read_csv(tracking_csv)
#df = interpolate(df)

vid_out = cv2.VideoWriter('/home/tarun/Desktop/plots_for_committee_meeting/beer-tracking-direction-2024-08-01_19_01_00.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (1920,1088))


frame_number = 0

while True:
	ret, frame = cap.read()
	if not ret:
		break

	
	## resize to the shape that yolo is detecting on 
	frame = cv2.resize(frame, (1920, 1088))

	df_frame = df.loc[df.frame_number == frame_number]

	for index, row in df_frame.iterrows():
	    x1,y1,x2,y2, ant_id, direction = row['x1'], row['y1'], row['x2'], row['y2'], row['ant_id'], row['direction']
	    #cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
	    if direction == 'away':
	    	cv2.putText(frame, str(ant_id), (int((x1+x2)/2), int((y1+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
	    else:
	    	cv2.putText(frame, str(ant_id), (int((x1+x2)/2), int((y1+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)


	vid_out.write(frame)

	cv2.imshow('Frame',frame)
	cv2.waitKey(30)
	frame_number += 1

vid_out.release()
cap.release()
