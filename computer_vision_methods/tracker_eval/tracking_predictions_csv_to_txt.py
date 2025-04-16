'''

script to read and convert output csvs from sort tracking to the .txt file format required by TrackEval
frame_number, ant_id, x,y,w,h,conf,x,y,z
where conf,x,y,z are -1,-1,-1,-1.


for the ground truth xmls from CVAT, frame_number always starts from 0 in the xml (but needs to be 1 indexed for TrackEval), but for
the predictions, I have tracking predictions on all frames. I only want to keep the ones annotated (start_frame - start_frame + 30) and subtract (start_frame-1)
so that they start from 1 to compare with the gt.

'''

import sys
sys.path.append('../../')
sys.path.append('../')
sys.path.append('../../utils')
sys.path.append('../../preprocessing_for_annotation')
sys.path.append('../../visualization')
import shutil
import cv2
import os
import pandas as pd

image_width, image_height = 1920,1080


def convert_prediction_csv_to_trackeval_txt(folder_where_annotated_video_came_from, box_size, label, start_frame):
	df = pd.read_csv(folder_where_annotated_video_came_from + label + '_herdnet_tracking_with_direction_and_angle.csv')
	## only keep the ones we have annotations for
	df = df.loc[(df.frame_number >= start_frame) & (df.frame_number < start_frame+30)]

	f = open('/home/tarun/Desktop/TrackEval/data/trackers/mot_challenge/ant_tracks-val/SORT-herdnet/data/' + label + '.txt', 'w')

	
	for index, row in df.iterrows():
		## convert start_frame to start_frame+30 to 1-30
		frame = row['frame_number'] - (start_frame-1)
		ant_id = row['ant_id']
		x1 = row['x1']
		y1 = row['y1']
		w = row['x2'] - row['x1']
		h = row['y2'] - row['y1']

		f.write(str(frame) + ',' + str(ant_id) + ',' + str(x1) + ',' + str(y1) + ',' + str(w) + ',' + str(h) + ',-1,-1,-1,-1' + '\n')

	f.close()






### Train

#convert_to_yolo_labels('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/shack/2024-08-23_14_01_01_patches.xml', '/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-22-2024_to_09-11-2024/2024-08-23_14_01_01/', 20, '2024-08-23_14_01_01')
#convert_to_yolo_labels('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/shack/2024-08-01_20_01_00_patches.xml', '/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-22-2024/2024-08-01_20_01_00/', 20, '2024-08-01_20_01_00')
#convert_to_yolo_labels('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/beer/2024-08-01_19_01_00_patches.xml', '/media/tarun/Backup5TB/all_ant_data/beer-tree-08-01-2024_to_08-30-2024/2024-08-01_19_01_00/', 20, '2024-08-01_19_01_00')
#convert_to_yolo_labels('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/rain/2024-10-04_00_01_06_patches.xml', '/media/tarun/Backup5TB/all_ant_data/rain-tree-10-03-2024_to_10-25-2024/2024-10-04_00_01_06/', 20, '2024-10-04_00_01_06')

#convert_to_yolo_labels('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/rain/2024-09-01_11_01_00.xml', '/media/tarun/Backup5TB/all_ant_data/rain-tree-08-22-2024_to_09-02-2024/2024-09-01_11_01_00/', 20, '2024-09-01_11_01_00')
#convert_to_yolo_labels('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/beer/2024-08-03_13_01_01.xml', '/media/tarun/Backup5TB/all_ant_data/beer-tree-08-01-2024_to_08-30-2024/2024-08-03_13_01_01/', 20, '2024-08-03_13_01_01')




### Val
convert_prediction_csv_to_trackeval_txt('/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-26-2024/2024-08-13_11_01_01/', 20, '2024-08-13_11_01_01', 500)
convert_prediction_csv_to_trackeval_txt('/media/tarun/Backup5TB/all_ant_data/beer-10-22-2024_to_11-02-2024/2024-10-27_23_01_01/', 20, '2024-10-27_23_01_01', 200)
convert_prediction_csv_to_trackeval_txt('/media/tarun/Backup5TB/all_ant_data/rain-tree-10-03-2024_to_10-19-2024/2024-10-09_23_01_00/', 20, '2024-10-09_23_01_00', 250)

### Test
#convert_to_yolo_labels('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/shack/2024-08-22_03_01_01_patches.xml', '/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-22-2024/2024-08-22_03_01_01/', 20, '2024-08-22_03_01_01')
#convert_to_yolo_labels('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/rain/2024-08-22_21_01_00_patches.xml', '/media/tarun/Backup5TB/all_ant_data/rain-tree-08-22-2024_to_09-02-2024/2024-08-22_21_01_00/', 20, '2024-08-22_21_01_00')
