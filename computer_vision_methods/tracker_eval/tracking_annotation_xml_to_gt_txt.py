'''

This is to create ground truth text files for our tracking annotations so that we can use the TrackEval library to obtain MOT metrics.

We have point annotations from CVAT in CVAT_video export type (XML) and converts the xml data to the MOT Challenge ground truth format 
frame_number, id, x1, y1, w, h, conf, x,y,z 
where conf, x,y,z are all -1.



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
import read_xml_files_from_cvat, converting_points_to_boxes

image_width, image_height = 1920,1080
#image_width, image_height = 1920/3,1080/3 ## for patches




def convert_to_csv(annotation_xml_file, folder_where_annotated_video_came_from, box_size, label):
	gts_points_dict = read_xml_files_from_cvat.xml_to_dict_cvat_for_videos(annotation_xml_file)
	dictionary_of_boxes = {}
	for f in gts_points_dict.keys():
		dictionary_of_boxes[f] = []
		dictionary_of_boxes[f] = converting_points_to_boxes.convert_points_to_boxes(gts_points_dict[f], box_size, 1920, 1080)
	
	f = open('/home/tarun/Desktop/TrackEval/data/gt/mot_challenge/ant_tracks-val/' + label + '/gt/gt.txt', 'w')
    
	for frame in dictionary_of_boxes:
		boxes = dictionary_of_boxes[frame]
		for b in boxes:
			ant_id = b[0]
			x1 = b[1]
			y1 = b[2]
			w = b[3] - b[1]
			h = b[4] - b[2]
			## frame_number is 1 indexed
			f.write(str(1 + frame) + ',' + str(ant_id) + ',' + str(x1) + ',' + str(y1) + ',' + str(w) + ',' + str(h) + ',-1,-1,-1,-1' + '\n')
			
	f.close()




### Train

#convert_to_yolo_labels('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/shack/2024-08-23_14_01_01_patches.xml', '/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-22-2024_to_09-11-2024/2024-08-23_14_01_01/', 20, '2024-08-23_14_01_01')
#convert_to_yolo_labels('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/shack/2024-08-01_20_01_00_patches.xml', '/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-22-2024/2024-08-01_20_01_00/', 20, '2024-08-01_20_01_00')
#convert_to_yolo_labels('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/beer/2024-08-01_19_01_00_patches.xml', '/media/tarun/Backup5TB/all_ant_data/beer-tree-08-01-2024_to_08-30-2024/2024-08-01_19_01_00/', 20, '2024-08-01_19_01_00')
#convert_to_yolo_labels('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/rain/2024-10-04_00_01_06_patches.xml', '/media/tarun/Backup5TB/all_ant_data/rain-tree-10-03-2024_to_10-25-2024/2024-10-04_00_01_06/', 20, '2024-10-04_00_01_06')

#convert_to_yolo_labels('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/rain/2024-09-01_11_01_00.xml', '/media/tarun/Backup5TB/all_ant_data/rain-tree-08-22-2024_to_09-02-2024/2024-09-01_11_01_00/', 20, '2024-09-01_11_01_00')
#convert_to_yolo_labels('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/beer/2024-08-03_13_01_01.xml', '/media/tarun/Backup5TB/all_ant_data/beer-tree-08-01-2024_to_08-30-2024/2024-08-03_13_01_01/', 20, '2024-08-03_13_01_01')




### Val
convert_to_csv('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/shack/2024-08-13_11_01_01_tracking.xml', '/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-22-2024/2024-08-13_11_01_01/', 20, '2024-08-13_11_01_01')
convert_to_csv('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/beer/2024-10-27_23_01_01_tracking.xml', '/media/tarun/Backup5TB/all_ant_data/beer-10-22-2024_to_10-28-2024/2024-10-27_23_01_01/', 20, '2024-10-27_23_01_01')
convert_to_csv('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/rain/2024-10-09_23_01_00_tracking.xml', '/media/tarun/Backup5TB/all_ant_data/rain-tree-10-03-2024_to_10-25-2024/2024-10-09_23_01_00/', 20, '2024-10-09_23_01_00')

### Test
#convert_to_yolo_labels('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/shack/2024-08-22_03_01_01_patches.xml', '/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-22-2024/2024-08-22_03_01_01/', 20, '2024-08-22_03_01_01')
#convert_to_yolo_labels('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/rain/2024-08-22_21_01_00_patches.xml', '/media/tarun/Backup5TB/all_ant_data/rain-tree-08-22-2024_to_09-02-2024/2024-08-22_21_01_00/', 20, '2024-08-22_21_01_00')
