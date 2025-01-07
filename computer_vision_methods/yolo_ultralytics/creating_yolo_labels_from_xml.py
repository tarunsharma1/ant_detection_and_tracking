'''

We have point annotations from CVAT in XML format, however yolo requires one text file per image with coords of bounding box.
This script converts given cvat annotation files to yolo label files


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
from evaluation_metrics import read_and_convert_ground_truth

#image_width, image_height = 1920,1080
image_width, image_height = 1920/3,1080/3 ## for patches

def convert_to_yolo_labels(annotation_xml_file, folder_where_annotated_video_came_from, box_size, label):
	dictionary_of_boxes = read_and_convert_ground_truth(annotation_xml_file, folder_where_annotated_video_came_from, box_size) ## key is full path of image name, value is list of lists of unnormalized boxes
	for img in dictionary_of_boxes:
		## copy over image from location to the yolo dataset training folder
		shutil.copy(img, '/home/tarun/Desktop/antcam/datasets/ants_manual_annotation_patches/images/val/')
		## create and open a text file with the same name as the image in the labels folder. 
		## Also convert from X,Y,X,Y to normalized X_center,Y_center,W,H
		img_name = img.split('/')[-1].split('.')[0]
		f = open('/home/tarun/Desktop/antcam/datasets/ants_manual_annotation_patches/labels/val/' + img_name + '.txt', 'w')
		boxes = dictionary_of_boxes[img]
		for b in boxes:
			x_center = (b[0] + b[2])/2
			y_center = (b[1] + b[3])/2
			w = b[2] - b[0]
			h = b[3] - b[1]
			f.write('0 ' + str(x_center/image_width) + ' ' + str(y_center/image_height) + ' ' + str(w/image_width) + ' ' + str(h/image_height) + '\n')
			
		f.close()





### Train

#convert_to_yolo_labels('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/shack/2024-08-23_14_01_01_patches.xml', '/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-22-2024_to_09-11-2024/2024-08-23_14_01_01/', 20, '2024-08-23_14_01_01')
#convert_to_yolo_labels('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/shack/2024-08-01_20_01_00_patches.xml', '/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-22-2024/2024-08-01_20_01_00/', 20, '2024-08-01_20_01_00')
#convert_to_yolo_labels('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/beer/2024-08-01_19_01_00_patches.xml', '/media/tarun/Backup5TB/all_ant_data/beer-tree-08-01-2024_to_08-30-2024/2024-08-01_19_01_00/', 20, '2024-08-01_19_01_00')
#convert_to_yolo_labels('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/rain/2024-10-04_00_01_06_patches.xml', '/media/tarun/Backup5TB/all_ant_data/rain-tree-10-03-2024_to_10-25-2024/2024-10-04_00_01_06/', 20, '2024-10-04_00_01_06')

### Val
#convert_to_yolo_labels('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/shack/2024-08-13_11_01_01_patches.xml', '/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-22-2024/2024-08-13_11_01_01/', 20, '2024-08-13_11_01_01')
#convert_to_yolo_labels('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/beer/2024-10-27_23_01_01_patches.xml', '/media/tarun/Backup5TB/all_ant_data/beer-10-22-2024_to_10-28-2024/2024-10-27_23_01_01/', 20, '2024-10-27_23_01_01')
#convert_to_yolo_labels('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/rain/2024-10-09_23_01_00_patches.xml', '/media/tarun/Backup5TB/all_ant_data/rain-tree-10-03-2024_to_10-25-2024/2024-10-09_23_01_00/', 20, '2024-10-09_23_01_00')

### Test
#convert_to_yolo_labels('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/shack/2024-08-22_03_01_01_patches.xml', '/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-22-2024/2024-08-22_03_01_01/', 20, '2024-08-22_03_01_01')
#convert_to_yolo_labels('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/beer/2024-08-03_13_01_01_patches.xml', '/media/tarun/Backup5TB/all_ant_data/beer-tree-08-01-2024_to_08-30-2024/2024-08-03_13_01_01/', 20, '2024-08-03_13_01_01')
#convert_to_yolo_labels('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/rain/2024-08-22_21_01_00_patches.xml', '/media/tarun/Backup5TB/all_ant_data/rain-tree-08-22-2024_to_09-02-2024/2024-08-22_21_01_00/', 20, '2024-08-22_21_01_00')
