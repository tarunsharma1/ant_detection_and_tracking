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
import numpy as np

image_width, image_height = 1920,1080
#image_width, image_height = 1920/3,1080/3 ## for patches


'''
Since the predictions (post tracking) csvs are masked, we need to apply the masks here as well to the ground truth tracks. 
'''
def filter_detections(detections, mask):
    """
    Removes detections that fall inside the masked area.
    
    detections: List of bounding boxes in [x, y, x, y] format.
    mask: Binary mask where 0 means ignore.
    
    Returns: Filtered list of detections.
    """
    filtered_detections = []
    
    for (id_, x1, y1, x2, y2) in detections:
        # Compute the center of the bounding box
        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
        
        # Check if the center is inside the mask (0 = ignore, 255 = keep)
        if mask[center_y, center_x] == 255:
            filtered_detections.append([id_, x1, y1, x2, y2])
    
    return np.array(filtered_detections)



def convert_to_csv(annotation_xml_file, folder_where_annotated_video_came_from, box_size, label):
	gts_points_dict = read_xml_files_from_cvat.xml_to_dict_cvat_for_videos(annotation_xml_file)
	dictionary_of_boxes = {}
	for f in gts_points_dict.keys():
		dictionary_of_boxes[f] = []
		dictionary_of_boxes[f] = converting_points_to_boxes.convert_points_to_boxes(gts_points_dict[f], box_size, 1920, 1080)

	mask = cv2.imread(mask_dict[label], 0)
	ret, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

	f = open('/home/tarun/Desktop/TrackEval/data/gt/mot_challenge/ant_tracks-val/' + label + '/gt/gt.txt', 'w')
    
	for frame in dictionary_of_boxes:
		boxes = dictionary_of_boxes[frame]
		boxes = filter_detections(boxes, mask_bin)
		for b in boxes:
			ant_id = b[0]
			x1 = b[1]
			y1 = b[2]
			w = b[3] - b[1]
			h = b[4] - b[2]
			## frame_number is required to be 1 indexed for TrackEval
			f.write(str(1 + frame) + ',' + str(ant_id) + ',' + str(x1) + ',' + str(y1) + ',' + str(w) + ',' + str(h) + ',-1,-1,-1,-1' + '\n')
			
	f.close()




### Train

#convert_to_yolo_labels('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/shack/2024-08-23_14_01_01_patches.xml', '/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-22-2024_to_09-11-2024/2024-08-23_14_01_01/', 20, '2024-08-23_14_01_01')
#convert_to_yolo_labels('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/shack/2024-08-01_20_01_00_patches.xml', '/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-22-2024/2024-08-01_20_01_00/', 20, '2024-08-01_20_01_00')
#convert_to_yolo_labels('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/beer/2024-08-01_19_01_00_patches.xml', '/media/tarun/Backup5TB/all_ant_data/beer-tree-08-01-2024_to_08-30-2024/2024-08-01_19_01_00/', 20, '2024-08-01_19_01_00')
#convert_to_yolo_labels('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/rain/2024-10-04_00_01_06_patches.xml', '/media/tarun/Backup5TB/all_ant_data/rain-tree-10-03-2024_to_10-25-2024/2024-10-04_00_01_06/', 20, '2024-10-04_00_01_06')

#convert_to_yolo_labels('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/rain/2024-09-01_11_01_00.xml', '/media/tarun/Backup5TB/all_ant_data/rain-tree-08-22-2024_to_09-02-2024/2024-09-01_11_01_00/', 20, '2024-09-01_11_01_00')
#convert_to_yolo_labels('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/beer/2024-08-03_13_01_01.xml', '/media/tarun/Backup5TB/all_ant_data/beer-tree-08-01-2024_to_08-30-2024/2024-08-03_13_01_01/', 20, '2024-08-03_13_01_01')

mask_dict = {'2024-10-09_23_01_00':'/home/tarun/Desktop/masks/rain-tree-10-03-2024_to_10-19-2024.png', 
    '2024-10-27_23_01_01': '/home/tarun/Desktop/masks/beer-10-22-2024_to_11-02-2024.png', 
    '2024-08-13_11_01_01': '/home/tarun/Downloads/shack-tree-diffuser-08-01-2024_to_08-26-2024.png'
    }


### Val
convert_to_csv('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/shack/2024-08-13_11_01_01_tracking.xml', '/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-22-2024/2024-08-13_11_01_01/', 20, '2024-08-13_11_01_01')
convert_to_csv('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/beer/2024-10-27_23_01_01_tracking.xml', '/media/tarun/Backup5TB/all_ant_data/beer-10-22-2024_to_10-28-2024/2024-10-27_23_01_01/', 20, '2024-10-27_23_01_01')
convert_to_csv('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/rain/2024-10-09_23_01_00_tracking.xml', '/media/tarun/Backup5TB/all_ant_data/rain-tree-10-03-2024_to_10-25-2024/2024-10-09_23_01_00/', 20, '2024-10-09_23_01_00')

### Test
#convert_to_yolo_labels('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/shack/2024-08-22_03_01_01_patches.xml', '/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-22-2024/2024-08-22_03_01_01/', 20, '2024-08-22_03_01_01')
#convert_to_yolo_labels('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/rain/2024-08-22_21_01_00_patches.xml', '/media/tarun/Backup5TB/all_ant_data/rain-tree-08-22-2024_to_09-02-2024/2024-08-22_21_01_00/', 20, '2024-08-22_21_01_00')
