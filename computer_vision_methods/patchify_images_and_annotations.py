'''

For small object detection, one approach often taken is to divide the image into smaller patches and have each
patch be a training example. This would allow for more training data and also would make augmentations more feasiable.
This code takes an XML annotation file downloaded from CVAT, and for each of the images contained in that XML file,
divides the image into a number of patches, filters out and keeps respective annotations in those patches only,
and creates a new XML file where each patch is treated as an image and its corresponding annotations are stored in the
typical CVAT XML annotation format. 

We can then use a script like creating_yolo_labels_from_xml.py with these new XMLs without having to modify anything in that code 

For purposes of simplification, stride = patch_size (i.e no overlap in patches)

'''

import sys
sys.path.append('../')
sys.path.append('../utils')
sys.path.append('../preprocessing_for_annotation')
sys.path.append('../visualization')
import cv2
import numpy as np
import math
from collections import defaultdict
import read_xml_files_from_cvat, converting_points_to_boxes
import create_xml_annotations

class Patchify:
	def __init__(self, annotation_xml_file, folder_where_annotated_video_came_from):
		self.annotation_xml_file = annotation_xml_file
		self.folder_where_annotated_video_came_from = folder_where_annotated_video_came_from
		self.image_width, self.image_height = 1920,1080
		self.grid_size = 3 ### we want a 3x3 grid of patches with no overlap  
		self.patch_width, self.patch_height = int(self.image_width/self.grid_size), int(self.image_height/self.grid_size)
		print ('creating xml file for CVAT')
		self.xml_root = create_xml_annotations.create_xml_file_boilerplate(job_id='12345')
		self.id = 0
				
	def load_annotations(self):
		gts_points_dict = read_xml_files_from_cvat.xml_to_dict_cvat_for_images(self.annotation_xml_file, self.folder_where_annotated_video_came_from)
		
		for img_file in gts_points_dict:
			points = gts_points_dict[img_file] ## this is a list of lists [[x_center, y_center], [x_center, y_center], [x_center, y_center], ...]
			self.patchify(img_file, points)

	def patchify(self, img_file, points):
		### for given annotations for an image, go through them and divide them into bins where each bin contains annotations for patch_i
		bins = defaultdict(list)
		for (x,y) in points:
			grid_pos_X = int(x/self.patch_width)
			grid_pos_Y = int(y/self.patch_height)

			bins[grid_pos_X + self.grid_size*grid_pos_Y].append([x%self.patch_width, y%self.patch_height])

		### for a given img_file, read the image, divide it into patches
		img = cv2.imread(img_file)
		name = img_file.split('/')[-1].split('.')[0]
		for i in range(self.grid_size):
			for k in range(self.grid_size):
				ind = i*self.grid_size + k
				img_patch = img[i*self.patch_height:(i+1)*self.patch_height,k*self.patch_width:(k+1)*self.patch_width,:]
				patch_points = bins[ind]

				### visualize to make sure
				# for (px,py) in patch_points:
				# 	cv2.circle(img_patch, (px, py), 5, (0, 0, 255), -1)
				# cv2.imshow('w',img_patch)
				# cv2.waitKey(1000)

				patched_img_name = self.folder_where_annotated_video_came_from + '/' + name + '_patch_' + str(ind) + '.jpg'
				cv2.imwrite(patched_img_name, img_patch)

				if len(patch_points):
					create_xml_annotations.add_img_and_points_to_xml_file(self.xml_root, patch_points, name + '_patch_' + str(ind) + '.jpg', self.patch_width, self.patch_height, self.id, self.annotation_xml_file.split('.')[0] + '_patches.xml')
					self.id += 1
				

if __name__ == '__main__':
	#P = Patchify('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/shack/2024-08-22_03_01_01.xml', '/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-22-2024/2024-08-22_03_01_01/')
	#P = Patchify('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/shack/2024-08-01_20_01_00.xml', '/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-22-2024/2024-08-01_20_01_00/')
	#P = Patchify('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/shack/2024-08-13_11_01_01.xml', '/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-22-2024/2024-08-13_11_01_01/')
	#P = Patchify('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/shack/2024-08-23_14_01_01.xml', '/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-22-2024_to_09-11-2024/2024-08-23_14_01_01/')
	
	#P = Patchify('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/beer/2024-08-01_19_01_00.xml', '/media/tarun/Backup5TB/all_ant_data/beer-tree-08-01-2024_to_08-30-2024/2024-08-01_19_01_00/')
	#P = Patchify('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/beer/2024-10-27_23_01_01.xml', '/media/tarun/Backup5TB/all_ant_data/beer-10-22-2024_to_10-28-2024/2024-10-27_23_01_01/')
	#P = Patchify('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/beer/2024-08-03_13_01_01.xml', '/media/tarun/Backup5TB/all_ant_data/beer-tree-08-01-2024_to_08-30-2024/2024-08-03_13_01_01/')
	
	#P = Patchify('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/rain/2024-10-04_00_01_06.xml', '/media/tarun/Backup5TB/all_ant_data/rain-tree-10-03-2024_to_10-25-2024/2024-10-04_00_01_06/')
	#P = Patchify('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/rain/2024-10-09_23_01_00.xml', '/media/tarun/Backup5TB/all_ant_data/rain-tree-10-03-2024_to_10-25-2024/2024-10-09_23_01_00/')
	P = Patchify('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/rain/2024-08-22_21_01_00.xml', '/media/tarun/Backup5TB/all_ant_data/rain-tree-08-22-2024_to_09-02-2024/2024-08-22_21_01_00/')

	P.load_annotations()