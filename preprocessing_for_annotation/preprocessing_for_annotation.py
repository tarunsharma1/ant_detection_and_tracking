import cv2
import numpy as np
from skimage.feature import blob_doh
from skimage import measure
import matplotlib.pyplot as plt
import os
import json
from os import path
import copy
import glob
import sys
sys.path.append('../utils')
import convert_video_h264_to_mp4
import create_xml_annotations


"""

This code takes as input a video, converts it into the average subtracted version (one channel is original frame, one channel is the difference 
between the current frame and the average frame) in order to highlight moving ants,
writes a sequence of frames between a specified start and end frame number, performs blob detection on the average subtracted 
version to detect ants on a the starting frame of the mentioned sequence and stores these points in an xml file. This sequence of frames
is then uploaded to CVAT along with the xml file for the first frame only. I then manually correct these points (remove extra, added missing ones)
and track them for the sequence so that I get tracking annotations from CVAT.   

Note: Once I upload the sequence of frames to a new CVAT task, I will then update the job id in the xml file created to reflect the newly created job id.

"""

def calculate_avg_frame(video):
		cap = cv2.VideoCapture(video)
		#mask = cv2.imread('/home/tarun/Desktop/antcam/mask_gimp.png')
		#mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
		average_frame = None
		num_frames = 0	
		while True:
			ret, frame = cap.read()
			if frame is None:
				break
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			## apply mask
			#gray = cv2.bitwise_and(gray,gray, mask= ~mask)
			if average_frame is None:
				average_frame = gray.astype(float)
			else:
				average_frame += gray.astype(float)
			num_frames += 1

		average_frame = average_frame / num_frames
		average_frame = average_frame.astype("uint8")
		cap.release()
		return average_frame


def blob_detection(new_frame, frame_number, params=None):
		frame = new_frame.copy()
		
		list_of_points = []

		if not params:
			params = cv2.SimpleBlobDetector_Params()
			params.filterByColor = True
			params.blobColor = 255
			#params.minThreshold = 30
			#params.maxThreshold = 200
			params.filterByArea = True
			params.minArea = 10
			params.filterByCircularity = False
			params.filterByConvexity = False
			params.filterByInertia = False
			params.minDistBetweenBlobs = 10
			#params.minInertiaRatio = 0.1
		detector = cv2.SimpleBlobDetector_create(params)

		# Detect blobs.
		keypoints = detector.detect(frame)

		# Draw detected blobs as red circles.
		for keypoint in keypoints:
			pt_x, pt_y = keypoint.pt

			list_of_points.append([pt_x, pt_y])

			pt_x, pt_y = int(pt_x), int(pt_y)
			#frame[pt_y-3:pt_y+3, pt_x-3:pt_x+3] = [0,0,255]

		#cv2.imwrite('/home/tarun/Pictures/beer-tree/beer-tree-bloby-' + str(frame_number) + '.jpg',frame)
		 
		# Show keypoints
		#cv2.imshow("Keypoints", frame)
		#cv2.waitKey(30)

		return list_of_points


class Preprocessing_for_Annotation:
	def __init__(self, colony, parent_path, video_folder):
		self.colony = colony ## can be beer, shack, rain, rocks
		self.parent_path = parent_path
		self.video_folder = video_folder


	
	def convert_to_mp4(self):
		## check if h264 file has been converted to mp4. If not, do the convertion
		mp4_file = glob.glob(self.parent_path + self.video_folder + '/*.mp4')

		if len(mp4_file) == 0:
			## convert vid.h264 file to mp4 file
			convert_video_h264_to_mp4.convert(self.video_folder + '/vid.h264', self.video_folder + self.video_folder.split('/')[0] +'.mp4', path=self.parent_path, mask=False)
		else:
			print (' mp4 exists, continuing...')
		video = self.parent_path + self.video_folder + self.video_folder.split('/')[0] +'.mp4'
		return video


	def write_frames_and_xml_for_annotations(self, video, start_frame, end_frame, job_id):
		
		average_frame = calculate_avg_frame(video)
		frame_h,frame_w = average_frame.shape       #### shape is 1080,1920,1
		
		cap = cv2.VideoCapture(video)
		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

		frame_number = 0
		kernel = np.ones((2,2), np.uint8)

		while(1):
			ret, frame = cap.read()
			if frame is None:
				break
	
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			
			if frame_number >= end_frame+1:
				break


			new_frame = np.zeros((frame_h,frame_w,3), dtype="uint8")
			new_frame[:,:,0] = gray
			new_frame[:,:,1] = cv2.absdiff(gray,average_frame)*3
			

			### only frames between start_frame and end_frame for annotations 
			if frame_number >= start_frame and frame_number < end_frame:
				#cv2.imwrite('/home/tarun/Pictures/beer-tree/beer-tree-OG-' + str(frame_number) + '.jpg',new_frame)
				cv2.imwrite(self.parent_path + self.video_folder + self.video_folder[:-1] + '_' + str(frame_number) + '.jpg',new_frame)
				cv2.imwrite(self.parent_path + self.video_folder + self.video_folder[:-1] + '_OG_' + str(frame_number) + '.jpg',gray)

				
				## get blob detections for semi-assisted annotations only for first frame of this sequence and write to xml file. We are doing this on first frame only so that I can manually track on CVAT.
				if frame_number == start_frame:
					list_of_points = blob_detection(new_frame, frame_number)
					## create boierplate xml file with job id from CVAT
					print ('creating xml file for CVAT')
					root = create_xml_annotations.create_xml_file_boilerplate(job_id=job_id)
					create_xml_annotations.add_points_to_xml_file(root, list_of_points, filename= self.parent_path + self.video_folder + self.colony + '_' + self.video_folder[:-1] + '_frame_' + str(start_frame) + '_to_' + str(end_frame) + '.xml')
					print ('done writing to xml file')


			frame_number +=1

if __name__ == '__main__':
	A = Preprocessing_for_Annotation('shack', parent_path = '/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-22-2024/', video_folder='2024-08-22_14_01_01/')
	video = A.convert_to_mp4()
	A.write_frames_and_xml_for_annotations(video, 20, 50, "1280343")