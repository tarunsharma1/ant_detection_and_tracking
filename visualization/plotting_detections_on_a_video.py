''' 

code to take in a video, get corresponding point or box detections, plot per frame and save the video

'''
import sys
sys.path.append('../')
sys.path.append('../utils')
sys.path.append('../preprocessing_for_annotation')
sys.path.append('../computer_vision_methods')
sys.path.append('../computer_vision_methods/yolo_ultralytics')

import cv2
from ultralytics import YOLO

#import counting_and_inserting_counts_into_db
import converting_points_to_boxes
import yolo_predict

def plot_on_video(video, boxes=False):
	## get counts and coordinates for every frame
	### params for opencv blob detection
	params = cv2.SimpleBlobDetector_Params()
	params.filterByColor = True
	params.blobColor = 255
	#params.minThreshold = 30
	#params.maxThreshold = 200
	params.filterByArea = True
	params.minArea = 5
	params.filterByCircularity = False
	params.filterByConvexity = False
	params.filterByInertia = False
	params.minDistBetweenBlobs = 5
	#params.minInertiaRatio = 0.1

	#vid_out = cv2.VideoWriter('/home/tarun/Downloads/bloby_rocks_2024-08-30_16_01_00.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (1920,1080))
	#point_coords, ant_counts = counting_and_inserting_counts_into_db.count_ants_using_blob_detection(video)
	
	if boxes:
		boxes = converting_points_to_boxes.convert_points_to_boxes(point_coords, 20, 1920, 1080)
		### plot boxes and save

	cap = cv2.VideoCapture(video)
	frame_number = 0

	model = YOLO("/home/tarun/Desktop/ant_detection_and_tracking/computer_vision_methods/yolo_ultralytics/runs/detect/train3/weights/best.pt")

	while (1):
		ret, frame = cap.read()
		if frame is None:
			break

		list_of_boxes = yolo_predict.process_image(model, frame)

		#points = point_coords[frame_number]
		
		# for (px, py) in points:
		# 	px,py = int(px), int(py)
		# 	frame = cv2.circle(frame, (px,py), radius=4, color=(0, 0, 255), thickness=-1)

		for box in list_of_boxes:
			px,py = int((box[0] + box[2])/2), int((box[1] + box[3])/2)
			frame = cv2.circle(frame, (px,py), radius=4, color=(0, 0, 255), thickness=-1)

		#vid_out.write(frame)
		cv2.imshow('w', frame)
		cv2.waitKey(30)
		frame_number += 1



if __name__ == '__main__':
	plot_on_video('/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-22-2024/2024-08-02_11_01_00/2024-08-02_11_01_00_avg_subtracted.mp4')