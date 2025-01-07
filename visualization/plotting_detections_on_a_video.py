''' 

code to take in a video, get corresponding point or box detections, plot per frame and save the video

'''
import sys
sys.path.append('../')
sys.path.append('../utils')
sys.path.append('../preprocessing_for_annotation')
sys.path.append('../computer_vision_methods')
import cv2
import counting_using_blob_detection
import converting_points_to_boxes

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

	vid_out = cv2.VideoWriter('/home/tarun/Downloads/bloby_rocks_2024-08-30_16_01_00.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (1920,1080))
	point_coords, ant_counts = counting_using_blob_detection.count_ants_using_blob_detection(video)
	
	if boxes:
		boxes = converting_points_to_boxes.convert_points_to_boxes(point_coords, 20, 1920, 1080)
		### plot boxes and save

	cap = cv2.VideoCapture(video)
	frame_number = 0

	while (1):
		ret, frame = cap.read()
		if frame is None:
			break

		points = point_coords[frame_number]
		
		for (px, py) in points:
			px,py = int(px), int(py)
			frame = cv2.circle(frame, (px,py), radius=4, color=(0, 0, 255), thickness=-1)

		
		vid_out.write(frame)
		#cv2.imshow('w', frame)
		#cv2.waitKey(30)
		frame_number += 1



if __name__ == '__main__':
	plot_on_video('/media/tarun/Backup5TB/all_ant_data/rocks-tree-08-30-2024_to_09-11-2024/2024-08-30_16_01_00/2024-08-30_16_01_00.mp4')