import cv2
import math
from ultralytics import YOLO
import csv
import sys
repo_path = '/home/tarun/Desktop/ant_detection_and_tracking'
sys.path.append(repo_path + '/preprocessing_for_annotation')
import preprocessing_for_annotation
import numpy as np
import glob
import os

def process_image(model, img, imgsz=1920):
	results = model.predict(img, device='cpu', save=False, imgsz=imgsz, workers=1, show_labels=False, conf=0.1, line_width=1, iou=0.7, max_det=1000)  # return a list of Results objects
	
	boxes = results[0].boxes  # Boxes object for bounding box outputs
	list_of_boxes = boxes.data.numpy()[:,0:5].tolist() ## list of lists [[x1,y1,x2,y2,c], [x1,y1,x2,y2,c], [x1,y1,x2,y2,c]...]
	return list_of_boxes


def process_video_and_store_csv(model, vid):
	### run detections on every frame of the video and store: frame_id,x1,y1,x2,y2,conf in a csv assuming vid is average subtracted already
	average_frame = preprocessing_for_annotation.calculate_avg_frame(vid)
	frame_h,frame_w = average_frame.shape

	cap = cv2.VideoCapture(vid)
	vid_name = vid.split('/')[-1]
	vid_location = '/'.join(vid.split('/')[:-1]) + '/'

	csv_file = open(vid_location + vid_name.split('.')[0] + '_yolo_detections.csv', 'w', newline='')
	csv_writer = csv.writer(csv_file)
	csv_writer.writerow(['frame_number','x1', 'y1', 'x2', 'y2', 'confidence'])

	frame_number = 0 ## always 0 indexed
	while(1):
		ret, frame = cap.read()
		if frame is None:
			break
		
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		new_frame = np.zeros((frame_h,frame_w,3), dtype="uint8")
		new_frame[:,:,0] = gray
		new_frame[:,:,1] = cv2.absdiff(gray,average_frame)*3
		list_of_boxes = process_image(model, new_frame, imgsz=1920)
		for b in list_of_boxes:
			## add frame number at the front
			b.insert(0, frame_number)

		csv_writer.writerows(list_of_boxes)
		frame_number +=1

	csv_file.close()




if __name__ == '__main__':
	model = YOLO("/home/tarun/Desktop/ant_detection_and_tracking/computer_vision_methods/yolo_ultralytics/runs/detect/train3/weights/best.pt")
	#list_of_boxes = process_image(model,"/media/tarun/Backup5TB/all_ant_data/beer-tree-07-17-2024_to_07-31-2024/2024-07-21_06_01_02/2024-07-21_06_01_02_100.jpg", imgsz=1920)

	## this should be the original gray video
	vid_folders = glob.glob('/media/tarun/Backup5TB/all_ant_data/beer-10-22-2024_to_11-02-2024/*')
	for vid_folder in vid_folders:
		folder = vid_folder
		name = vid_folder.split('/')[-1]
		video = folder + '/' +  name + '.mp4'
		if os.path.exists(video):
			print ('######### ' + video + ' ###############')
			process_video_and_store_csv(model, video)