import sys
sys.path.append('../')
sys.path.append('../utils')
sys.path.append('../preprocessing_for_annotation')

import preprocessing_for_annotation
import cv2
import numpy as np
from mysql_dataset import database_helper
"""

This file reads videos from the database, converts it into an average subtracted video and maybe applies a mask too,
performs opencv simple blob detection to detect ants on a subset of frames, returns the average of the number of blobs detected as the count
of the video and stores it in the database.

Instead of running blob detection on every single frame, we are going to run it on a evenly distributed subset of 10 frames

"""

connection = database_helper.create_connection("localhost", "root", "master", "ant_colony_db")



### params for opencv blob detection
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

def count_ants_using_blob_detection(video, subset_of_frames):
	## takes in full path of a video as a str and a list of frame numbers to calculate counts on

	print ('working on video:', video)
	average_frame = preprocessing_for_annotation.calculate_avg_frame(video)
	cap = cv2.VideoCapture(video)
	frame_h,frame_w = average_frame.shape       #### shape is 1080,1920,1

	cap = cv2.VideoCapture(video)
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

	ant_counts = []

	point_coords = []

	frame_number = 0
	while(1):
		ret, frame = cap.read()
		if frame is None:
			break
		if frame_number > max(subset_of_frames):
			break
		if frame_number in subset_of_frames:
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			## apply mask here

			###
			new_frame = np.zeros((frame_h,frame_w,3), dtype="uint8")
			new_frame[:,:,0] = gray
			new_frame[:,:,1] = cv2.absdiff(gray,average_frame)*3

			list_of_points = preprocessing_for_annotation.blob_detection(new_frame, frame_number, params)
			ant_counts.append(len(list_of_points))
			point_coords.append(list_of_points)

		frame_number += 1
	return point_coords, ant_counts
		



def insert_count_into_db():
	query = f"""
	SELECT video_id from Videos;
	"""
	videos = []
	videos_db = database_helper.execute_query(connection, query)

	for vid in videos_db:
		videos.append(vid[0])

	number_of_frames_to_calculate_average_count=50
	subset_of_frames = list(np.linspace(0,total_frames-1, number_of_frames_to_calculate_average_count, dtype=int))


	for video in videos:
		_, ant_counts = count_ants_using_blob_detection(video)

		#print ('counts are : ', ant_count)
		average_ant_count = np.mean(np.array(ant_counts))
		ant_count_std_dev = np.std(np.array(ant_counts))
		print ('average ant count is ' + str(average_ant_count) + ' std dev is' + str(ant_count_std_dev))
		query = f"""
		INSERT INTO Counts (video_id, blob_detection_average_count, blob_detection_std_dev)
		VALUES ('{video}', {average_ant_count}, {ant_count_std_dev});"""
		database_helper.execute_query(connection, query)
		connection.commit()



if __name__ == '__main__':
	insert_count_into_db()


connection.close()



