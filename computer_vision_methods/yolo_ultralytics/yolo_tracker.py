import sys
sys.path.append('../')

from ultralytics import YOLO
import csv
from sort.sort import *
import cv2
import math
import pandas as pd



# ant_dict_for_movement = {} # mapping id to distance moved in the last window_length_for_movement_thresh frames
# ant_dict_for_movement_flags = {} # If an ID was moving and then stops, it shouldn't be discarded. This is a flag per id and if the flag is ever set then don't check the distance moved again.
# window_length_for_movement_thresh = 20 # number of frames to calculate distance moved (to remove non-ant static boxes)
# dist_moved_threshold = 30 # pixels to have moved in window_length_for_movement_thresh frames to be considered an ant 

def check_for_movement(x1,y1,x2,y2,ant_id):
	''' legacy function not used anymore '''

	if ant_id not in ant_dict_for_movement.keys():
		ant_dict_for_movement[ant_id] = []
		ant_dict_for_movement_flags[ant_id] = 0
	
	
	## add current coordinates
	dist_moved = 0

	
	if len(ant_dict_for_movement[ant_id]) > window_length_for_movement_thresh: ### actually I need to change this because this is always checking the last 5 frames only. If an ID was moving and then stops, then that's going to be missed.
		## add latest coords and pop one
		ant_dict_for_movement[ant_id].pop(0)
	
	ant_dict_for_movement[ant_id].append([(x1+x2)/2, (y1+y2)/2])

	# accumulate distance moved over the last window_length_for_movement_thresh frames
	for i in range(len(ant_dict_for_movement[ant_id])-1):
		dist_moved += euclidean_distance(ant_dict_for_movement[ant_id][i], ant_dict_for_movement[ant_id][i+1])


	return dist_moved



def filter_detections(detections, mask):
    """
    Removes detections that fall inside the masked area.
    
    detections: List of bounding boxes in [x, y, x, y] format.
    mask: Binary mask where 0 means ignore.
    
    Returns: Filtered list of detections.
    """
    filtered_detections = []
    
    for (x1, y1, x2, y2) in detections:
        # Compute the center of the bounding box
        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
        
        # Check if the center is inside the mask (0 = ignore, 255 = keep)
        if mask[center_y, center_x] == 255:
            filtered_detections.append([x1, y1, x2, y2])
    
    return np.array(filtered_detections)


def distance_from_center(x,y):
	"""
	calculates distance between nest entrance (center coords) and ant position (x,y)
	"""
	center_x, center_y = center_coordinates
	return math.sqrt((y - center_y)**2 + (x - center_x)**2)



def track_video(video_detections_csv, vid_path, vid_name, video):
	
	mot_tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.1)

	### read precomputed detections

	df = pd.read_csv(video_detections_csv)
	df = df.loc[df.confidence >= 0.362]


	cap = cv2.VideoCapture(video)


	#vid_out = cv2.VideoWriter('/home/tarun/Desktop/plots_for_committee_meeting/beer-away-toward-2024-08-03_20_01_01.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (1920,1088))

	dist_moved_all_ants = []
	ant_dict_for_interpolation = {}

	## dictionary to keep track of position (center of box) every frame in order to calculate the direction based on position in current frame and previous frame 
	ant_center = {}
	## dictionary to keep track of direction, (up or down or same) based on the previous frame 
	ant_direction = {}
	## dictionary to keep track of angle in degrees for every ant based on position in current and prev frames
	ant_angle = {}
	
	frame_number = 0

	total_ants_going_towards = []
	total_ants_going_away = []


	#### create a new csv file to store results of tracked ants only along with their direction of movement
	csv_file = open(vid_path + '/' + vid_name.split('.')[0] + '_yolo_tracking_with_direction_and_angle.csv', 'w', newline='')
	csv_writer = csv.writer(csv_file)
	csv_writer.writerow(['frame_number', 'ant_id', 'x1', 'y1', 'x2', 'y2', 'direction', 'angle'])

	

	while True:
		ants_going_towards = 0
		ants_going_away = 0

		ret, frame = cap.read()
		if not ret:
			break

		
		## resize to the shape that yolo is detecting on 
		frame = cv2.resize(frame, (1920, 1088))

		
		df_frame = df.loc[df.frame_number == frame_number]

		all_boxes = df_frame[['x1', 'y1', 'x2', 'y2']].values.tolist()
		all_boxes = np.array(all_boxes)
		#print ('########################')
		#print ('ants detected : ' + str(all_boxes.shape[0]))
		
		all_boxes = filter_detections(all_boxes, mask_bin)
		#print ('ants detected after mask filtering : ' + str(all_boxes.shape[0]))

		if all_boxes.shape[0] == 0:
			frame_number += 1
			continue

		track_bbs_ids = mot_tracker.update(all_boxes)
		#print ('ants tracked : ' + str(track_bbs_ids.shape[0]))

		for tracked_ant in track_bbs_ids:
			x1,y1,x2,y2,ant_id = tracked_ant
			x1,y1,x2,y2 = int(x1),int(y1), int(x2), int(y2)

			if ant_id not in ant_center:
				ant_center[ant_id] = [(x1+x2)/2, (y1+y2)/2]
			else:
				prev_x, prev_y = ant_center[ant_id]
				curr_x, curr_y = (x1+x2)/2, (y1+y2)/2

				## check if distance from nest entrance is increasing or decreasing
				change_in_distance_from_center = distance_from_center(curr_x, curr_y) - distance_from_center(prev_x, prev_y)
				if abs(change_in_distance_from_center) >= direction_threshold and abs(change_in_distance_from_center) <20: ### to prevent id switches from changing direction
					if change_in_distance_from_center > 0:
						## going away from nest entrance
						ant_direction[ant_id] = 1
					else:
						## going toward	nest entrance
						ant_direction[ant_id] = 2

					## tan inverse (y2-y1/x2-x1). We inverted the Ys in order to make 90 degs upward and 270 down (because the frames Y coords go from top left to bottom left)
					ant_angle[ant_id] = (math.degrees(math.atan2( prev_y-curr_y, curr_x - prev_x)) + 360) % 360
				
				# else:
				# 	## skip ants that don't move > threshold. This is only for only for the saving moving ants only method.
				# 	ant_center[ant_id] = [curr_x, curr_y]
				# 	continue

				ant_center[ant_id] = [curr_x, curr_y]

			## if ant hasn't changed direction, or has now stopped moving, still draw the old direction it was traveling in
			if ant_id in ant_direction:
				if ant_direction[ant_id] == 1: 
					#cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
					csv_writer.writerow([frame_number, ant_id, x1, y1, x2, y2, 'away', ant_angle[ant_id]])
					ants_going_away += 1
				
				elif ant_direction[ant_id] == 2:
					#cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
					csv_writer.writerow([frame_number, ant_id, x1, y1, x2, y2, 'toward', ant_angle[ant_id]])
					ants_going_towards += 1
				
				#cv2.putText(frame, str(round(ant_angle[ant_id])), (int((x1+x2)/2), int((y1+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)
		
		total_ants_going_towards.append(ants_going_towards)
		total_ants_going_away.append(ants_going_away)

		#cv2.circle(frame, center_coordinates, 5, (0,0,255), -1)
		#cv2.imshow('Frame',frame)
		#cv2.waitKey(30)
		frame_number += 1

		#vid_out.write(frame)

	csv_file.close()
	if len(total_ants_going_away) != 0 and len(total_ants_going_towards)!=0:
		print (f'avg ants going away from nest entrance: {round(np.mean(np.array(total_ants_going_away)))} and going toward: {round(np.mean(np.array(total_ants_going_towards)))}')
	#vid_out.release()
	cap.release()
	




if __name__ == '__main__':

	direction_threshold = 2

	vid_folders = glob.glob('/media/tarun/Backup5TB/all_ant_data/beer-tree-08-01-2024_to_08-10-2024/*')

	### shack ############
	#mask = cv2.imread('/home/tarun/Downloads/shack-tree-diffuser-08-01-2024_to_08-26-2024.png',0)
	#center_coordinates = (960, 400)

	#mask = cv2.imread('/home/tarun/Downloads/shack-tree-diffuser-08-26-2024_to_09-18-2024.png',0)
	#center_coordinates = (1300, 400)


	### rain ##############
	#mask = cv2.imread('/home/tarun/Desktop/masks/rain-tree-08-22-2024_to_09-02-2024.png',0)
	#center_coordinates = (1050, 600)

	### rain ##############
	#mask = cv2.imread('/home/tarun/Desktop/masks/rain-tree-11-15-2024_to_12-06-2024.png', 0)
	#center_coordinates = (1050, 450)

	### rain ##############
	#mask = cv2.imread('/home/tarun/Desktop/masks/rain-tree-10-03-2024_to_10-19-2024.png', 0)
	#center_coordinates = (900, 550)


	### beer ##############
	mask = cv2.imread('/home/tarun/Desktop/masks/beer-tree-08-01-2024_to_08-10-2024.png',0)
	center_coordinates = (1120, 500)

	### beer ##############
	#mask = cv2.imread('/home/tarun/Desktop/masks/beer-10-22-2024_to_11-02-2024.png',0)
	#center_coordinates = (1120, 600)



	ret, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

	for vid_folder in vid_folders:
		vid_path = vid_folder
		vid_name = vid_folder.split('/')[-1]
		video = vid_path + '/' +  vid_name + '.mp4'
		
		video_detections_csv = vid_path + '/' + vid_name + '_yolo_detections.csv'
		if os.path.exists(video_detections_csv):
			print ('## processing ' + video)
			track_video(video_detections_csv, vid_path, vid_name, video)
		







