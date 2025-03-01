import sys
sys.path.append('../')

from ultralytics import YOLO
import csv
from sort.sort import *
import cv2
import math
import pandas as pd

ant_dict_for_movement = {} # mapping id to distance moved in the last window_length_for_movement_thresh frames
ant_dict_for_movement_flags = {} # If an ID was moving and then stops, it shouldn't be discarded. This is a flag per id and if the flag is ever set then don't check the distance moved again.
window_length_for_movement_thresh = 20 # number of frames to calculate distance moved (to remove non-ant static boxes)
dist_moved_threshold = 30 # pixels to have moved in window_length_for_movement_thresh frames to be considered an ant 

def euclidean_distance(pt1, pt2):
	return math.sqrt((pt2[1]-pt1[1])**2 + (pt2[0]-pt1[0])**2)


def check_for_movement(x1,y1,x2,y2,ant_id):
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

### shack ############
#mask = cv2.imread('/home/tarun/Desktop/masks/2024-08-01_20_01_00_10_resized.png',0)
#center_coordinates = (960, 400)

### rain ##############
mask = cv2.imread('/home/tarun/Desktop/masks/2024-08-22_21_01_00_100_resized.png',0)
center_coordinates = (1050, 600)


ret, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)


mot_tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.1)


#source = '/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-22-2024/2024-08-01_20_01_00/2024-08-01_20_01_00_avg_subtracted.mp4'
#source = '/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-22-2024/2024-08-01_23_01_01/2024-08-01_23_01_01_avg_subtracted.mp4'
#source = '/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-22-2024/2024-08-02_01_01_00/2024-08-02_01_01_00_avg_subtracted.mp4'
#source = '/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-22-2024/2024-08-02_03_01_06/2024-08-02_03_01_06_avg_subtracted.mp4'
#source = '/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-22-2024/2024-08-02_05_01_01/2024-08-02_05_01_01_avg_subtracted.mp4'
#source = '/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-22-2024/2024-08-02_05_01_01/2024-08-02_05_01_01_avg_subtracted.mp4'

source = '/media/tarun/Backup5TB/all_ant_data/rain-tree-08-22-2024_to_09-02-2024/2024-08-23_05_01_01/2024-08-23_05_01_01_avg_subtracted.mp4'





### read precomputed detections
vid_name = source.split('/')[-1].split('_avg_subtracted.mp4')[0]
vid_path = '/'.join(source.split('/')[:-1])
video_detections_csv = vid_path + '/' + vid_name + '_yolo_detections.csv'

df = pd.read_csv(video_detections_csv)
df = df.loc[df.confidence >= 0.362]


cap = cv2.VideoCapture(source)


#vid_out = cv2.VideoWriter('./test_output_shack.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (1920,1088))

dist_moved_all_ants = []
ant_dict_for_interpolation = {}

## dictionary to keep track of position (center of box) every frame in order to calculate the direction based on position in current frame and previous frame 
ant_center = {}
## dictionary to keep track of direction, (up or down or same) based on the previous frame 
ant_direction = {}
direction_threshold = 2
frame_number = 0

ants_going_towards = 0
ants_going_away = 0

while True:
	ret, frame = cap.read()

	if not ret:
		break	
	ids_rejected = 0
	## resize to the shape that yolo is detecting on 
	frame = cv2.resize(frame, (1920, 1088))
	
	df_frame = df.loc[df.frame_number == frame_number]

	all_boxes = df_frame[['x1', 'y1', 'x2', 'y2']].values.tolist()
	all_boxes = np.array(all_boxes)
	print ('########################')
	print ('ants detected : ' + str(all_boxes.shape[0]))
	
	all_boxes = filter_detections(all_boxes, mask_bin)
	print ('ants detected after mask filtering : ' + str(all_boxes.shape[0]))

	track_bbs_ids = mot_tracker.update(all_boxes)
	print ('ants tracked : ' + str(track_bbs_ids.shape[0]))

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
			if abs(change_in_distance_from_center) >= direction_threshold:
				if change_in_distance_from_center > 0:
					## going away from nest entrance
					ant_direction[ant_id] = 1
				else:
					## going toward	nest entrance
					ant_direction[ant_id] = 2
				
			ant_center[ant_id] = [curr_x, curr_y]

		## if ant hasn't changed direction, still draw the old direction it was traveling in
		if ant_id in ant_direction:
			if ant_direction[ant_id] == 1:
				cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
				ants_going_away += 1
			
			else:
				cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
				ants_going_towards += 1
			

			#cv2.putText(frame, str(ant_id), (int((x1+x2)/2), int((y1+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)


		## before calculating distance moved, check if flag for this ID has been set (i.e has it ever moved before). If yes, skip the calculation and return -1
		# if ant_id not in ant_dict_for_movement_flags:
		# 	ant_dict_for_movement_flags[ant_id] = 0

		# elif ant_dict_for_movement_flags[ant_id] == 1:
		# 	## this ID has moved more than the threshold at some point so it should be an ant and not removed
		# 	cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
		# 	cv2.putText(frame, str(ant_id), (int((x1+x2)/2), int((y1+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)
			
		# else:
		# 	## we don't know if it is an ant or not, we need to check	
		# 	## eliminate the ones that don't move much in 5 frames ##
		# 	dist_moved = check_for_movement(x1,y1,x2,y2,ant_id)
		# 	#dist_moved_all_ants.append(dist_moved)

		# 	if dist_moved > dist_moved_threshold:
		# 		## save to a dictionary per id ..i.e for every id you have X=frame number, Y=box coords and then do linear interpolation for missed frames.
		# 		#ant_dict_for_interpolation[ant_id]
				
		# 		## set flag for future memory so that we don't remove this ID if the ant stops later
		# 		ant_dict_for_movement_flags[ant_id] = 1
				
		# 		cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
		# 		cv2.putText(frame, str(ant_id), (int((x1+x2)/2), int((y1+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)
		# 	else:
		# 		## reject this ID for now but it still gets checked everytime in case it moves at some point. 
		# 		ids_rejected +=1
		
	#print ('rejected ' + str(ids_rejected) + ' ids based on movement')

	#print (f' boxes actually drawn: {ants_going_up + ants_going_down}')
	#print (f' ants going up this frame: {ants_going_up} and going down: {ants_going_down}')
	cv2.circle(frame, center_coordinates, 5, (0,0,255), -1)
	cv2.imshow('Frame',frame)
	cv2.waitKey(30)
	frame_number += 1

	#vid_out.write(frame)


print (f'avg ants going away from nest entrance: {round(ants_going_away/frame_number)} and going toward: {round(ants_going_towards/frame_number)}')


#plt.hist(dist_moved_all_ants, 100)
#plt.show()

#vid_out.release()
cap.release()