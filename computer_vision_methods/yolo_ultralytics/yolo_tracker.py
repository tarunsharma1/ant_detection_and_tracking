import sys
sys.path.append('../')

from ultralytics import YOLO

from sort.sort import *
import cv2
import math


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



# Load an official or custom model

model = YOLO('/home/tarun/Desktop/ant_detection_and_tracking/computer_vision_methods/yolo_ultralytics/runs/detect/train_blue/weights/best.pt')  # Load a custom trained model
mot_tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.1)


source = '/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-22-2024/2024-08-22_14_01_01/2024-08-22_14_01_01_avg_subtracted.mp4'


results = model(device='cpu', source=source, conf=0.1, iou=0.1, imgsz=1920, line_width=1, max_det=1000, show_labels=False, stream=True)


cap = cv2.VideoCapture(source)
vid_out = cv2.VideoWriter('./test_output_shack.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (1920,1088))

dist_moved_all_ants = []
ant_dict_for_interpolation = {}

for idx,result in enumerate(results):
	#if idx > 100:
	#	break
	ids_rejected = 0
	ret, frame = cap.read()
	## resize to the shape that yolo is detecting on 
	frame = cv2.resize(frame, (1920, 1088))
	

	all_boxes = result.boxes.xyxy.numpy() #this is of shape (n_boxes,4) in absolute coordinates
	print ('ants detected : ' + str(all_boxes.shape[0]))
	track_bbs_ids = mot_tracker.update(all_boxes)
	print ('ants tracked : ' + str(track_bbs_ids.shape[0]))
	
	for tracked_ant in track_bbs_ids:
		x1,y1,x2,y2,ant_id = tracked_ant
		x1,y1,x2,y2 = int(x1),int(y1), int(x2), int(y2)

		## before calculating distance moved, check if flag for this ID has been set (i.e has it ever moved before). If yes, skip the calculation and return -1
		if ant_id not in ant_dict_for_movement_flags.keys():
			ant_dict_for_movement_flags[ant_id] = 0

		elif ant_dict_for_movement_flags[ant_id] == 1:
			## this ID has moved more than the threshold at some point so it should be an ant and not removed
			cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
			cv2.putText(frame, str(ant_id), (int((x1+x2)/2), int((y1+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)
			
		else:
			## we don't know if it is an ant or not, we need to check	
			## eliminate the ones that don't move much in 5 frames ##
			dist_moved = check_for_movement(x1,y1,x2,y2,ant_id)
			#dist_moved_all_ants.append(dist_moved)

			if dist_moved > dist_moved_threshold:
				## save to a dictionary per id ..i.e for every id you have X=frame number, Y=box coords and then do linear interpolation for missed frames.
				#ant_dict_for_interpolation[ant_id]
				
				## set flag for future memory so that we don't remove this ID if the ant stops later
				#ant_dict_for_movement_flags[ant_id] = 1
				
				cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
				cv2.putText(frame, str(ant_id), (int((x1+x2)/2), int((y1+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)
			else:
				## reject this ID for now but it still gets checked everytime in case it moves at some point. 
				ids_rejected +=1
		
	print ('rejected ' + str(ids_rejected) + ' ids based on movement')

	cv2.imshow('Frame',frame)
	cv2.waitKey(30)
	vid_out.write(frame)

#plt.hist(dist_moved_all_ants, 100)
#plt.show()

vid_out.release()
cap.release()