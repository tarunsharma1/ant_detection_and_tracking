import sys
sys.path.append('./')

import csv
from sort.sort import *
import cv2
import math
import pandas as pd
import numpy as np


import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import random



def precompute_distance_maps(center_mask, chunk_size = 100000, debug=False):
    """
    Precompute closest boundary point lookup table for every pixel using brute force method.
    This gives us the accuracy of brute force with the speed of a lookup table.
    
    Args:
        center_mask: binary mask of entrance (entrance=0, background=255 or vice versa)
        debug: if True, display the detected boundary overlay
    
    Returns:
        closest_boundary_lookup: 2D array where each pixel contains the index of its closest boundary point
        boundary_points: list of boundary points (index matches lookup table)
    """
    # Ensure mask is uint8
    mask = (center_mask > 0).astype(np.uint8) * 255
    
    # Invert so entrance is white (object to detect)
    mask_inv = cv2.bitwise_not(mask)

    # Find contours of the entrance (now white)
    contours, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(contours) == 0:
        print("Warning: No contours found in mask")
        return np.zeros_like(mask, dtype=np.int32), np.array([])

    # Use the largest contour (should be the entrance)
    largest_contour = max(contours, key=cv2.contourArea)
    boundary_points = largest_contour[:, 0, :]

    # Precompute closest boundary point for every pixel using brute force
    height, width = center_mask.shape
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    pixels = np.stack((xx, yy), axis=-1).reshape(-1, 2)  # (N,2)

    closest_idx = np.zeros(len(pixels), dtype=np.int32)

    for start in tqdm(range(0, len(pixels), chunk_size), desc="Precomputing boundary lookup"):
        #print (''start)
        end = min(start + chunk_size, len(pixels))
        batch = pixels[start:end]  # (B,2)

        # vectorized distance calculation for this chunk
        diffs = batch[:, None, :] - boundary_points[None, :, :]
        dists_sq = np.sum(diffs**2, axis=-1)  # (B, M)
        closest_idx[start:end] = np.argmin(dists_sq, axis=1)

    closest_boundary_lookup = closest_idx.reshape(height, width)
    #return closest_boundary_lookup, boundary_points

    print("Precomputation complete!")

    # Debug visualization
    if debug:
        overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(overlay, [largest_contour], -1, (0, 0, 255), 2)  # red boundary
        plt.figure(figsize=(6,6))
        plt.imshow(overlay[..., ::-1])  # BGR→RGB
        plt.title("Detected boundary (red)")
        plt.axis("off")
        plt.show()

    return closest_boundary_lookup, boundary_points

def closest_boundary_point(x, y, closest_boundary_lookup, boundary_points):
    """
    Look up the closest boundary point for a given (x, y) using precomputed lookup table.
    """
    h, w = closest_boundary_lookup.shape
    if not (0 <= x < w and 0 <= y < h):
        return None  # out of frame

    closest_idx = closest_boundary_lookup[y, x]  # index of closest boundary point
    if closest_idx < 0 or closest_idx >= len(boundary_points):
        return None
    
    return tuple(boundary_points[closest_idx])

def find_closest_boundary_point_brute_force(x, y, boundary_points):
    """
    Brute force method to find the truly closest boundary point.
    This is for debugging/comparison.
    """
    if len(boundary_points) == 0:
        return None
    
    min_dist = float('inf')
    closest_point = None
    
    for point in boundary_points:
        px, py = point
        dist = np.sqrt((x - px)**2 + (y - py)**2)
        if dist < min_dist:
            min_dist = dist
            closest_point = (int(px), int(py))
    
    return closest_point


def calculate_angle_between_vectors(vec1, vec2):
    """
    Calculate the angle between two vectors using dot product.
    
    Args:
        vec1, vec2: Lists or arrays representing 2D vectors [x, y]
    
    Returns:
        angle: Angle in degrees between the vectors
    """
    # Convert to numpy arrays
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    # Calculate dot product
    dot_product = np.dot(vec1, vec2)
    
    # Calculate magnitudes
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)
    
    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return -1
    
    # Calculate cosine of angle
    cos_angle = dot_product / (magnitude1 * magnitude2)
    
    # Clamp to avoid numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Convert to degrees
    angle_radians = math.acos(cos_angle)
    angle_degrees = math.degrees(angle_radians)
    
    return angle_degrees


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

def visualize_ant_vectors(frame, ant_center, boundary_point, radial_vector, velocity_vector, ant_id):
    """
    Overlay visualization for one ant:
    - Green dot = ant center
    - Red dot   = nearest boundary point
    - Blue line = radial vector
    - Yellow line = velocity vector
    """
    cx, cy = map(int, ant_center)
    bx, by = boundary_point

    # Draw ant center
    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    # Draw boundary point
    cv2.circle(frame, (bx, by), 4, (0, 0, 255), -1)

    # Draw radial vector (boundary → ant)
    cv2.line(frame, (bx, by), (cx, cy), (255, 0, 0), 2)

    # Draw velocity vector (previous → current ant pos)
    vel_end = (int(cx + velocity_vector[0]*2), int(cy + velocity_vector[1]*2))
    cv2.line(frame, (cx, cy), vel_end, (0, 255, 255), 2)

    # Label with ant ID
    cv2.putText(frame, f"ID:{int(ant_id)}", (cx+5, cy-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

def track_video(closest_boundary_lookup, boundary_points, video_detections_csv, vid_path, vid_name, video, max_age, min_hits, iou_threshold, debug=False):
	
	#mot_tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.1)
	
	mot_tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)

	### read precomputed detections

	df = pd.read_csv(video_detections_csv)
	#df = df.loc[df.confidence >= 0.362]
	df = df.loc[df.confidence >= 0.327] ### <--- 0.327 is the threshold with the best F1 score for herdnet


	cap = cv2.VideoCapture(video)


	#vid_out = cv2.VideoWriter('/home/tarun/Desktop/plots_for_committee_meeting/beer-away-toward-loiter-2024-08-13_00_01_06_thresh_30.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (1920,1080))

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
	csv_file = open(vid_path + '/' + vid_name.split('.')[0] + '_herdnet_tracking_with_direction_and_angle_closest_boundary_thresh_20_' + str(max_age)+'_'+ str(min_hits) + '_' + str(iou_threshold) + '.csv', 'w', newline='')
	csv_writer = csv.writer(csv_file)
	csv_writer.writerow(['frame_number', 'ant_id', 'x1', 'y1', 'x2', 'y2', 'direction', 'angle'])



	while True:
		ants_going_towards = 0
		ants_going_away = 0

		ret, frame = cap.read()
		if not ret:
			break

		
		## resize to the shape that yolo is detecting on 
		#frame = cv2.resize(frame, (1920, 1088)) <-- for yolo
		frame = cv2.resize(frame, (1920, 1080))
		#frame = cv2.bitwise_and(frame,frame, mask = mask_bin)


		
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

				## method 1:check if distance from a manually defined center point in the nest entrance is increasing or decreasing
				#direction_threshold = 2
				# change_in_distance_from_center = distance_from_center(curr_x, curr_y) - distance_from_center(prev_x, prev_y)
				# if abs(change_in_distance_from_center) >= direction_threshold and abs(change_in_distance_from_center) <20: ### to prevent id switches from changing direction
				# 	if change_in_distance_from_center > 0:
				# 		## going away from nest entrance
				# 		ant_direction[ant_id] = 1
				# 	else:
				# 		## going toward	nest entrance
				# 		ant_direction[ant_id] = 2
				## else:
				## 	## skip ants that don't move > threshold. This is only for only for the saving moving ants only method.
				## 	ant_center[ant_id] = [curr_x, curr_y]
				## 	continue

				#############################################################################################

				#############################################################################################
				## method 2: use dot product between radial and velocity vectors to determine direction where radial vector is the vector from the closest point on the center_mask boundary to the ant's center point and velocity vector is the vector from the previous frame to the current frame
				
				# Get closest boundary point using precomputed lookup table (brute force method)
				boundary_point = closest_boundary_point(int(curr_x), int(curr_y), closest_boundary_lookup, boundary_points)

				
				# Debug visualization
				if debug:
					cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
					
					# ##Draw the detected boundary
					if len(boundary_points) > 0:
						boundary_contour = boundary_points.reshape(-1, 1, 2).astype(np.int32)
						cv2.drawContours(frame, [boundary_contour], -1, (0, 255, 255), 2)  # Yellow boundary
					
					# Draw distance transform result (red)
					if boundary_point is not None:
						cv2.circle(frame, boundary_point, 5, (0,0,255), -1)
						cv2.line(frame, (int(curr_x), int(curr_y)), boundary_point, (255,0,0), 2)
						dist_dt = np.sqrt((curr_x - boundary_point[0])**2 + (curr_y - boundary_point[1])**2)
						cv2.putText(frame, f"DT:{dist_dt:.1f}", (int(curr_x), int(curr_y)-10), 
								   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
					
					
					
					if boundary_point is None:
						cv2.putText(frame, "NO BOUNDARY", (int(curr_x), int(curr_y)-10), 
								   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
					
					cv2.imshow('Debug Boundary Points', frame)
					cv2.waitKey(150)
					continue

				
				if boundary_point is not None:
					bx, by = boundary_point
					
					# Calculate radial vector (from boundary to ant)
					radial_vector = [curr_x - bx, curr_y - by]
					
					# Calculate velocity vector (from previous to current position)
					velocity_vector = [curr_x - prev_x, curr_y - prev_y]
					
					# Calculate angle between radial and velocity vectors using dot product
					angle_between_vectors = calculate_angle_between_vectors(radial_vector, velocity_vector)

					if debug:
						visualize_ant_vectors(frame, (curr_x, curr_y), boundary_point, radial_vector, velocity_vector, ant_id)
					
					## if mag of velocity vector or radial vector is 0 (ant not moving or sitting on boundary)
					if angle_between_vectors == -1:
						ant_direction[ant_id] = 3

					else:
						# Determine direction based on angle
						# If angle < 90 degrees: moving away from center (radial and velocity vectors align)
						# If angle > 90 degrees: moving toward center (radial and velocity vectors oppose)
						threshold = 20  # Reduced threshold for more sensitivity
						if angle_between_vectors < 90 - threshold:
							ant_direction[ant_id] = 1  # away
						elif angle_between_vectors > 90 + threshold:
							ant_direction[ant_id] = 2  # toward
						else:
							ant_direction[ant_id] = 3  # loitering/unknown
				else:
					# Fallback: use distance-based method if boundary point not found
					ant_direction[ant_id] = 3  # unknown
				#############################################################################################
				


				## tan inverse (y2-y1/x2-x1). We inverted the Ys in order to make 90 degs upward and 270 down (because the frames Y coords go from top left to bottom left)
				#ant_angle[ant_id] = (math.degrees(math.atan2( prev_y-curr_y, curr_x - prev_x)) + 360) % 360
				ant_angle[ant_id] = angle_between_vectors
				
				ant_center[ant_id] = [curr_x, curr_y]

			## if ant hasn't changed direction, or has now stopped moving, still draw the old direction it was traveling in
			if ant_id in ant_direction:
				if ant_direction[ant_id] == 1: 
					#cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
					#cv2.putText(frame, str(round(ant_angle[ant_id])), (int((x1+x2)/2), int((y1+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
					csv_writer.writerow([frame_number, ant_id, x1, y1, x2, y2, 'away', ant_angle[ant_id]])
					ants_going_away += 1
				
				elif ant_direction[ant_id] == 2:
					#cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
					#cv2.putText(frame, str(round(ant_angle[ant_id])), (int((x1+x2)/2), int((y1+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
					
					csv_writer.writerow([frame_number, ant_id, x1, y1, x2, y2, 'toward', ant_angle[ant_id]])
					ants_going_towards += 1
				
				elif ant_direction[ant_id] == 3:
					csv_writer.writerow([frame_number, ant_id, x1, y1, x2, y2, 'unknown', ant_angle[ant_id]])	
					#cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)
					#cv2.putText(frame, str(round(ant_angle[ant_id])), (int((x1+x2)/2), int((y1+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
					
				
				#cv2.putText(frame, str(round(ant_angle[ant_id])), (int((x1+x2)/2), int((y1+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
				#cv2.putText(frame, str(int(ant_id)), (int((x1+x2)/2), int((y1+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
				
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
	#vid_folders = ['/media/tarun/Backup5TB/all_ant_data/rain-tree-10-03-2024_to_10-19-2024/2024-10-09_23_01_00']

	### shack ############
	#vid_folders = glob.glob('/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-26-2024/*')
	#mask = cv2.imread('/home/tarun/Desktop/masks/shack-tree-diffuser-08-01-2024_to_08-26-2024.png',0)
	#center_mask = cv2.imread('/home/tarun/Desktop/masks/shack-tree-diffuser-08-01-2024_to_08-26-2024_center_only.png', 0)
	#center_coordinates = (960, 400)

	### shack #############
	# vid_folders = glob.glob('/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-26-2024_to_09-18-2024/*')
	# mask = cv2.imread('/home/tarun/Desktop/masks/shack-tree-diffuser-08-26-2024_to_09-18-2024.png',0)
	# center_mask = cv2.imread('/home/tarun/Desktop/masks/shack-tree-diffuser-08-26-2024_to_09-18-2024_center_only.png',0)
	#center_coordinates = (1300, 400)


	### rain ##############
	#vid_folders = glob.glob('/media/tarun/Backup5TB/all_ant_data/rain-tree-08-22-2024_to_09-02-2024/*')
	#mask = cv2.imread('/home/tarun/Desktop/masks/rain-tree-08-22-2024_to_09-02-2024.png',0)
	#center_mask = cv2.imread('/home/tarun/Desktop/masks/rain-tree-08-22-2024_to_09-02-2024_center_only.png',0)
	#center_coordinates = (1050, 600)

	### rain ##############
	#vid_folders = glob.glob('/media/tarun/Backup5TB/all_ant_data/rain-tree-11-15-2024_to_12-06-2024/*')
	#mask = cv2.imread('/home/tarun/Desktop/masks/rain-tree-11-15-2024_to_12-06-2024.png', 0)
	#center_mask = cv2.imread('/home/tarun/Desktop/masks/rain-tree-11-15-2024_to_12-06-2024_center_only.png', 0)
	#center_coordinates = (1050, 450)

	### rain ##############
	#vid_folders = glob.glob('/media/tarun/Backup5TB/all_ant_data/rain-tree-10-03-2024_to_10-19-2024/*')
	#mask = cv2.imread('/home/tarun/Desktop/masks/rain-tree-10-03-2024_to_10-19-2024.png', 0)
	#center_mask = cv2.imread('/home/tarun/Desktop/masks/rain-tree-10-03-2024_to_10-19-2024_center_only.png', 0)
	#center_coordinates = (900, 550)


	### beer ##############
	#vid_folders = glob.glob('/media/tarun/Backup5TB/all_ant_data/beer-tree-08-01-2024_to_08-10-2024/*')
	#mask = cv2.imread('/home/tarun/Desktop/masks/beer-tree-08-01-2024_to_08-10-2024.png',0)
	#center_mask = cv2.imread('/home/tarun/Desktop/masks/beer-tree-08-01-2024_to_08-10-2024_center_only.png', 0)
	#center_coordinates = (1120, 500)

	### beer ##############
	vid_folders = glob.glob('/media/tarun/Backup5TB/all_ant_data/beer-10-22-2024_to_11-02-2024/*')
	mask = cv2.imread('/home/tarun/Desktop/masks/beer-10-22-2024_to_11-02-2024.png',0)
	center_mask = cv2.imread('/home/tarun/Desktop/masks/beer-10-22-2024_to_11-02-2024_center_only.png',0)
	
	#center_coordinates = (1120, 600)


	mask = cv2.resize(mask, (1920, 1080))
	center_mask = cv2.resize(center_mask, (1920, 1080))
	ret, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
	ret, center_mask_bin = cv2.threshold(center_mask, 127, 255, cv2.THRESH_BINARY)


	### calculate distance map once per run (as we are using same mask for all videos in one run)
	closest_boundary_lookup, boundary_points = precompute_distance_maps(center_mask, chunk_size=100000, debug=False)



	for vid_folder in vid_folders:
		vid_path = vid_folder
		vid_name = vid_folder.split('/')[-1]
		video = vid_path + '/' +  vid_name + '.mp4'
		
		#video_detections_csv = vid_path + '/' + vid_name + '_yolo_detections.csv'
		video_detections_csv = vid_path + '/' + vid_name + '_herdnet_detections.csv'
		if os.path.exists(video_detections_csv):
			print ('## processing ' + video)
			# track_video(video_detections_csv, vid_path, vid_name, video, 5, 2, 0.1)
			# track_video(video_detections_csv, vid_path, vid_name, video, 10, 2, 0.1)
			# track_video(video_detections_csv, vid_path, vid_name, video, 15, 2, 0.1)
			# track_video(video_detections_csv, vid_path, vid_name, video, 5, 4, 0.1)
			# track_video(video_detections_csv, vid_path, vid_name, video, 10, 4, 0.1)
			# track_video(video_detections_csv, vid_path, vid_name, video, 15, 4, 0.1)
			# track_video(video_detections_csv, vid_path, vid_name, video, 1, 3, 0.3)
			# track_video(video_detections_csv, vid_path, vid_name, video, 1, 3, 0.1)
			# track_video(video_detections_csv, vid_path, vid_name, video, 10, 1, 0.1)
			# track_video(video_detections_csv, vid_path, vid_name, video, 7, 2, 0.1)
			track_video(closest_boundary_lookup, boundary_points, video_detections_csv, vid_path, vid_name, video, 7, 1, 0.1)
			
			


		







