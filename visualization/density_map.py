import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import numpy as np
import cv2


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



def plot_and_save_density_map(csv_file):
	## Load per-frame detections
	df = pd.read_csv(csv_file)

	## choose a threshold (right now the files have detections run with 0.1 threshold)
	df = df.loc[df.confidence > 0.1]

	all_boxes = df[['x1', 'y1', 'x2', 'y2']].values.tolist()
	all_boxes = np.array(all_boxes)
	## filter based on mask
	all_boxes = filter_detections(all_boxes, mask_bin)
	df_filtered = pd.DataFrame(all_boxes, columns = ['x1', 'y1', 'x2', 'y2'])

	vid_name = csv_file.split('/')[-1].split('.')[0]

	## Plot density heatmap
	X,Y = (df_filtered['x1'] + df_filtered['x2'])/2, -1* (df_filtered['y1'] + df_filtered['y2'])/2
	

	bin_size = 5  # Adjust based on video resolution
	x_min, x_max = 0, 1920
	y_min, y_max = -1088, 0

	# Define grid
	x_bins = np.arange(x_min, x_max, bin_size)
	y_bins = np.arange(y_min, y_max, bin_size)

	
	heatmap, _, _ = np.histogram2d(X, Y, bins=[x_bins, y_bins])



	plt.figure(figsize=(8, 6))
	plt.imshow(heatmap.T, origin="lower", extent=[x_min, x_max, y_min, y_max], cmap="jet", aspect='auto')
	plt.colorbar(label="Ant Activity (Raw Count)")
	plt.title(f"Ant Activity Heatmap ")
	plt.xlabel("X Position")
	plt.ylabel("Y Position")
	#plt.show()
	plt.savefig('/home/tarun/Desktop/ant_density_plots/shack_' + vid_name + '.png' )
	plt.close()



mask = cv2.imread('/home/tarun/Desktop/masks/2024-08-01_20_01_00_10_resized.png',0)
ret, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)


vid_folders = glob.glob('/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-22-2024/*')
for idx,vid_folder in enumerate(vid_folders):
	folder = vid_folder
	name = vid_folder.split('/')[-1]
	csv_file = folder + '/' + name + '_yolo_detections.csv'
	if os.path.exists(csv_file):
		print (f'plotting density map for {csv_file}')
		plot_and_save_density_map(csv_file)
	