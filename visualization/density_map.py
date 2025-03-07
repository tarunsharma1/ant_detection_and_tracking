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
	#df = df.loc[df.confidence >= 0.362]
	df = df.loc[df.direction == 'toward']
	

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

	
	heatmap, x_edges, y_edges = np.histogram2d(X, Y, bins=[x_bins, y_bins])

	### check for outlier bins in the histogram (potentially static ants)
	nonzero_heatmap = heatmap[heatmap > 1]
	mean_count = np.mean(nonzero_heatmap)
	std_count = np.std(nonzero_heatmap)

	## Define threshold for outliers (e.g., bins more than 3 stds from the mean)
	threshold = mean_count + 3 * std_count
	#print (threshold)
	## Identify outlier bins
	outlier_bins = np.argwhere(heatmap > threshold)
	plt.figure(figsize=(8, 6))
	
	
	## Mark and set to zero outlier bins
	print (f'{len(outlier_bins)*100.0/np.shape(nonzero_heatmap)[0]}% of histogram bins with count>1 detected as outliers')
	for row, col in outlier_bins:
		#pass
		heatmap[row,col] = 0
		#plt.scatter(x_edges[row], y_edges[col], color="red", edgecolors="black", marker="o", s=100, label="Outlier")


	plt.imshow(heatmap.T, origin="lower", extent=[x_min, x_max, y_min, y_max], cmap="viridis", aspect='auto')
	plt.colorbar(label="Ant Activity (Raw Count)")
	plt.title(f"Ant Activity Heatmap ")

	plt.xlabel("X Position")
	plt.ylabel("Y Position")
	#plt.show()
	plt.savefig('/home/tarun/Desktop/ant_density_plots/rain-tree-08-22-2024_to_09-02-2024/' + vid_name + '_toward.png' )
	plt.close()


### shack tree mask
#mask = cv2.imread('/home/tarun/Desktop/masks/2024-08-01_20_01_00_10_resized.png',0)

### rain tree mask
mask = cv2.imread('/home/tarun/Desktop/masks/2024-08-22_21_01_00_100_resized.png',0)

ret, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)



vid_folders = glob.glob('/media/tarun/Backup5TB/all_ant_data/rain-tree-08-22-2024_to_09-02-2024/*')
for idx,vid_folder in enumerate(vid_folders):
	#vid_folder = '/media/tarun/Backup5TB/all_ant_data/rain-tree-08-22-2024_to_09-02-2024/2024-08-23_05_01_01'
	name = vid_folder.split('/')[-1]
	#csv_file = vid_folder + '/' + name + '_yolo_detections.csv'
	csv_file = vid_folder + '/' + name + '_yolo_tracking_with_direction.csv'
	
	if os.path.exists(csv_file):
		print (f'plotting density map for {csv_file}')
		plot_and_save_density_map(csv_file)
	