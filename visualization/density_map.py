import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import numpy as np
import cv2
import math
from matplotlib import colors



def plot_and_save_density_map(csv_file):
	## Load per-frame detections
	df = pd.read_csv(csv_file)
	
	df = df.loc[df.direction == 'toward']
	if len(df) == 0:
		return

	df["dx"] = df.groupby("ant_id")["x1"].diff().fillna(0)
	df["dy"] = df.groupby("ant_id")["y1"].diff().fillna(0)
	df["displacement"] = np.sqrt(df["dx"]**2 + df["dy"]**2)

	#df_filtered = df[df["displacement"]>= 5]
	df_filtered = df
	
	vid_name = csv_file.split('/')[-1].split('.')[0]

	## Plot density heatmap
	X,Y = (df_filtered['x1'] + df_filtered['x2'])/2, -1* (df_filtered['y1'] + df_filtered['y2'])/2

	bin_size = 5  # Adjust based on video resolution
	x_min, x_max = 0, 1920
	y_min, y_max = -1080, 0

	# Define grid
	x_bins = np.arange(x_min, x_max, bin_size)
	y_bins = np.arange(y_min, y_max, bin_size)

	### use the numpy histogram2d for any mathematical calculation and the plt hist2D for visualization
	heatmap, x_edges, y_edges = np.histogram2d(X, Y, bins=[x_bins, y_bins])

	### check for outlier bins in the histogram (potentially static ants)
	nonzero_heatmap = heatmap[heatmap > 1]
	mean_count = np.mean(nonzero_heatmap)
	std_count = np.std(nonzero_heatmap)

	## Define threshold for outliers (e.g., bins more than 3 stds from the mean)
	threshold = mean_count + 3 * std_count
	thresholds.append(threshold) ### I'm using the mean of these across all the videos as the upper limit of the colorbar
	#print (threshold)
	
	## Identify outlier bins
	#outlier_bins = np.argwhere(heatmap > threshold)
	fig,ax = plt.subplots()
		
	
	## Mark and set to zero outlier bins
	#if np.shape(nonzero_heatmap)[0] != 0:
	#	print (f'{len(outlier_bins)*100.0/np.shape(nonzero_heatmap)[0]}% of histogram bins with count>1 detected as outliers')
	
	#for row, col in outlier_bins:
		#pass
		#heatmap[row,col] = 0
	#	plt.scatter(x_edges[row], y_edges[col], color="red", edgecolors="black", marker="o", s=100, label="Outlier")


	levels = range(33)
	
	hist, xedges, yedges, image = ax.hist2d(X, Y, bins=[x_bins, y_bins], cmap='viridis', norm=colors.BoundaryNorm(levels, 256))

	# Add colorbar with custom settings
	cbar = fig.colorbar(image, ax=ax, ticks=levels, extend='both')
	cbar.set_label('Custom Colorbar')

	# Set labels and title
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_title(vid_name.split('_herdnet_tracking_with_direction')[0] + '_toward')
	#plt.show()
	plt.savefig('/home/tarun/Desktop/ant_density_plots_herdnet/beer-10-22-2024_to_11-02-2024/' + vid_name + '_toward.png' )
	plt.close()



thresholds = []


vid_folders = glob.glob('/media/tarun/Backup5TB/all_ant_data/beer-10-22-2024_to_11-02-2024/*')

#vid_folders = ['/media/tarun/Backup5TB/all_ant_data/beer-tree-08-01-2024_to_08-10-2024/2024-08-04_00_01_16']
for idx,vid_folder in enumerate(vid_folders):
	name = vid_folder.split('/')[-1]
	#csv_file = vid_folder + '/' + name + '_yolo_tracking_with_direction.csv'
	csv_file = vid_folder + '/' + name + '_herdnet_tracking_with_direction_and_angle_7_1_0.1.csv'
	
	if os.path.exists(csv_file):
		print (f'plotting density map for {csv_file}')
		plot_and_save_density_map(csv_file)


print (math.ceil(np.nanmean(np.array(thresholds))))