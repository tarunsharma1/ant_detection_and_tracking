import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import numpy as np


def plot_and_save_density_map(csv_file):
	## Load per-frame detections
	df = pd.read_csv(csv_file)
	vid_name = csv_file.split('/')[-1].split('.')[0]

	# Plot density heatmap
	X,Y = (df['x1'] + df['x2'])/2, -1* (df['y1'] + df['y2'])/2
	#sns.scatterplot(data=df, x=X, y=-1*Y, hue='frame_number')
	#sns.kdeplot(data=df, x=X, y=-1*Y, fill=True, cmap="inferno", levels=50, thresh=0)

	bin_size = 5  # Adjust based on video resolution
	x_min, x_max = X.min(), X.max()
	y_min, y_max = Y.min(), Y.max()

	# Define grid
	x_bins = np.arange(x_min, x_max, bin_size)
	y_bins = np.arange(y_min, y_max, bin_size)

	heatmap, _, _ = np.histogram2d(X, Y, bins=[x_bins, y_bins])



	plt.figure(figsize=(8, 6))
	plt.imshow(heatmap.T, origin="lower", extent=[x_min, x_max, y_min, y_max], cmap="inferno", aspect='auto')
	plt.colorbar(label="Ant Activity (Raw Count)")
	plt.title(f"Ant Activity Heatmap ")
	plt.xlabel("X Position")
	plt.ylabel("Y Position")
	plt.show()
	#plt.savefig('/home/tarun/Desktop/ant_density_plots/shack_' + vid_name + '.png' )
	plt.close()



vid_folders = glob.glob('/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-22-2024/*')
for vid_folder in vid_folders:
	folder = vid_folder
	name = vid_folder.split('/')[-1]
	csv_file = folder + '/' + name + '_yolo_detections.csv'
	if os.path.exists(csv_file):
		print (f'plotting density map for {csv_file}')
		plot_and_save_density_map(csv_file)
	break