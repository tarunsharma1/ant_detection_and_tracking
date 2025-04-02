
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import numpy as np


def plot_and_save_circular_hist(csv_file):
	df = pd.read_csv(csv_file)
	
	df = df.loc[df.direction == 'away']
	if len(df) == 0:
		return

	
	df_filtered = df
	
	vid_name = csv_file.split('/')[-1].split('.')[0]

	## Plot density heatmap
	X,Y, angles = (df_filtered['x1'] + df_filtered['x2'])/2, -1* (df_filtered['y1'] + df_filtered['y2'])/2, df_filtered['angle']

	radians = np.deg2rad(angles)

	num_bins = 9
	bins = np.linspace(0,2*np.pi, num_bins + 1)

	fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

	counts, _, _ = ax.hist(radians, bins=bins)
	plt.show()













vid_folders = glob.glob('/media/tarun/Backup5TB/all_ant_data/beer-tree-08-01-2024_to_08-10-2024/*')

vid_folders = ['/media/tarun/Backup5TB/all_ant_data/beer-tree-08-01-2024_to_08-10-2024/2024-08-04_00_01_16']
for idx,vid_folder in enumerate(vid_folders):
	name = vid_folder.split('/')[-1]
	#csv_file = vid_folder + '/' + name + '_yolo_detections.csv'
	csv_file = vid_folder + '/' + name + '_yolo_tracking_with_direction_and_angle.csv'
	
	if os.path.exists(csv_file):
		print (f'plotting density map for {csv_file}')
		plot_and_save_circular_hist(csv_file)