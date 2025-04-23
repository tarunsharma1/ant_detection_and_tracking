
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import numpy as np
import math
import cv2
from scipy.stats import chisquare, kstest
from scipy.stats import entropy

'''

Convert angles to a vector (histogram) of discritized angles lets say in 15 degree buckets (so 24 buckets). For each bucket (0-15, 15-30, 30-45 ..., 330-360), 
There are two separate things we are looking at :

1. Distribution of ants going away or toward across trails. This should not be impacted by absolute numbers of ants (min max normalize histograms) and we are looking solely at 
   how the ants are distributed among the various trails (angle bins). So for this, after normalizing the histograms, we might use measures such as histogram correlation or overlap.

   The differences in distributions (trail structures) can be computed between away vs toward for every video, or away of video from day 1 vs away of video from day 30 to test for trail permanence.


2. For given trails (angle bins), what are the differences in ant numbers going either away or toward on each of those trails. For this maybe L2 or L1 distance per bin could work. 

Also I want trail skewness, i.e how different is the distribution from a uniform?

We also want a 24-24 matrix comparing away/toward of every hour with every other hour from that day

'''




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
	import ipdb; ipdb.set_trace()

	plt.show()



def calculate_angle_hist_vector(csv_file, direction, correction=False):
	df = pd.read_csv(csv_file)
	
	df = df.loc[df.direction == direction]
	if len(df) == 0:
		return

	angles = df['angle'].to_list()
	angles = np.array(angles)

	if correction:
		### add correction to either toward or away vector to make histograms aligned (i.e along a trail if an ant is going away, angle is 45. On the same trail if an ant is coming toward, angle is 225.
		### I want to make the two aligned so add 180 and mod by 360). Now if they are using the same trails then histograms for away vs toward should have maximum overlap.
		angles = (angles + 180)%360


	num_frames = max(df['frame_number'])


	## 24 bins of 15 degrees each
	hist, bin_edges = np.histogram(angles, bins=list(range(0,360,15)))

	## divide overall counts for every bin by number of frames so as to get average number of ants within that angle bin
	hist = hist/ num_frames

	### normalize by total number of ants (sum of hist) so that we can separate out differences the effect of differences in numbers of ants going away and toward vs the structure for the L2 score. 
	hist = hist/np.sum(hist)


	return hist, bin_edges


def euclidean_distance_between_away_and_toward(csv_file):
	'''
	Calculates correlation between angle vectors for 'away' vs 'toward' for a given video
	'''
	away_vector, away_bin_edges = calculate_angle_hist_vector(csv_file, 'away')
	toward_vector, toward_bin_edges = calculate_angle_hist_vector(csv_file, 'toward', correction=True)

	total_ant_vector = (away_vector + toward_vector)/2


	## calculate the similarity between the normalized histograms
	#away_vector = np.float32(away_vector)
	#toward_vector = np.float32(toward_vector)

	#score = cv2.compareHist(toward_vector, away_vector, cv2.HISTCMP_CORREL)
	
	score, chi2_p = chisquare(away_vector, toward_vector)
	#score = entropy(toward_vector, away_vector)
	#print(f"Chi-Square Statistic: {score}, p-value: {chi2_p}")

	#plt.stairs(away_vector, away_bin_edges, color='g')
	#plt.stairs(toward_vector, toward_bin_edges, color='r')

	#plt.stairs(total_ant_vector, away_bin_edges)


	l2_distance = math.sqrt(np.sum((away_vector - toward_vector)**2))
	l1_distance = np.sum(np.abs(away_vector - toward_vector))


	#plt.title( ' l1:' + str(round(l1_distance,3)))

	#plt.show()
	vid_name = csv_file.split('/')[-1].split('.')[0]
	#plt.ylim(0,0.2)
	#plt.title(vid_name.split('_herdnet')[0])
	#plt.savefig('/home/tarun/Desktop/ant_direction_histogram_plots_herdnet/' + vid_name + '.png' )
	#plt.close()

	return l2_distance














vid_folders = glob.glob('/media/tarun/Backup5TB/all_ant_data/rain-tree-08-22-2024_to_09-02-2024/*')

plot_corrs = []
for idx,vid_folder in enumerate(vid_folders):
	name = vid_folder.split('/')[-1]
	#csv_file = vid_folder + '/' + name + '_yolo_detections.csv'
	csv_file = vid_folder + '/' + name + '_herdnet_tracking_with_direction_and_angle_7_1_0.1.csv'
	
	if os.path.exists(csv_file):
		print (f'plotting density map for {csv_file}')
		#plot_and_save_circular_hist(csv_file)
		
		plot_corrs.append(euclidean_distance_between_away_and_toward(csv_file))

plt.plot(plot_corrs)
plt.show()
		