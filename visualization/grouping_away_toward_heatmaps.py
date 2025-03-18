'''
I have saved separate 2D histograms of ants going away and towards center. This code will go through those and group together (side by side) the away and toward maps and save them in 
a new folder in order to make a GIF

'''

import cv2
import glob
import numpy as np

away_plots = glob.glob('/home/tarun/Desktop/ant_density_plots/beer-tree-08-01-2024_to_08-10-2024/*_06_[0-9][0-9]_[0-9][0-9]_*_away.png')
away_plots.sort()

h,w = cv2.imread(away_plots[0], 0).shape

for away_img in away_plots:
	toward_img = away_img.split('_away.png')[0] + '_toward.png'
	## make new image with both of these next to each other
	new_img = np.zeros((h, w*2, 3), dtype=np.uint8)
	new_img[:,0:w,:] = cv2.imread(away_img)
	new_img[:,w:,:] = cv2.imread(toward_img)

	text = away_img.split('/')[-1].split('_yolo')[0]

	# font
	font = cv2.FONT_HERSHEY_SIMPLEX

	# org
	org = (int(w/1.5), 50)

	# fontScale
	fontScale = 1
	 
	# Red color in BGR
	color = (0, 0, 255)

	# Line thickness of 2 px
	thickness = 2
	 
	# Using cv2.putText() method
	image = cv2.putText(new_img, text, org, font, fontScale, 
	                 color, thickness, cv2.LINE_AA, False)

	cv2.imshow('w', image)
	cv2.waitKey(2000)


