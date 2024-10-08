#### code to check videos collected from a location by making a collage per day
#### where each grid of the collage is the first frame from a different hour of the same day


import cv2
import glob
import math
import sys
import numpy as np

#data_location = '/home/tarun/Desktop/chaney_ant_data/beer_tree/2024-03-16_to_2024-04-02_17days/'
data_location = '/media/tarun/Backup5TB/all_ant_data/shack-tree-retrieved-06-26-2024/'
days = list(range(10,12))
#days = ['06','07','08','09']
for day in days:
	date = '2024-06-'+str(day)

	## get all folders from that date
	all_folders = glob.glob(data_location + date + '*')
	all_folders = sorted(all_folders)

	number_of_timepoints = len(all_folders)
	print (number_of_timepoints)

	if number_of_timepoints < 4:
		print ('too few timepoints -> less than 4')
		sys.exit(0)

	 ### assuming 24 hours in a day (unless something unusual happens), each grid will be (480,360) (w,h) 
	cols = 4
	rows = math.ceil(number_of_timepoints/4)

	collage = np.zeros((360*cols, 480*6, 3))

	for idx,folder in enumerate(all_folders):
		row = int(idx/cols)
		col = int(idx%cols)


		cap = cv2.VideoCapture(folder + '/vid.h264')
	 
		# Check if camera opened successfully
		if (cap.isOpened()== False): 
		  print("Error opening video stream or file")
		  continue
		 
		# Read until video is completed
		frame_count = 0
		while(cap.isOpened()):
			# Capture frame-by-frame
			ret, frame = cap.read()
			if ret == True:
				if frame_count == 0:
					frame1 = frame
					# just get the first frame
					#break
				if frame_count == 10:
					frame2 = frame
					break
			
			frame_count += 1


		cap.release()

		gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
		gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
		diff = cv2.absdiff(gray1, gray2)
		thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		kernel = np.ones((2,2), np.uint8)
		img_erosion = cv2.erode(thresh, kernel, iterations=1)
		img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
		
		frame = cv2.cvtColor(img_dilation, cv2.COLOR_GRAY2BGR)

		frame = cv2.resize(frame, (480,360))
		cv2.putText(frame, str(folder.split('/')[-1]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (36,255,12), 3)

		collage[col*360:(col+1)*360, row*480:(row+1)*480, :] = frame


	cv2.imwrite('/home/tarun/Desktop/antcam/plots/collage_shack_test_'+ date + '.png', collage)