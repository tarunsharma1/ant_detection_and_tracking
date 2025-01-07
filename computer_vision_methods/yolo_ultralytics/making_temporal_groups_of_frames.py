'''

As detection in a single gray frame does not work well due to the ants blending into the tree, 
and bg subtraction does not work in cases of moving leaves and shadows, I want to try giving a group
of frames [frame_(n-2), frame_(n-1), frame_(n), frame_(n+1), frame_(n+2)] as input to a model and 
the output being ground truth detections only for frame_(n). They did this with a U-Net model in this paper 
https://paperswithcode.com/paper/a-method-for-detection-of-small-moving 

Over here I want to try the same thing with Yolov8. The code is to modify the frames currently in my dataset
folder and make new frames. I think I want to do three frames first instead of 5 because then I only have to modify
the images and I can use COCO weights as initialization. 



'''

import cv2
import numpy as np
import glob
from collections import defaultdict
import shutil

def make_new_image_from_group(group_of_images, ref_frame, folder_to_write = '/home/tarun/Desktop/antcam/datasets/ants_manual_annotation_gray_3_frame_seq', split='val'):
	'''
	takes a list of 3 image names, opens them and makes a new image where each channel of the new image is a frame
	saves the new images in a new dataset folder. Ref_frame is a list of center frames for which we keep annotations.
	Ref_frame[0] corresponds to the ref frame for the first group.
	folder_to_write is the folder where the result image should be stored
	'''
	for i in range(0,len(group_of_images)):
		ref = ref_frames[i]
		## copy over annotation file for ref_frame to new folder
		ref_annotation = ref.split('/images')[0] + '/labels' + ref.split('/images')[1].split('.jpg')[0] + '.txt'
		shutil.copy(ref_annotation, folder_to_write + '/labels/' + split + '/')
		
		## make new image as a composite of the group
		group = group_of_images[i]
		
		ref_img = cv2.imread(group[1])
		img_n_minus_1 = cv2.imread(group[0],0)
		img_n_plus_1 = cv2.imread(group[2],0)

		ref_img[:,:,0] = img_n_minus_1
		ref_img[:,:,2] = img_n_plus_1

		cv2.imwrite(folder_to_write + '/images/' + ref.split('/images')[1], ref_img)



training_images = glob.glob('/home/tarun/Desktop/antcam/datasets/ants_manual_annotation_gray/images/val/*.jpg')

video_frame_mapping = defaultdict(list)
for i in training_images:
	video = i.split('_OG_')[0]
	frame = i.split('_OG_')[1].split('.')[0] ## remove the .jpg
	video_frame_mapping[video].append(int(frame)) ## convert to int because sorting strings of numbers is inaccurate

### for each video now sort based on frames
for video in video_frame_mapping:
	groups = []
	ref_frames = []
	frames = sorted(video_frame_mapping[video])
	## make groups of 3 consequtive (frame(n-1), frame(n), frame(n+1))
	for i in range(1, len(frames)-1):
		ref_frames.append(video + '_OG_' + str(frames[i]) + '.jpg')
		groups.append([video + '_OG_' + str(frames[i-1]) + '.jpg', ref_frames[-1], video + '_OG_' + str(frames[i+1]) + '.jpg'])

	make_new_image_from_group(groups, ref_frames)







