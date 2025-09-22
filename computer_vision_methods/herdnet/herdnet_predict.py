import sys
sys.path.append('/home/tarun/Desktop/HerdNet/')

from animaloc.utils.seed import set_seed

set_seed(9292)

import argparse
import torch
import os
import pandas
import warnings
import numpy
import PIL
import csv
from PIL import Image
import albumentations as A
from animaloc.datasets import CSVDataset
from animaloc.data.transforms import MultiTransformsWrapper, DownSample, PointsToMask, FIDT, Rotate90
from torch.utils.data import DataLoader
from animaloc.models import HerdNet
from torch import Tensor
from animaloc.models import LossWrapper
from animaloc.train.losses import FocalLoss
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from animaloc.train import Trainer
from animaloc.eval import PointsMetrics, HerdNetStitcher, HerdNetEvaluator
from animaloc.utils.useful_funcs import mkdir, current_date
from animaloc.vizual import draw_points, draw_text
from animaloc.eval.lmds import HerdNetLMDS

import torchvision.transforms as transforms
import cv2
import sys
repo_path = '/home/tarun/Desktop/ant_detection_and_tracking'
sys.path.append(repo_path + '/preprocessing_for_annotation')
import preprocessing_for_annotation
import numpy as np
import glob
import os



def process_image(stitcher, img, transforms, lmds):
	# img = cv2.imread('/home/tarun/Desktop/antcam/datasets/herdnet_ants_manual_annotation/train/2024-10-04_00_01_06_427.jpg')
	#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	# img_tensor = transforms(img)
	# preds = stitcher(img_tensor)
	# heatmap, clsmap = preds[:,:1,:,:], preds[:,1:,:,:]
	# ### coords are in loc
	# counts, locs, labels, scores, dscores = lmds((heatmap, clsmap))

	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img_tensor = transforms(img)
	img_tensor = img_tensor.to('cuda')
	preds = stitcher(img_tensor)
	heatmap, clsmap = preds[:,:1,:,:], preds[:,1:,:,:]
	counts, locs, labels, scores, dscores = lmds((heatmap, clsmap))

	list_of_boxes = []

	for i in range(0, counts[0][0]):
		point_y, point_x = locs[0][i] ### <- these are center coords. Convert to x1,y1,x2,y2.
		conf = dscores[0][i]
		x1 = max(0, point_x - 10)
		y1 = max(0, point_y - 10)
		x2 = min(point_x+10, 1920)
		y2 = min(point_y+10, 1080)

		box = [x1, y1, x2, y2, conf]

		
		list_of_boxes.append(box)

	return list_of_boxes




def process_video_and_store_csv(stitcher, vid, transforms, lmds):
	### run detections on every frame of the video and store: frame_id,x1,y1,x2,y2,conf in a csv assuming vid is average subtracted already
	average_frame = preprocessing_for_annotation.calculate_avg_frame(vid)
	frame_h,frame_w = average_frame.shape

	cap = cv2.VideoCapture(vid)
	vid_name = vid.split('/')[-1]
	vid_location = '/'.join(vid.split('/')[:-1]) + '/'

	csv_file = open(vid_location + vid_name.split('.')[0] + '_herdnet_detections.csv', 'w', newline='')
	csv_writer = csv.writer(csv_file)
	csv_writer.writerow(['frame_number','x1', 'y1', 'x2', 'y2', 'confidence'])

	frame_number = 0 ## always 0 indexed
	while(1):
		ret, frame = cap.read()
		if frame is None:
			break
		
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		new_frame = np.zeros((frame_h,frame_w,3), dtype="uint8")
		new_frame[:,:,0] = gray
		new_frame[:,:,1] = cv2.absdiff(gray,average_frame)*3
		list_of_boxes = process_image(stitcher, new_frame, transforms, lmds)
		for b in list_of_boxes:
			## add frame number at the front
			b.insert(0, frame_number)

		csv_writer.writerows(list_of_boxes)
		frame_number +=1

	csv_file.close()
	   
   



if __name__ == '__main__':
	patch_size = 512
	num_classes = 2
	down_ratio = 2

	if torch.cuda.is_available():
		map_location = torch.device('cuda')

	checkpoint = torch.load('/home/tarun/Desktop/HerdNet/output_HNP/best_model_with_mean_and_std.pth', map_location=map_location)
	classes = checkpoint['classes']
	num_classes = len(classes) + 1
	img_mean = checkpoint['mean']
	img_std = checkpoint['std']

	model = HerdNet(num_classes=num_classes, pretrained=False)
	model = LossWrapper(model, [])
	model.load_state_dict(checkpoint['model_state_dict'])
	model.eval()

	stitcher = HerdNetStitcher(
            model = model,
            size = (512,512),
            overlap = 160,
            down_ratio = 2,
            up = True, 
            reduction = 'mean',
            device_name = 'cuda'
            )
	transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=img_mean, std=img_std)])
	lmds_kwargs: dict = {'kernel_size': (3, 3), 'adapt_ts': 0.2, 'neg_ts': 0.1}
	lmds = HerdNetLMDS(up=False, **lmds_kwargs)

	## val set only
	#vid_folders = ['/media/tarun/Backup5TB/all_ant_data/rain-tree-10-03-2024_to_10-19-2024/2024-10-09_23_01_00',
	#'/media/tarun/Backup5TB/all_ant_data/beer-10-22-2024_to_11-02-2024/2024-10-27_23_01_01', 
	#'/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-26-2024/2024-08-13_11_01_01']
	
	# vid_folders = glob.glob('/media/tarun/Backup5TB/all_ant_data/rain-tree-08-22-2024_to_09-02-2024/*')
	# vid_folders.extend(glob.glob('/media/tarun/Backup5TB/all_ant_data/rain-tree-10-03-2024_to_10-19-2024/*'))
	# vid_folders.extend(glob.glob('/media/tarun/Backup5TB/all_ant_data/rain-tree-11-15-2024_to_12-06-2024/*'))

	vid_folders = glob.glob('/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-26-2024/*')
	vid_folders.extend(glob.glob('/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-26-2024_to_09-18-2024/*'))
	

	for vid_folder in vid_folders:
		folder = vid_folder
		name = vid_folder.split('/')[-1]
		video = folder + '/' +  name + '.mp4'
		if os.path.exists(video):
			print ('######### ' + video + ' ###############')
			process_video_and_store_csv(stitcher, video, transforms, lmds)