import ultralytics
import torch
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8n.yaml').load('yolov8n.pt')

# Train the model
#results = model.train(device='cpu', data='./datasets/shack_tree/shack_tree.yaml', epochs=100, imgsz=640, batch=16, workers=1) ### ---> folder: train14 mAP50 on val with conf=0.01 iou=0.5 imgsz=640 batch=16: 0.176
#results = model.train(device='cpu', data='./datasets/shack_tree/shack_tree.yaml', epochs=100, imgsz=640, batch=16, workers=1, scale=0, translate=0, fliplr=0, mosaic=0,auto_augment=None, erasing=0, crop_fraction=0)
#results = model.train(device='cpu', data='/home/tarun/Desktop/antcam/datasets/ants_manual_annotation/ants_manual_annotation.yaml', epochs=100, imgsz=1920, batch=8, workers=1, scale=0, copy_paste=1, translate=0, fliplr=0, mosaic=0,auto_augment=None, erasing=0, crop_fraction=0)
results = model.train(device='cpu', data='/home/tarun/Desktop/antcam/datasets/ants_manual_annotation_patches/ants_manual_annotation_patches.yaml', epochs=100, imgsz=640, batch=8, workers=1, scale=0.1, copy_paste=1, translate=0.1, fliplr=0.5, flipud=0.5, mosaic=0,auto_augment=None, erasing=0, crop_fraction=0)
