from ultralytics import YOLO
#from sort.sort import *
import cv2
import math
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict
import numpy as np
from PIL import Image


model_path = "/home/tarun/Desktop/ant_detection_and_tracking/computer_vision_methods/yolo_ultralytics/runs/detect/train2_blue_patches/weights/best.pt"

detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=model_path,
    confidence_threshold=0.2,
    device="cpu",  # or 'cuda:0'
)

#img = "/media/tarun/Backup5TB/all_ant_data/beer-tree-07-17-2024_to_07-31-2024/2024-07-21_06_01_02/2024-07-21_06_01_02_100.jpg"
img = "/home/tarun/Desktop/antcam/datasets/ants_manual_annotation/images/val/2024-08-22_03_01_01_323.jpg"

#result = get_prediction(img, detection_model)
result = get_sliced_prediction(
    img,
    detection_model,
    slice_height=int(1080/3),
    slice_width=int(1920/3),
    overlap_height_ratio=0,
    overlap_width_ratio=0,
)

object_prediction_list = result.object_prediction_list
all_boxes = []

frame = cv2.imread(img)

for pred in object_prediction_list:
	x1,y1,x2,y2 = pred.bbox.to_xyxy()
	x1,y1,x2,y2 = int(x1),int(y1), int(x2), int(y2)
	all_boxes.append([x1,y1,x2,y2])
	
	cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imshow('w', frame)
cv2.waitKey(0)