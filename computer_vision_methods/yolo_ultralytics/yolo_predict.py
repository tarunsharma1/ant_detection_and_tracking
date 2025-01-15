import cv2
import math
from ultralytics import YOLO



def process_image(model, img, imgsz=1920):
	results = model.predict(img, device='cpu', save=True, imgsz=imgsz, workers=1, show_labels=True, conf=0.362, line_width=1, iou=0.7, max_det=1000)  # return a list of Results objects
	
	boxes = results[0].boxes  # Boxes object for bounding box outputs
	list_of_boxes = boxes.data.numpy()[:,0:5].tolist() ## list of lists [[x1,y1,x2,y2,c], [x1,y1,x2,y2,c], [x1,y1,x2,y2,c]...]
	return list_of_boxes


if __name__ == '__main__':
	model = YOLO("/home/tarun/Desktop/ant_detection_and_tracking/computer_vision_methods/yolo_ultralytics/runs/detect/train3/weights/best.pt")
	list_of_boxes = process_image(model,"/media/tarun/Backup5TB/all_ant_data/beer-tree-07-17-2024_to_07-31-2024/2024-07-21_06_01_02/2024-07-21_06_01_02_100.jpg", imgsz=1920)