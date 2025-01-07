import cv2
import math
from ultralytics import YOLO

# Load a model - train4 is trained on blue images
model = YOLO("/home/tarun/Desktop/ant_detection_and_tracking/computer_vision_methods/yolo_ultralytics/runs/detect/train/weights/best.pt")
# Run batched inference on a list of images
#img = "/media/tarun/Backup5TB/all_ant_data/beer-tree-07-17-2024_to_07-31-2024/2024-07-19_23_01_01/2024-07-19_23_01_01_520.jpg"
img = "/media/tarun/Backup5TB/all_ant_data/beer-tree-07-17-2024_to_07-31-2024/2024-07-21_06_01_02/2024-07-21_06_01_02_100.jpg"

results = model.predict(img, device='cpu', save=True, imgsz=1920, workers=1, show_labels=False, conf=0.1, line_width=1, iou=0.1, max_det=1000)  # return a list of Results objects
import ipdb;ipdb.set_trace()

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    #masks = result.masks  # Masks object for segmentation masks outputs
    #keypoints = result.keypoints  # Keypoints object for pose outputs
    #probs = result.probs  # Probs object for classification outputs
    #obb = result.obb  # Oriented boxes object for OBB outputs
    
    list_of_boxes = boxes.data.numpy()[:,0:4].tolist() ## list of lists [[x1,y1,x2,y1], [x1,y1,x2,y1], [x1,y1,x2,y1]...]

    #result.show()  # display to screen
    #result.save(filename="./runs/result.jpg")  # save to disk