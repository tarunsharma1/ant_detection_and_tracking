from ultralytics import YOLO
#from sort.sort import *
import cv2
import math
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict
import numpy as np
from PIL import Image
import os
import sys
import csv
repo_path = '/home/tarun/Desktop/ant_detection_and_tracking'
sys.path.append(repo_path + '/preprocessing_for_annotation')
import preprocessing_for_annotation



def process_image(model, img, imgsz=1920):
    ### sahi usually takes the image path as argument in get_sliced_prediction. It is probably opening it in RGB instead of BGR (opencv does BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = get_sliced_prediction(
    img,
    model,
    slice_height=int(384),
    slice_width=int(640),
    overlap_height_ratio=0,
    overlap_width_ratio=0)

    object_prediction_list = result.object_prediction_list
    list_of_boxes = []

    for pred in object_prediction_list:
        x1,y1,x2,y2 = pred.bbox.to_xyxy()
        
        ## skip the smaller boxes (sometimes SAHI predicts really small boxes idk why)
        w = x2 - x1
        h = y2 - y1
        if w < 10 or h < 10:
           continue
        
        ## lets fix box size to 20
        center_x = round((x1+x2)/2)
        center_y = round((y1+y2)/2)
        x1,y1 = max(center_x-10, 0), max(center_y-10, 0)
        x2,y2 = min(center_x+10, 1919), min(center_y+10, 1079)

        score = pred.score.value
        
        list_of_boxes.append([x1,y1,x2,y2,score])

    return list_of_boxes


def visualize_single_image(detection_model, img):
    #img = "/home/tarun/Desktop/antcam/datasets/ants_manual_annotation/images/val/2024-10-09_23_01_00_250.jpg"
    #frame = cv2.imread(img)
    frame = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    result = get_sliced_prediction(
    img,
    detection_model,
    slice_height=int(360),
    slice_width=int(640),
    overlap_height_ratio=0,
    overlap_width_ratio=0)
    #result = get_prediction(img, detection_model)

    object_prediction_list = result.object_prediction_list
    all_boxes = []

    for pred in object_prediction_list:
        x1,y1,x2,y2 = pred.bbox.to_xyxy()
        
        w = x2 - x1
        h = y2 - y1
        if w < 10 or h < 10:
            continue
        
        ## lets fix box size to 20
        center_x = round((x1+x2)/2)
        center_y = round((y1+y2)/2)
        x1,y1 = max(center_x-10, 0), max(center_y-10, 0)
        x2,y2 = min(center_x+10, 1919), min(center_y+10, 1079)
        score = pred.score.value
        x1,y1,x2,y2 = int(x1),int(y1), int(x2), int(y2)

        all_boxes.append([x1,y1,x2,y2, score])
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.imshow('w', frame)
    cv2.waitKey(30)


def process_video_and_store_csv(model, vid, start_frame, end_frame):
    ### run detections on every frame of the video and store: frame_id,x1,y1,x2,y2,conf in a csv assuming vid is average subtracted already
    average_frame = preprocessing_for_annotation.calculate_avg_frame(vid)
    frame_h,frame_w = average_frame.shape

    cap = cv2.VideoCapture(vid)
    vid_name = vid.split('/')[-1]
    vid_location = '/'.join(vid.split('/')[:-1]) + '/'

    csv_file = open(vid_location + vid_name.split('.')[0] + '_sahi_detections.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame_number','x1', 'y1', 'x2', 'y2', 'confidence'])

    frame_number = 0 ## always 0 indexed
    while(1):
        ret, frame = cap.read()
        if frame is None:
            break
        if frame_number < start_frame-1:
            frame_number +=1
            continue
        if frame_number > end_frame+1:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_frame = np.zeros((frame_h,frame_w,3), dtype="uint8")
        new_frame[:,:,0] = gray
        new_frame[:,:,1] = cv2.absdiff(gray,average_frame)*3
        
        #visualize_single_image(model, new_frame)
        list_of_boxes = process_image(model, new_frame, imgsz=1920)
        for b in list_of_boxes:
            ## add frame number at the front
            b.insert(0, frame_number)

        csv_writer.writerows(list_of_boxes)
        frame_number +=1

    csv_file.close()





if __name__ == "__main__":

    model_path = "/home/tarun/Desktop/ant_detection_and_tracking/computer_vision_methods/yolo_ultralytics/runs/detect/train2_blue_patches/weights/best.pt"

    model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=model_path,
    confidence_threshold=0.1,
    device="cpu")

    

    #### val set ###
    vid_folders = [('/media/tarun/Backup5TB/all_ant_data/rain-tree-10-03-2024_to_10-19-2024/2024-10-09_23_01_00', 250, 280), 
    ('/media/tarun/Backup5TB/all_ant_data/beer-10-22-2024_to_11-02-2024/2024-10-27_23_01_01',200,230),
    ('/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-26-2024/2024-08-13_11_01_01', 500,530)]
    
    for (vid_folder,start_frame, end_frame) in vid_folders:
        folder = vid_folder
        name = vid_folder.split('/')[-1]
        video = folder + '/' +  name + '.mp4'
        if os.path.exists(video):
            print ('######### ' + video + ' ###############')
            process_video_and_store_csv(model, video, start_frame, end_frame)


    #visualize_single_image(model)











