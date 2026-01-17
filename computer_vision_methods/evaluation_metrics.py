import sys
sys.path.append('/home/tarun/Desktop/ant_detection_and_tracking/')
sys.path.append('/home/tarun/Desktop/ant_detection_and_tracking/utils')
sys.path.append('/home/tarun/Desktop/ant_detection_and_tracking/preprocessing_for_annotation')
sys.path.append('/home/tarun/Desktop/ant_detection_and_tracking/visualization')

import json
import matplotlib.pyplot as plt
import numpy as np
import copy
import read_xml_files_from_cvat, converting_points_to_boxes
import preprocessing_for_annotation
import plots_comparing_predictions_and_gt
import cv2
import pandas as pd



parameters = {'axes.labelsize':8,'axes.titlesize':8, 'xtick.labelsize':8, 'font.family':"sans-serif", 'font.sans-serif':['Arial'], 'font.size':8, 'svg.fonttype':'none'}
plt.rcParams.update(parameters)


def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    # compare rows first, find which box is below the other
    if box_1[0] >= box_2[2] or box_2[0] >= box_1[2]:
        iou = 0
        return iou

    # compare cols to see if one box to the left or right or other box
    if box_1[1] >= box_2[3] or box_2[1] >= box_1[3]:
        iou = 0
        return iou

    # calculate intersection
    
    intersection_row = abs(max(box_1[0], box_2[0]) - min(box_1[2], box_2[2]))
    intersection_col = abs(max(box_1[1], box_2[1]) - min(box_1[3], box_2[3]))
    
    
    intersection = intersection_row * intersection_col

    # when calculating union dont include intersection area twice
    union = ((box_1[2] - box_1[0]) * (box_1[3] - box_1[1])) + ((box_2[2] - box_2[0]) * (box_2[3] - box_2[1])) - intersection
    iou = intersection/union

    
    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    local_preds = copy.deepcopy(preds)
    '''
    BEGIN YOUR CODE
    '''
    for pred_file, pred in local_preds.items():
        gt_new = gts[pred_file]
        
        gt = []
        for i in range(len(gt_new)):
        	gt.append(gt_new[i])



        for i in range(len(gt)):
            # convert the gt to integers (some of them are floats)
            gt[i] = [int(item) for item in gt[i]]

            TP_flag = 0
            for j in range(len(pred)):
                
                if pred[j] == -1:
                    continue

                ### take only those predictions which are above the confidence threshold
                # if pred[j][-1] < conf_thr:
                # 	pred[j] = -1
                # 	continue
                
                iou = compute_iou(pred[j][:4], gt[i])
                
                if iou > iou_thr:

                    # if we already associated a prediction with this gt then do not recount it as TP..instead it will be counted as a FP
                    if TP_flag == 1:
                        continue

                    # if no prediction has been associated with this gt, then count it as TP and remove it from list (so its not counted as a FP for a different gt)
                    TP += 1
                    TP_flag = 1
                    # remove it from our list of preds (set it to -1 for now)
                    pred[j] = -1
            
            # if none of the predictions had an overlap with this particular gt, then we know it is a miss
            if TP_flag == 0:
                FN += 1

        # remove the -1s and whatever remains now in pred are all false positives
        pred_clean = [y for y in pred if y!= -1]
        FP += len(pred_clean)
        #import ipdb;ipdb.set_trace()

    #print (TP, FP, FN, conf_thr)

    '''
    END YOUR CODE
    '''
    return TP, FP, FN



def plot_pr_curve(preds_val, gts_val):
    confidence_thrs = []
    for fname in preds_val:
        bboxs = preds_val[fname]
        for box in bboxs:
            confidence_thrs.append(box[4])

    
    #confidence_thrs = np.sort(np.array(confidence_thrs))
    confidence_thrs = np.linspace( np.max(confidence_thrs),0, num=50)



    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('precision recall curve ')

    for iou_thresh in np.array([0.5]):
        tp_train = np.zeros(len(confidence_thrs))
        fp_train = np.zeros(len(confidence_thrs))
        fn_train = np.zeros(len(confidence_thrs))

        for i, conf_thr in enumerate(confidence_thrs):
            thresholded_preds_val = {}
            for image in preds_val:
                thresholded_preds_val[image] = []
                for points in preds_val[image]:
                    if points[4] >= conf_thr:
                        thresholded_preds_val[image].append([points[0], points[1], points[2], points[3], points[4]])

            tp_train[i], fp_train[i], fn_train[i] = compute_counts(thresholded_preds_val, gts_val, iou_thr=iou_thresh)
            

        # Plot training set PR curves
        precision_list = []
        recall_list = []

        for i in range(0,confidence_thrs.shape[0]):
            print ('###' + str(confidence_thrs[i]) + '###')
            print (f'TPs : {tp_train[i]}, FPs : {fp_train[i]}, FNs: {fn_train[i]}')
            
            precision = tp_train[i]/(tp_train[i] + fp_train[i])
            recall = tp_train[i]/(tp_train[i] + fn_train[i])
            if precision==0 and recall==0:
                continue
            print (precision, recall)
            print (f'F1 score at threshold {confidence_thrs[i]} is {2*precision*recall/(precision + recall)}')

            precision_list.append(precision)
            recall_list.append(recall)

        #print (iou_thresh)
        #print (recall_list, precision_list)
        #import ipdb;ipdb.set_trace()
        plt.plot(np.array(recall_list), np.array(precision_list),c='tab:red'), #label='herdnet max F1 score 0.782 at threshold 0.327')

    plt.legend()
    plt.show()



'''

right now code is setup to take in two dictionaries, one for predictions one for gt, of the format 
{ 'frame1': [[box1], [box2], ...], 'frame2': [[box1], [box2], ...], ...} 
where box1 is of format [x1,y1,x2,y2,confidence]


In our case, GT will always come from point based annotations in CVAT, which is giving us XML files per video (30 frames per video).
We need a function to first read the XML files per video (downloaded from cvat), and convert it to a format of 
{'video1_frame1': [boxes], 'video1_frame2': [boxes], ... , 'video2_frame1': [boxes], ...}

For the predictions via blob detection, we can call the function counting_using_blob_detection.count_ants_using_blob_detection(video_id, subset_of_frames)
which will return a list of coords per frame (list of lists) and also a list of counts per frame. Both these lists are of size subset_of_frames.
We will then reformat this into the same format as for the GT above and then we should be good to go.

'''

def filter_detections(detections_dict, mask):
    """
    Removes detections that fall inside the masked area.
    
    detections: List of bounding boxes in [x, y, x, y] format.
    mask: Binary mask where 0 means ignore.
    
    Returns: Filtered list of detections.
    """
    filtered_detections_dict = {}

    for img in detections_dict:
        filtered_detections_dict[img] = []
        detections = detections_dict[img]

        #### for ground truth points
        if len(detections[0]) ==4:
            for (x1, y1, x2, y2) in detections:
                # Compute the center of the bounding box
                center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                
                # Check if the center is inside the mask (0 = ignore, 255 = keep)
                if mask[center_y, center_x] == 255:
                    filtered_detections_dict[img].append([x1, y1, x2, y2])

        ### for predictions
        elif len(detections[0]) == 5:
            for (x1, y1, x2, y2, score) in detections:
                # Compute the center of the bounding box
                center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                
                # Check if the center is inside the mask (0 = ignore, 255 = keep)
                if mask[center_y, center_x] == 255:
                    filtered_detections_dict[img].append([x1, y1, x2, y2, score])

        
    return filtered_detections_dict




def read_and_convert_ground_truth(annotation_xml_file, folder_where_annotated_video_came_from, box_size):
    gts_points_dict = read_xml_files_from_cvat.xml_to_dict_cvat_for_images(annotation_xml_file, folder_where_annotated_video_came_from)
    gts_boxes_dict = {}
    for f in gts_points_dict.keys():
        gts_boxes_dict[f] = []
        gts_boxes_dict[f] = converting_points_to_boxes.convert_points_to_boxes(gts_points_dict[f], box_size, 1920, 1080) 

    return gts_boxes_dict


def get_blob_detection_preds(gts_boxes_dict):
    preds_boxes_dict = {}
    for image in list(gts_boxes_dict.keys()):
        preds_boxes_dict[image] = []
        frame = cv2.imread(image)
        list_of_points = preprocessing_for_annotation.blob_detection(frame)
        preds_boxes_dict[image] = converting_points_to_boxes.convert_points_to_boxes(list_of_points, 20, 1920,1080)

    return preds_boxes_dict


def get_yolo_detection_preds(gts_boxes_dict, folder_where_annotated_video_came_from, label):
    ''' read detections from the precomputed csvs that we have run prediction on already '''

    video_detections_csv = folder_where_annotated_video_came_from + label + '_yolo_detections_train12.csv'
    df = pd.read_csv(video_detections_csv)

    #df = df.loc[df.confidence >= 0.381]

    
    preds_boxes_dict = {}
    for image in list(gts_boxes_dict.keys()):
        preds_boxes_dict[image] = []
        
        frame_number = int(image.split('_')[-1].split('.jpg')[0])
        df_frame = df.loc[df.frame_number == frame_number]
        list_of_points = df_frame[['x1', 'y1', 'x2', 'y2', 'confidence']].values.tolist()

        #list_of_points = yolo_predict.process_image(model, image, imgsz=1920)
        preds_boxes_dict[image] = list_of_points
    
    return preds_boxes_dict


def get_sahi_detection_preds(gts_boxes_dict, folder_where_annotated_video_came_from, label):
    ''' read detections from the precomputed csvs that we have run prediction on already '''

    video_detections_csv = folder_where_annotated_video_came_from + label + '_sahi_detections.csv'
    df = pd.read_csv(video_detections_csv)

    #df = df.loc[df.confidence >= 0.359]

    
    preds_boxes_dict = {}
    for image in list(gts_boxes_dict.keys()):
        preds_boxes_dict[image] = []
        
        frame_number = int(image.split('_')[-1].split('.jpg')[0])
        df_frame = df.loc[df.frame_number == frame_number]
        list_of_points = df_frame[['x1', 'y1', 'x2', 'y2', 'confidence']].values.tolist()

        #list_of_points = yolo_predict.process_image(model, image, imgsz=1920)
        preds_boxes_dict[image] = list_of_points
    
    return preds_boxes_dict

### this is for herdnet predictions on val set by running my herdnet_predict.py script
def get_herdnet_detection_preds(gts_boxes_dict, folder_where_annotated_video_came_from, label):
    ''' read detections from the precomputed csvs that we have run prediction on already '''

    video_detections_csv = folder_where_annotated_video_came_from + label + '_herdnet_detections.csv'
    df = pd.read_csv(video_detections_csv)

    df = df.loc[df.confidence >= 0.327]

    
    preds_boxes_dict = {}
    for image in list(gts_boxes_dict.keys()):
        preds_boxes_dict[image] = []
        
        frame_number = int(image.split('_')[-1].split('.jpg')[0])
        df_frame = df.loc[df.frame_number == frame_number]
        list_of_points = df_frame[['x1', 'y1', 'x2', 'y2', 'confidence']].values.tolist()

        #list_of_points = yolo_predict.process_image(model, image, imgsz=1920)
        preds_boxes_dict[image] = list_of_points
    
    return preds_boxes_dict

### this is for reading tracking results (detection + tracking) and counting unique ant IDs per frame
def get_sort_detections(gts_boxes_dict, folder_where_annotated_video_came_from, label):
    ''' read tracking results from precomputed CSVs and count unique ant IDs per frame '''
    
    video_tracking_csv = folder_where_annotated_video_came_from + label + '_herdnet_tracking_with_direction_and_angle_closest_boundary_thresh_20_7_1_0.1.csv'
    df = pd.read_csv(video_tracking_csv)
    
    # Dictionary to store counts per frame (mapping image paths to counts)
    track_counts_dict = {}
    
    for image in list(gts_boxes_dict.keys()):
        frame_number = int(image.split('_')[-1].split('.jpg')[0])
        df_frame = df.loc[df.frame_number == frame_number]
        
        # Count unique ant IDs for this frame
        unique_ant_count = df_frame['ant_id'].nunique() if len(df_frame) > 0 else 0
        track_counts_dict[image] = unique_ant_count
    
    return track_counts_dict

'''
This next method is to evaluate herdnet predictions as spit out by running the training script 

def get_herdnet_detection_preds(gts_boxes_dict, folder_where_annotated_video_came_from, label):
    video_detections_csv = '/home/tarun/Desktop/antcam/datasets/herdnet_ants_manual_annotation/val/20250402_HerdNet_results/20250402_detections.csv'
    df = pd.read_csv(video_detections_csv)
    #df = df.loc[df.dscores >= 0.4]
    
    preds_boxes_dict = {}
    for image in list(gts_boxes_dict.keys()):
        preds_boxes_dict[image] = []
        
        frame_number = int(image.split('_')[-1].split('.jpg')[0])
        df_frame = df.loc[df.images == label + '_' + str(frame_number) + '.jpg']
        list_of_points = df_frame[['x', 'y']].values.tolist()
        scores = df_frame['dscores'].values.tolist()

        list_of_boxes = converting_points_to_boxes.convert_points_to_boxes(list_of_points, 20, 1920, 1080)

        ## add back the confidence scores to the coordinates
        for i,box in enumerate(list_of_boxes):
            box.append(scores[i])

        preds_boxes_dict[image] = list_of_boxes
        
    return preds_boxes_dict
'''
    



def get_metrics(annotation_xml_file, folder_where_annotated_video_came_from, box_size, label):
    mask = cv2.imread(mask_dict[label], 0)
    ret, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    ## mask is (1920, 1088) for yolo, resize to (1920,1080) for sahi and herdnet


    ### read and convert ground truth points from cvat into boxes #####
    gts_boxes_dict = read_and_convert_ground_truth(annotation_xml_file, folder_where_annotated_video_came_from, box_size)
    ## apply mask
    gts_boxes_dict = filter_detections(gts_boxes_dict, mask_bin)
    
    #### visualize first image to see if boxes are correct and of appropriate size
    #plots_comparing_predictions_and_gt.plot_boxes_on_image(list(gts_boxes_dict.keys())[0], gts_boxes_dict[list(gts_boxes_dict.keys())[0]] , label='2024-08-22_03_01_01_ground_truth')


    ### get predictions for the images in the ground truth dict #####
    #preds_boxes_dict = get_blob_detection_preds(gts_boxes_dict)
    #preds_boxes_dict = get_yolo_detection_preds(gts_boxes_dict, folder_where_annotated_video_came_from, label)
    #preds_boxes_dict = get_sahi_detection_preds(gts_boxes_dict, folder_where_annotated_video_came_from, label)
    preds_boxes_dict = get_herdnet_detection_preds(gts_boxes_dict, folder_where_annotated_video_came_from, label)
    
    ## apply mask
    preds_boxes_dict = filter_detections(preds_boxes_dict, mask_bin)

    #plots_comparing_predictions_and_gt.plot_boxes_on_image(list(preds_boxes_dict.keys())[0], preds_boxes_dict[list(preds_boxes_dict.keys())[0]], label='2024-08-22_03_01_01_prediction')
    plots_comparing_predictions_and_gt.plot_ant_counts_gt_vs_preds(gts_boxes_dict, preds_boxes_dict, label)


    tp, fp, fn = compute_counts(preds_boxes_dict, gts_boxes_dict)
    f1_score = (2*tp)/(2*tp + fp + fn)
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    print (label)
    print ('true positives, false positives, false negs:', tp,fp,fn)
    print ('f1 score is :', f1_score)
    print ('precision, recall:',precision, recall)

    return gts_boxes_dict, preds_boxes_dict



if __name__ == '__main__':

    mask_dict = {'2024-10-09_23_01_00':'/home/tarun/Desktop/masks/rain-tree-10-03-2024_to_10-19-2024.png', 
    '2024-10-27_23_01_01': '/home/tarun/Desktop/masks/beer-10-22-2024_to_11-02-2024.png', 
    '2024-08-13_11_01_01': '/home/tarun/Desktop/masks/shack-tree-diffuser-08-01-2024_to_08-26-2024.png',
    '2024-08-22_03_01_01': '/home/tarun/Desktop/masks/shack-tree-diffuser-08-01-2024_to_08-26-2024.png',
    '2024-08-22_21_01_00': '/home/tarun/Desktop/masks/rain-tree-08-22-2024_to_09-02-2024.png',
    '2024-10-04_12_01_01': '/home/tarun/Desktop/masks/rain-tree-10-03-2024_to_10-19-2024.png',
    '2024-08-05_16_01_01': '/home/tarun/Desktop/masks/shack-tree-diffuser-08-01-2024_to_08-26-2024.png',
    '2024-08-09_11_01_02': '/home/tarun/Desktop/masks/beer-tree-08-01-2024_to_08-10-2024.png',
    '2024-10-22_22_01_00': '/home/tarun/Desktop/masks/beer-10-22-2024_to_11-02-2024.png'

    }

    ### validation set ########
    # gts_1, preds_1 = get_metrics('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/rain/2024-10-09_23_01_00.xml', '/media/tarun/Backup5TB/all_ant_data/rain-tree-10-03-2024_to_10-19-2024/2024-10-09_23_01_00/', 20, '2024-10-09_23_01_00')
    # gts_2, preds_2 = get_metrics('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/beer/2024-10-27_23_01_01.xml', '/media/tarun/Backup5TB/all_ant_data/beer-10-22-2024_to_11-02-2024/2024-10-27_23_01_01/', 20, '2024-10-27_23_01_01')
    # gts_3, preds_3 = get_metrics('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/shack/2024-08-13_11_01_01.xml', '/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-26-2024/2024-08-13_11_01_01/', 20, '2024-08-13_11_01_01')

    # gts_1.update(gts_2)
    # gts_1.update(gts_3)
    # preds_1.update(preds_2)
    # preds_1.update(preds_3)
    ######################################

    ######## test set ####################
    gts_1, preds_1 = get_metrics('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/shack/2024-08-22_03_01_01.xml', '/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-26-2024/2024-08-22_03_01_01/', 20, '2024-08-22_03_01_01')
    track_counts_1 = get_sort_detections(gts_1, '/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-26-2024/2024-08-22_03_01_01/', '2024-08-22_03_01_01')
    
    gts_2, preds_2 = get_metrics('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/rain/2024-08-22_21_01_00.xml', '/media/tarun/Backup5TB/all_ant_data/rain-tree-08-22-2024_to_09-02-2024/2024-08-22_21_01_00/', 20, '2024-08-22_21_01_00')
    track_counts_2 = get_sort_detections(gts_2, '/media/tarun/Backup5TB/all_ant_data/rain-tree-08-22-2024_to_09-02-2024/2024-08-22_21_01_00/', '2024-08-22_21_01_00')
    
    gts_3, preds_3 = get_metrics('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/rain/2024-10-04_12_01_01.xml', '/media/tarun/Backup5TB/all_ant_data/rain-tree-10-03-2024_to_10-19-2024/2024-10-04_12_01_01/', 20, '2024-10-04_12_01_01')
    track_counts_3 = get_sort_detections(gts_3, '/media/tarun/Backup5TB/all_ant_data/rain-tree-10-03-2024_to_10-19-2024/2024-10-04_12_01_01/', '2024-10-04_12_01_01')
    
    gts_4, preds_4 = get_metrics('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/shack/2024-08-05_16_01_01.xml', '/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-26-2024/2024-08-05_16_01_01/', 20, '2024-08-05_16_01_01')
    track_counts_4 = get_sort_detections(gts_4, '/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-26-2024/2024-08-05_16_01_01/', '2024-08-05_16_01_01')
    
    gts_5, preds_5 = get_metrics('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/beer/2024-08-09_11_01_02.xml', '/media/tarun/Backup5TB/all_ant_data/beer-tree-08-01-2024_to_08-10-2024/2024-08-09_11_01_02/', 20, '2024-08-09_11_01_02')
    track_counts_5 = get_sort_detections(gts_5, '/media/tarun/Backup5TB/all_ant_data/beer-tree-08-01-2024_to_08-10-2024/2024-08-09_11_01_02/', '2024-08-09_11_01_02')
    
    gts_6, preds_6 = get_metrics('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/beer/2024-10-22_22_01_00.xml', '/media/tarun/Backup5TB/all_ant_data/beer-10-22-2024_to_11-02-2024/2024-10-22_22_01_00/', 20, '2024-10-22_22_01_00')
    track_counts_6 = get_sort_detections(gts_6, '/media/tarun/Backup5TB/all_ant_data/beer-10-22-2024_to_11-02-2024/2024-10-22_22_01_00/', '2024-10-22_22_01_00')
    

    gts_1.update(gts_2)
    gts_1.update(gts_3)
    gts_1.update(gts_4)
    gts_1.update(gts_5)
    gts_1.update(gts_6)

    
    preds_1.update(preds_2)
    preds_1.update(preds_3)
    preds_1.update(preds_4)
    preds_1.update(preds_5)
    preds_1.update(preds_6)
    
    track_counts_1.update(track_counts_2)
    track_counts_1.update(track_counts_3)
    track_counts_1.update(track_counts_4)
    track_counts_1.update(track_counts_5)
    track_counts_1.update(track_counts_6)

    ##############################################

    ## Calculate overall F1 score (micro-averaging) across all test videos
    overall_tp, overall_fp, overall_fn = compute_counts(preds_1, gts_1)
    overall_f1 = (2*overall_tp)/(2*overall_tp + overall_fp + overall_fn) if (2*overall_tp + overall_fp + overall_fn) > 0 else 0.0
    overall_precision = overall_tp/(overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
    overall_recall = overall_tp/(overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0
    
    print('\n' + '='*80)
    print('OVERALL TEST SET METRICS (Micro-averaging)')
    print('='*80)
    print(f'Overall TP, FP, FN: {overall_tp}, {overall_fp}, {overall_fn}')
    print(f'Overall F1 score: {overall_f1:.4f}')
    print(f'Overall Precision: {overall_precision:.4f}')
    print(f'Overall Recall: {overall_recall:.4f}')
    print('='*80 + '\n')

    ## make a scatter plot of gt_ant_count vs tracked_ant_count every frame
    # Map video labels to colonies
    video_to_colony = {
        '2024-08-22_03_01_01': 'shack', '2024-08-05_16_01_01': 'shack',
        '2024-08-22_21_01_00': 'rain', '2024-10-04_12_01_01': 'rain',
        '2024-08-09_11_01_02': 'beer', '2024-10-22_22_01_00': 'beer'
    }
    colony_colors = {'beer': 'blue', 'rain': 'green', 'shack': 'orange'}
    
    gt_count = []
    pred_count = []
    colors = []
    for f in list(gts_1.keys()):
        gt_count.append(len(gts_1[f]))
        ### detections count
        ##pred_count.append(len(preds_1[f]))
        ### tracking count
        pred_count.append(track_counts_1.get(f, 0))
        # Extract video ID from file path (assumes video ID is in the path)
        video_id = None
        for vid_id in video_to_colony.keys():
            if vid_id in f:
                video_id = vid_id
                break
        colony = video_to_colony.get(video_id, 'unknown') if video_id else 'unknown'
        colors.append(colony_colors.get(colony, 'black'))

    # Create a new figure to avoid interference from previous plots
    plt.figure()
    plt.title('ground truth vs tracked ant count per frame')
    for colony in ['beer', 'rain', 'shack']:
        mask = [c == colony_colors[colony] for c in colors]
        if any(mask):
            plt.scatter([gt_count[i] for i in range(len(gt_count)) if mask[i]], 
                       [pred_count[i] for i in range(len(pred_count)) if mask[i]], 
                       c=colony_colors[colony], label=colony, alpha=0.6)
    plt.ylabel('tracked ant count (unique ant IDs per frame)')
    plt.xlabel('ground truth ant count')
    plt.xlim(0,250)
    plt.ylim(0,250)
    plt.plot([0,50,100,150,200,250], [0,50,100,150,200,250], color='red', linestyle='--', label='y=x')
    plt.legend()
    plt.savefig('ground_truth_vs_tracked_ant_count_per_frame.svg', format='svg', bbox_inches='tight')
    #plt.show()

    plot_pr_curve(preds_1, gts_1)
