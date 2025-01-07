import sys
sys.path.append('../')
sys.path.append('../utils')
sys.path.append('../preprocessing_for_annotation')
sys.path.append('../visualization')

import json
import matplotlib.pyplot as plt
import numpy as np
import copy
import read_xml_files_from_cvat, converting_points_to_boxes
import preprocessing_for_annotation
import plots_comparing_predictions_and_gt

import cv2

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


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
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
    #plt.ylim([0,1])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('precision recall curve - ant test set')

    for iou_thresh in np.array([0.5]):
        tp_train = np.zeros(len(confidence_thrs))
        fp_train = np.zeros(len(confidence_thrs))
        fn_train = np.zeros(len(confidence_thrs))

        for i, conf_thr in enumerate(confidence_thrs):
            tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_val, gts_val, iou_thr=iou_thresh, conf_thr=conf_thr)
            

        # Plot training set PR curves
        precision_list = []
        recall_list = []

        for i in range(0,confidence_thrs.shape[0]):
            print ('###' + str(confidence_thrs[i]) + '###')
            
            precision = tp_train[i]/(tp_train[i] + fp_train[i])
            recall = tp_train[i]/(tp_train[i] + fn_train[i])
            if precision==0 and recall==0:
                continue
            print (precision, recall)

            precision_list.append(precision)
            recall_list.append(recall)

        #print (iou_thresh)
        #print (recall_list, precision_list)
        #import ipdb;ipdb.set_trace()
        plt.plot(np.array(recall_list), np.array(precision_list),c='tab:blue', label=' model trained on courtyard + ant nest data')

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


def get_metrics(annotation_xml_file, folder_where_annotated_video_came_from, box_size, label):
    ### read and convert ground truth points from cvat into boxes #####
    gts_boxes_dict = read_and_convert_ground_truth(annotation_xml_file, folder_where_annotated_video_came_from, box_size)

    #### visualize first image to see if boxes are correct and of appropriate size
    #plots_comparing_predictions_and_gt.plot_boxes_on_image(list(gts_boxes_dict.keys())[0], gts_boxes_dict[list(gts_boxes_dict.keys())[0]] , label='2024-08-22_03_01_01_ground_truth')


    ### get predictions for the images in the ground truth dict #####
    preds_boxes_dict = get_blob_detection_preds(gts_boxes_dict)
    #plots_comparing_predictions_and_gt.plot_boxes_on_image(list(preds_boxes_dict.keys())[0], preds_boxes_dict[list(preds_boxes_dict.keys())[0]], label='2024-08-22_03_01_01_prediction')


    plots_comparing_predictions_and_gt.plot_ant_counts_gt_vs_preds(gts_boxes_dict, preds_boxes_dict, label)


    tp, fp, fn = compute_counts(preds_boxes_dict, gts_boxes_dict)
    f1_score = (2*tp)/(2*tp + fp + fn)
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)

    print ('true positives, false positives, false negs:', tp,fp,fn)
    print ('f1 score is :', f1_score)
    print ('precision, recall:',precision, recall)

    return gts_boxes_dict, preds_boxes_dict



if __name__ == '__main__':
    gts_1, preds_1 = get_metrics('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/shack/2024-08-22_03_01_01.xml', '/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-22-2024/2024-08-22_03_01_01/', 20, '2024-08-22_03_01_01')
    gts_2, preds_2 = get_metrics('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/shack/2024-08-01_20_01_00.xml', '/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-22-2024/2024-08-01_20_01_00/', 20, '2024-08-01_20_01_00')
    gts_3, preds_3 = get_metrics('/home/tarun/Desktop/antcam/downloaded_annotations_from_cvat/shack/2024-08-13_11_01_01.xml', '/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-22-2024/2024-08-13_11_01_01/', 20, '2024-08-13_11_01_01')

    gts_1.update(gts_2)
    gts_1.update(gts_3)
    preds_1.update(preds_2)
    preds_1.update(preds_3)

    ## make a scatter plot of gt_ant_count vs preds_ant_count every frame
    gt_count = []
    pred_count = []
    for f in list(gts_1.keys()):
        gt_count.append(len(gts_1[f]))
        pred_count.append(len(preds_1[f]))

    plt.title('ground truth vs prediction per frame')
    plt.scatter(gt_count, pred_count, c='black')
    plt.ylabel('blob detection counts')
    plt.xlabel('ground truth counts')
    plt.xlim(0,200)
    plt.ylim(0,200)
    plt.plot([0,50,100,150,200], [0,50,100,150,200], color='red')
    plt.show()

    ##plot_pr_curve(preds_boxes_dict, gts_boxes_dict)