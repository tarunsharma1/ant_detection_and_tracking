import matplotlib.pyplot as plt
import numpy as np


def plot_boxes_on_image(image, list_of_boxes, label='image'):
    frame = cv2.imread(image)
    for box in list_of_boxes:
        frame = cv2.rectangle(frame,(box[0],box[1]),(box[2],box[3]),(0,255,0),1)
    #cv2.imshow('w', frame)
    #cv2.waitKey(1000)
    cv2.imwrite('/home/tarun/Downloads/'+label+'.jpg', frame)


def plot_ant_counts_gt_vs_preds(gts_boxes_dict, preds_boxes_dict, label= ''):
    ### we know that the dicts will contain a sequence of frames
    count_gt = []
    count_pred = []
    frame_number = []

    for file in gts_boxes_dict:
        count_gt.append(len(gts_boxes_dict[file]))
        count_pred.append(len(preds_boxes_dict[file]))

        frame_number.append(float(file.split('.')[0].split('_')[-1]))

    plt.plot(frame_number, count_gt, 'g', label='ground truth')
    plt.plot(frame_number, count_pred, 'b', label='blob detection')
    plt.xlabel('frame number')
    plt.ylabel('counts')
    #plt.ylim(0,50)
    plt.title('ant counts for '+ label)
    plt.legend()
    
    print ('mean of ant counts over all frames ground truth',np.mean(np.array(count_gt))) 
    print ('std dev of ant counts over all frames ground truth',np.std(np.array(count_gt))) 
    
    print ('mean of ant counts over all frames predictions',np.mean(np.array(count_pred)))
    print ('std dev of ant counts over all frames predictions', np.std(np.array(count_pred)))


    plt.show()
