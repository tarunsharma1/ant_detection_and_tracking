# ant_detection_and_tracking
Code for monitoring velvety tree ant on multiple different colonies by deploying multiple ethocams. Includes entire data pipeline of preprocessing data for annotations, data management using mySQL database, computer vision analysis of video data, plotting and visualizing results and more. 

Compared blob detection, YOLO, YOLO trained on patches followed by inference using SAHI, and HerdNet for detection. Evaluated detection using PR curves and best F1 scores on validation set.
Also evaluated after applying custom masks (to block out colony entrances) on annotations and predictions.   

TODO: visualize errors (fps and fns) to see if there are annotation mistakes or some patterns.
TODO: Try DETR.

Used SORT tracking and evaluated using TrackEval. 

TODO: hyperparameter grid search of SORT params using detections from HerdNet comparing HOTA scores.

Visualized plots of overall ant counts vs hour for different batches of data collected, ant count vs temperature, tracked ant direction (away vs toward) against hour, spatial heatmap of ant activity across a video to visualize trails, and circular histograms of ant vectors at various points in time. 

TODO: Finish work on ant encounters.  
