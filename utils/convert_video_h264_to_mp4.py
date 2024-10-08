import cv2
import argparse


def convert(input_vid, output, path, mask=False):

    print('using path ' + path)
    print ('creating mp4 file: ' + path + output)
    cap = cv2.VideoCapture(path + input_vid)

    # Define the codec and create VideoWriter object
    vid_out = cv2.VideoWriter(path + output, cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (1920,1080))

    
    frame_count = 0

    while(cap.isOpened()):
        # if frame_count > 60:
        #     break
        ret, frame = cap.read()
        if ret==True:
            frame_count += 1
            #frame = cv2.flip(frame,0)

            ### apply mask if needed
            if mask:
                # mask = cv2.imread('/home/tarun/Desktop/antcam/mask_gimp.png')
                # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.bitwise_and(gray,gray, mask= ~mask)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            
            vid_out.write(frame)
            
            # cv2.imshow('frame',frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
        else:
            break

    # Release everything if job is finished
    cap.release()
    vid_out.release()
    cv2.destroyAllWindows()
    print (' ###### h264 to mp4 convertion complete #####')

if __name__ == '__main__':
    path = '/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-22-2024/'

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', "--input",)    
    parser.add_argument('-o', "--output",)
    parser.add_argument('-m', "--mask")
    parser.add_argument("--path")    

    args = parser.parse_args()
    input_vid = args.input
    output = args.output

    if args.path:
        path = args.path

    convert(input_vid, output, path, args.mask)
