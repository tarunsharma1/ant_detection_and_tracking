import cv2
import glob
import numpy as np

def resize_image(src, dst):
	img = cv2.imread(src)
	resized_image = cv2.resize(img, (1920, 1088))
	cv2.imwrite(dst + src.split('/')[-1].split('.')[0] + '_resized.jpg', resized_image)



def visualize_mask(img, mask):
	mask = cv2.imread(mask,0)
	center_coordinates = (1050, 600)
	ret, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

	img = cv2.imread(img,0)
	res = cv2.bitwise_and(img,img,mask = mask_bin)
	cv2.imshow('w', res)
	cv2.waitKey(0)


if __name__ == '__main__':
	resize_image('/media/tarun/Backup5TB/all_ant_data/rain-tree-10-03-2024_to_10-25-2024/2024-10-20_00_01_16/2024-10-20_00_01_16_OG_20.jpg', '/media/tarun/Backup5TB/all_ant_data/rain-tree-10-03-2024_to_10-25-2024/2024-10-20_00_01_16/')
	visualize_mask('/media/tarun/Backup5TB/all_ant_data/rain-tree-10-03-2024_to_10-25-2024/2024-10-20_00_01_16/2024-10-20_00_01_16_OG_20_resized.jpg' , '/home/tarun/Desktop/masks/2024-10-23_00_01_06_OG_20_resized.png')