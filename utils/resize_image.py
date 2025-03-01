import cv2
import glob
import numpy as np

def resize_image(src, dst):
	img = cv2.imread(src)
	resized_image = cv2.resize(img, (1920, 1088))
	cv2.imwrite(dst + src.split('/')[-1].split('.')[0] + '_resized.jpg', resized_image)


def average_plot():
	img_paths =  glob.glob('/home/tarun/Desktop/ant_density_plots/*.png')
	avg_img = np.float32(cv2.imread(img_paths[0]))
	for img_path in img_paths:
		img = cv2.imread(img_path)
		avg_img += img
	avg_img = avg_img/len(img_paths)
	avg_img = np.uint8(avg_img)
	cv2.imshow('w', avg_img)
	cv2.waitKey(0)



if __name__ == '__main__':
	resize_image('/media/tarun/Backup5TB/all_ant_data/rain-tree-08-22-2024_to_09-02-2024/2024-08-22_21_01_00/2024-08-22_21_01_00_100.jpg', '/media/tarun/Backup5TB/all_ant_data/rain-tree-08-22-2024_to_09-02-2024/2024-08-22_21_01_00/')
	#average_plot()