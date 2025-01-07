import os
import glob

path = '/home/tarun/temp/'
f = glob.glob(path + '/*:*')

for i in range(0, len(f)):
	folder_name = f[i].split('/')[-1]
	os.rename(path + folder_name,path + folder_name.split("_")[0] + '_' + folder_name.split('_')[1].split(':')[0] + '_' + folder_name.split('_')[1].split(':')[1] + '_'  + folder_name.split('_')[1].split(':')[2])