import database_helper
import glob
import sys
sys.path.append('../utils')
import convert_video_h264_to_mp4
import json
import datetime
import os

"""

Code to create a MySQL database, tables and to populate the tables. Populating videos table from a parent folder 
involves converting the videos to mp4, parsing through the metadata and populating the temp and humidity values and timestamp.


"""


connection = database_helper.create_connection("localhost", "root", "master")

##Create database
create_database_query = "CREATE DATABASE IF NOT EXISTS ant_colony_db"
database_helper.execute_query(connection, create_database_query)

## Connect to the new database
connection = database_helper.create_connection("localhost", "root", "master", "ant_colony_db")

## create tables -> sites, videos, and counts 
database_helper.create_tables(connection)
print ('tables created successfully')


database_helper.insert_site(connection, 1, 'beer', '34.217435, -118.15154')
database_helper.insert_site(connection, 2, 'shack', '34.217746, -118.15075')
database_helper.insert_site(connection, 3, 'rain', '34.217406, -118.15280')
database_helper.insert_site(connection, 4, 'rocks', '34.216903, -118.15495')
print ('sites table populated')


def populate_videos_table(colony, parent_folder):
	## function to parse through folder structure, extract environmental variables, and populate the videos table
	## get site_id from the DB using the colony name
	site_id = database_helper.get_site_id_from_site_name(connection, colony)

	site_id = int(site_id)
	all_folders = glob.glob(parent_folder + '/*')
	for subfolder in all_folders:
		vid_name = subfolder.split('/')[-1]
		if os.path.exists(subfolder + '/vid.h264'):
			## check if video has been converted to mp4 already, if not then do the conversion
			if not os.path.exists(subfolder + '/' + vid_name + '.mp4'):
				## convert video to mp4 
				convert_video_h264_to_mp4.convert('vid.h264', vid_name + '.mp4', path=subfolder + '/', mask=False )
				print ('converted video to mp4')

			video_id = subfolder + '/' + vid_name + '.mp4'

			## get temperature, humidity, and lux from metadata file
			if os.path.exists(subfolder + '/' + 'data.txt'):
				f = open(subfolder + '/' + 'data.txt', 'r')
				data = json.load(f)
				temperature = data['temperature']
				humidity = data['humidity']
				lux = data['light']['lux']
			else:
				print ('####### metadata file does not exist for ' + subfolder )
				temperature = humidity = lux = None

			## get the timestamp into datetime format
			date = vid_name.split('_')[0]
			time = ':'.join(vid_name.split('_')[1:])
			timestamp = date + ' ' + time
			timestamp = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")

			## insert into videos table
			database_helper.insert_video(connection, video_id, site_id, timestamp, temperature, humidity, lux)

		else:
			print (' ######### video does not exist - skipping:' + subfolder)




populate_videos_table('shack', '/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-22-2024/')