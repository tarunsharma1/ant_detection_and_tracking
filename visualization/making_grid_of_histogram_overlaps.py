import sys
sys.path.append('../')
from mysql_dataset import database_helper
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict
import itertools
import circular_histogram_ant_counts








#euclidean_distance_between_two_csvs



def grid_of_hours_for_a_given_day(start_day, number_of_hours=24, site_id = 1):
	'''
		calculate histogram overlap score of tracking file from every hour to every other hour for the same day and make a 24 x 24 symmetric grid. 
		this function calculates and returns this grid
	'''
	grid = np.zeros((24,24))

	df_per_site = df.loc[df.site_id == site_id]

	## these should be all the videos for a given start_day from a given site. Should be 24 videos
	df_per_site_per_day = df_per_site.loc[(df_per_site.time_stamp.dt.day == (start_day).start_time.day) & (df_per_site.time_stamp.dt.month == (start_day).start_time.month)]

	for i in range(0, number_of_hours):
		df_per_site_per_day_per_hour = df_per_site_per_day.loc[df_per_site_per_day.time_stamp.dt.hour == i]
		
		if len(df_per_site_per_day_per_hour) == 0:
			continue


		temperature1 = df_per_site_per_day_per_hour['temperature'].item()
		lux1 = df_per_site_per_day_per_hour['LUX'].item()
		tracking_with_direction_csv1 = df_per_site_per_day_per_hour['herdnet_tracking_with_direction_csv'].item()

		for k in range(0, number_of_hours):
			df_per_site_per_day_per_hour = df_per_site_per_day.loc[df_per_site_per_day.time_stamp.dt.hour == k]
			
			if len(df_per_site_per_day_per_hour) == 0:
				continue

			temperature2 = df_per_site_per_day_per_hour['temperature'].item()
			lux2 = df_per_site_per_day_per_hour['LUX'].item()
			tracking_with_direction_csv2 = df_per_site_per_day_per_hour['herdnet_tracking_with_direction_csv'].item()

			grid[i,k] = circular_histogram_ant_counts.euclidean_distance_between_two_csvs(tracking_with_direction_csv1, tracking_with_direction_csv2)



	return grid





'''

to do: histogram similarity value between a video at every hour and video at the same hour on subsequent days? to see how much trails have changed across days.
maybe it can a separate grid for every hour. Lets say we pick day 1 8am, then grid can be 24 (hours) x number_of_days. So first row will be how does day 1 8am compare to 
midnight of day1, day2, ...dayN. Second row will be how does day 1 8am compare to 1am of day1, day2, ...,dayN and so on. We will have 24 such grids one for every hour. 

'''















connection = database_helper.create_connection("localhost", "root", "master", "ant_colony_db")
cursor = connection.cursor()
query = f"""
	select Counts.video_id, Counts.yolo_detection_only_csv, Counts.yolo_tracking_with_direction_csv, Counts.herdnet_detection_only_csv, Counts.herdnet_tracking_with_direction_csv,
	Videos.temperature, Videos.humidity, Videos.LUX, Videos.time_stamp, Videos.site_id from Counts INNER JOIN Videos on Counts.video_id=Videos.video_id;
	"""
cursor.execute(query)
table_rows = cursor.fetchall()

site_mapping = {1:'beer', 2:'shack', 3:'rain'}

df = pd.DataFrame(table_rows, columns=cursor.column_names)

## make sure the time_stamp column is in pandas datetime format
df["time_stamp"] = pd.to_datetime(df["time_stamp"])

## create a pandas period object that has the frequency of day, this allows me to say day + 1, day + 2, in order to get subsequent days 
## Note : we can also do this per hour using hour = pd.Period('2022-02-09 16:00:00', freq='H')

#hours_period = pd.Period('2024-08-25', freq='H')
days_period = pd.Period('2024-10-03', freq='D')

## get the sites where we have data for the period we are in interested in
sites = [3]

#plot_ant_counts_vs_variables_scatter()

for i in range(0,10):
	#plot_data_for_a_day_range(days_period, 0, site)
	#plot_data_for_a_day_range(days_period, 10, site)
	grid = grid_of_hours_for_a_given_day(days_period + i, 24, sites[0])
	
	plt.figure(figsize=(4, 3), dpi=160)
	plt.imshow(grid)
	plt.title('rain tree day 10-0' + str(3+i)+'-2024')
	plt.colorbar()
	#plt.show()
	plt.savefig('/home/tarun/Desktop/plots_for_committee_meeting/histogram_similarity_grids/rain_grid_10-0'+str(3+i)+'-2024.png', dpi=160)



# df2 = df[['temperature','time_stamp']].set_index('time_stamp').sort_index()

# ### this resamples
# hourly_resampled = (df[['temperature','time_stamp']].set_index('time_stamp')).resample('H').mean()

# series1 = pd.Series(hourly_resampled.reset_index(drop=True).temperature)
# series2 = pd.Series(df2.reset_index(drop=True).temperature)

# ## to check for NaNs
# series1.loc[series1.isna() == True]

#import ipdb;ipdb.set_trace()

# plot_data_by_hour()