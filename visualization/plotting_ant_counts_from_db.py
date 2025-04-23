import sys
sys.path.append('../')
from mysql_dataset import database_helper
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict
import itertools
from encounters import count_encounters_per_frame
"""

code to read counts and std dev from Counts table in the DB and plot ant counts against time per day, 
average ant count per site over multiple days, 


"""

def bootstrap(data, n):
	## randomly resample n times and take the mean each time
	bootstrapped_data = np.zeros(n)
	for i in range(0,n):
		sample = np.random.choice(data, size=len(data))
		bootstrapped_data[i] = np.mean(np.array(sample))
	return bootstrapped_data


def confidence_interval(data):
	## get the 95% confidence interval by getting the 2.5th and 97.5th percentile of the data
	conf_interval = np.percentile(data,[2.5,97.5])
	#print (conf_interval)
	return conf_interval[0], conf_interval[1]


def plot_data_for_a_day_range(start_day, number_of_days=0, site_id = 1):
	df_per_site = df.loc[df.site_id == site_id]
	df_per_site_per_day = df_per_site.loc[(df_per_site.time_stamp.dt.day == start_day.start_time.day) & (df_per_site.time_stamp.dt.month == start_day.start_time.month)]
	

	hours = df_per_site_per_day['time_stamp'].dt.hour
	temperature = df_per_site_per_day['temperature']
	lux = df_per_site_per_day['LUX']
	
	detection_only_csv = df_per_site_per_day['yolo_detection_only_csv']
	tracking_with_direction_csv = df_per_site_per_day['yolo_tracking_with_direction_csv']

	ant_counts_away = []
	ant_counts_toward = []
	ant_counts = [] ## total
	
	for file in tracking_with_direction_csv.to_list():
		data = pd.read_csv(file)
		## average total ant count is total number of boxes in the video / number of frames
		ant_counts.append(len(data) / data['frame_number'].nunique())

		### away
		data_away = data[data.direction == 'away']
		ant_counts_away.append(len(data_away) / data_away['frame_number'].nunique())

		### toward
		data_toward = data[data.direction == 'toward']
		ant_counts_toward.append(len(data_toward) / data_toward['frame_number'].nunique())



	## one ant count per video
	ant_counts, ant_counts_away, ant_counts_toward = np.array(ant_counts), np.array(ant_counts_away), np.array(ant_counts_toward)
	ant_std_dev, ant_std_dev_away, ant_std_dev_toward = np.zeros_like(ant_counts), np.zeros_like(ant_counts_away), np.zeros_like(ant_counts_toward)

	fig, axes = plt.subplots(2, 1, figsize=(8, 10))
	fig.suptitle('Data for ' + str(number_of_days+1) + ' day(s) site : ' + site_mapping[site_id])

	alpha = 0.4

	if number_of_days == 0:
		## just one trace so plotting with std dev 
		alpha = 1
		axes[0].errorbar(hours, ant_counts, yerr=ant_std_dev , label=str(start_day.start_time.month) + '-' +str(start_day.start_time.day), color='b', alpha=alpha)
		axes[0].errorbar(hours, ant_counts_away, yerr=ant_std_dev_away , label=str(start_day.start_time.month) + '-' +str(start_day.start_time.day), color='g', alpha=alpha)
		axes[0].errorbar(hours, ant_counts_toward, yerr=ant_std_dev_toward , label=str(start_day.start_time.month) + '-' +str(start_day.start_time.day), color='r', alpha=alpha)

	else:
		## no std dev in these plots because then it would get too messy
		axes[0].errorbar(hours, ant_counts, label=str(start_day.start_time.month) + '-' +str(start_day.start_time.day), color='b', alpha=alpha)
	
	axes[1].plot(hours, temperature, label=str(start_day.start_time.month) + '-' + str(start_day.start_time.day), color='r', alpha=alpha)

	### for subsequent days if any
	for i in range(0, number_of_days):
		df_per_site_per_day = df_per_site.loc[(df_per_site.time_stamp.dt.day == (start_day+i).start_time.day) & (df_per_site.time_stamp.dt.month == (start_day+i).start_time.month)]

		hours = df_per_site_per_day['time_stamp'].dt.hour
		temperature = df_per_site_per_day['temperature']
		lux = df_per_site_per_day['LUX']
		

		detection_only_csv = df_per_site_per_day['yolo_detection_only_csv']
		tracking_with_direction_csv = df_per_site_per_day['yolo_tracking_with_direction_csv']

		ant_counts_away = []
		ant_counts_toward = []
		ant_counts = [] ## total
		
		for file in tracking_with_direction_csv.to_list():
			data = pd.read_csv(file)
			## average total ant count is total number of boxes in the video / number of frames
			if len(data)!=0:
				ant_counts.append(len(data) / data['frame_number'].nunique())
			else:
				ant_counts.append(0)
			### away
			data_away = data[data.direction == 'away']
			if len(data_away) !=0:
				ant_counts_away.append(len(data_away) / data_away['frame_number'].nunique())
			else:
				ant_counts_away.append(0)

			### toward
			data_toward = data[data.direction == 'toward']
			if len(data_toward) != 0:
				ant_counts_toward.append(len(data_toward) / data_toward['frame_number'].nunique())
			else:
				ant_counts_toward.append(0)


		## one ant count per video
		ant_counts, ant_counts_away, ant_counts_toward = np.array(ant_counts), np.array(ant_counts_away), np.array(ant_counts_toward)
		ant_std_dev, ant_std_dev_away, ant_std_dev_toward = np.zeros_like(ant_counts), np.zeros_like(ant_counts_away), np.zeros_like(ant_counts_toward)


		#axes[0].errorbar(hours, ant_counts, label=str((start_day+i).start_time.month) + '-' +str((start_day+i).start_time.day), color='b', alpha=alpha)
		axes[0].errorbar(hours, ant_counts_away, yerr=ant_std_dev_away , label=str(start_day.start_time.month) + '-' +str(start_day.start_time.day), color='g', alpha=alpha)
		axes[0].errorbar(hours, ant_counts_toward, yerr=ant_std_dev_toward , label=str(start_day.start_time.month) + '-' +str(start_day.start_time.day), color='r', alpha=alpha)

		axes[1].plot(hours, temperature, label=str((start_day+i).start_time.month) + '-' + str((start_day+i).start_time.day), color='r', alpha=alpha)

	axes[0].set_xticks(range(0,24)) 
	axes[0].set_ylabel('ant count')
	axes[0].set_xlabel('hour')

	axes[1].set_xticks(range(0,24)) 
	axes[1].set_ylabel('temperature')
	axes[1].set_xlabel('hour')

	plt.show()
	plt.clf()


def scatter_plot_for_day_range(start_day, number_of_days=0, site_id = 1):
	df_per_site = df.loc[df.site_id == site_id]

	counts_away_per_hour_across_days = defaultdict(list)
	counts_toward_per_hour_across_days = defaultdict(list)


	for i in range(0, number_of_days):
		## these are all the videos for that day (should be 24 videos for a full day)
		df_per_site_per_day = df_per_site.loc[(df_per_site.time_stamp.dt.day == (start_day+i).start_time.day) & (df_per_site.time_stamp.dt.month == (start_day+i).start_time.month)]

		for index, video in df_per_site_per_day.iterrows():
			
			hour = video['time_stamp'].hour
			temperature = video['temperature']
			lux = video['LUX']

			detection_only_csv = video['herdnet_detection_only_csv']
			tracking_with_direction_csv = video['herdnet_tracking_with_direction_csv']

			print (f' reading {tracking_with_direction_csv}')
			data = pd.read_csv(tracking_with_direction_csv)
			
			### away
			data_away = data[data.direction == 'away']
			if len(data_away) !=0:
				counts_away_per_hour_across_days[hour].append(len(data_away) / data_away['frame_number'].nunique())
			else:
				counts_away_per_hour_across_days[hour].append(0)

			### toward
			data_toward = data[data.direction == 'toward']
			if len(data_toward) != 0:
				counts_toward_per_hour_across_days[hour].append(len(data_toward) / data_toward['frame_number'].nunique())
			else:
				counts_toward_per_hour_across_days[hour].append(0)

	fig, axes = plt.subplots()
	x_list = []
	y_list_away, y_list_toward = [], []

	y_away_err_lower, y_away_err_upper = [], []
	y_toward_err_lower, y_toward_err_upper = [], []
	bootstrapped_means_away, bootstrapped_means_toward = [], []

	for h in range(24):
		away_values = counts_away_per_hour_across_days[h]
		y_list_away.append(away_values)
		bootstrapped_data = bootstrap(away_values, 10000)
		bootstrapped_means_away.append(np.mean(bootstrapped_data))
		conf_min, conf_max = confidence_interval(bootstrapped_data)
		y_away_err_lower.append(np.mean(bootstrapped_data) - conf_min)
		y_away_err_upper.append(conf_max - np.mean(bootstrapped_data))
		
		toward_values = counts_toward_per_hour_across_days[h]
		y_list_toward.append(toward_values)
		bootstrapped_data = bootstrap(toward_values, 10000)
		conf_min, conf_max = confidence_interval(bootstrapped_data)
		bootstrapped_means_toward.append(np.mean(bootstrapped_data))
		y_toward_err_lower.append(np.mean(bootstrapped_data) - conf_min)
		y_toward_err_upper.append(conf_max - np.mean(bootstrapped_data))

		x_list.append([h]*len(away_values))

	
	x_axis = list(itertools.chain.from_iterable(x_list))

	## add some jitter
	x_axis_away = x_axis + np.random.uniform(low=0.10, high=0.10, size=len(x_axis))
	x_axis_toward = x_axis + np.random.uniform(low=-0.10, high=-0.10, size=len(x_axis))
	
	#axes.scatter(x_axis_away, np.array(list(itertools.chain.from_iterable(y_list_away)))/np.array(list(itertools.chain.from_iterable(y_list_toward))), marker='.', c=[[0,1,0]], s=20)
	

	axes.scatter(x_axis_away, list(itertools.chain.from_iterable(y_list_away)), marker='.', c='g', s=20, alpha=0.3)
	axes.errorbar(range(24) + np.random.uniform(low=0.10, high=0.10, size=24), bootstrapped_means_away, yerr=[y_away_err_lower, y_away_err_upper], fmt=".", c='g', ecolor='k', elinewidth=1, label='away')

	axes.scatter(x_axis_toward, list(itertools.chain.from_iterable(y_list_toward)), marker='.', c='r', s=20, alpha=0.3)
	axes.errorbar(range(24) + np.random.uniform(low=-0.10, high=-0.10, size=24), bootstrapped_means_toward, yerr=[y_toward_err_lower, y_toward_err_upper], fmt=".", c='r', ecolor='k', elinewidth=1, label='toward')
	
	axes.legend()
	plt.title('Rain tree 11-15-2024 to 12-06-2024')
	plt.xticks(range(24))
	plt.xlabel('Hour')
	plt.ylabel('Mean Ant count')
	plt.show()


def scatter_plot_encounters_for_day_range(start_day, number_of_days=0, site_id = 1):
	df_per_site = df.loc[df.site_id == site_id]

	encounters_away_per_hour_across_days = defaultdict(list)
	encounters_toward_per_hour_across_days = defaultdict(list)


	for i in range(0, number_of_days):
		## these are all the videos for that day (should be 24 videos for a full day)
		df_per_site_per_day = df_per_site.loc[(df_per_site.time_stamp.dt.day == (start_day+i).start_time.day) & (df_per_site.time_stamp.dt.month == (start_day+i).start_time.month)]

		for index, video in df_per_site_per_day.iterrows():
			
			hour = video['time_stamp'].hour
			temperature = video['temperature']
			lux = video['LUX']
			

			detection_only_csv = video['yolo_detection_only_csv']
			tracking_with_direction_csv = video['yolo_tracking_with_direction_csv']

			print (f' reading {tracking_with_direction_csv}')
			data = pd.read_csv(tracking_with_direction_csv)
			data["x_center"] = (data["x1"] + data["x2"]) / 2
			data["y_center"] = (data["y1"] + data["y2"]) / 2
			
			### away
			data_away = data[data.direction == 'away']
			if len(data_away) !=0:
				encounters = count_encounters_per_frame(data_away, threshold=5)
				encounters_away_per_hour_across_days[hour].append(sum(encounters.values()) / data_away['frame_number'].nunique())
			else:
				encounters_away_per_hour_across_days[hour].append(0)

			### toward
			data_toward = data[data.direction == 'toward']
			if len(data_toward) != 0:
				encounters = count_encounters_per_frame(data_toward, threshold=5)		
				encounters_toward_per_hour_across_days[hour].append(sum(encounters.values()) / data_toward['frame_number'].nunique())
			else:
				encounters_toward_per_hour_across_days[hour].append(0)

	fig, axes = plt.subplots()
	x_list = []
	y_list_away, y_list_toward = [], []

	y_away_err_lower, y_away_err_upper = [], []
	y_toward_err_lower, y_toward_err_upper = [], []
	bootstrapped_means_away, bootstrapped_means_toward = [], []

	for h in range(24):
		away_values = encounters_away_per_hour_across_days[h]
		y_list_away.append(away_values)
		bootstrapped_data = bootstrap(away_values, 10000)
		bootstrapped_means_away.append(np.mean(bootstrapped_data))
		conf_min, conf_max = confidence_interval(bootstrapped_data)
		y_away_err_lower.append(np.mean(bootstrapped_data) - conf_min)
		y_away_err_upper.append(conf_max - np.mean(bootstrapped_data))
		
		toward_values = encounters_toward_per_hour_across_days[h]
		y_list_toward.append(toward_values)
		bootstrapped_data = bootstrap(toward_values, 10000)
		conf_min, conf_max = confidence_interval(bootstrapped_data)
		bootstrapped_means_toward.append(np.mean(bootstrapped_data))
		y_toward_err_lower.append(np.mean(bootstrapped_data) - conf_min)
		y_toward_err_upper.append(conf_max - np.mean(bootstrapped_data))

		x_list.append([h]*len(away_values))

	
	x_axis = list(itertools.chain.from_iterable(x_list))

	## add some jitter
	x_axis_away = x_axis + np.random.uniform(low=0.10, high=0.10, size=len(x_axis))
	x_axis_toward = x_axis + np.random.uniform(low=-0.10, high=-0.10, size=len(x_axis))
	
	#axes.scatter(x_axis_away, np.array(list(itertools.chain.from_iterable(y_list_away)))/np.array(list(itertools.chain.from_iterable(y_list_toward))), marker='.', c=[[0,1,0]], s=20)
	

	axes.scatter(x_axis_away, list(itertools.chain.from_iterable(y_list_away)), marker='.', c='g', s=20, alpha=0.3)
	axes.errorbar(range(24) + np.random.uniform(low=0.10, high=0.10, size=24), bootstrapped_means_away, yerr=[y_away_err_lower, y_away_err_upper], fmt=".", c='g', ecolor='k', elinewidth=1, label='away')

	axes.scatter(x_axis_toward, list(itertools.chain.from_iterable(y_list_toward)), marker='.', c='r', s=20, alpha=0.3)
	axes.errorbar(range(24) + np.random.uniform(low=-0.10, high=-0.10, size=24), bootstrapped_means_toward, yerr=[y_toward_err_lower, y_toward_err_upper], fmt=".", c='r', ecolor='k', elinewidth=1, label='toward')
	
	axes.legend()
	plt.title('Rain tree 11-15-2024 to 12-05-2024')
	plt.xticks(range(24))
	plt.xlabel('Hour')
	plt.ylabel('Avg number of encounters per video')
	plt.show()





def plot_data_by_hour():
	## days on the x axis, ant count on the y axis
	fig, axes = plt.subplots(nrows=24, ncols=2, figsize=(6, 8))
	fig.suptitle('Days vs variables per hour (each row is an hour) ')

	alpha = 1

	for h in range(0,24):
		ant_counts = df.groupby(df['time_stamp'].dt.hour)['blob_detection_average_count'].apply(list)[h]
		days = df.groupby(df['time_stamp'].dt.hour)['time_stamp'].apply(lambda x:[x.dt.day])[h][0]
		temperature = df.groupby(df['time_stamp'].dt.hour)['temperature'].apply(list)[h]

		axes[h][0].plot(days, ant_counts, color='b', alpha=alpha)
		axes[h][0].set_xticks(range(1,31))
		
		axes[h][1].plot(days, temperature, color='r', alpha=alpha)
		axes[h][1].set_xticks(range(1,31))


	axes[-1][0].set_xlabel('days of a month')
	axes[-1][1].set_xlabel('days of a month')

	plt.show()




def plot_ant_counts_vs_variables_scatter():
	## scatter plots of all ant counts vs temperature, ant counts vs humidty, ant counts vs hour
	## TODO: make these subplots
	plt.scatter(df['temperature'], df['yolo'])
	plt.xlabel('temperature')
	plt.ylabel('ant count')
	plt.show()
	plt.clf()
	plt.scatter(df['humidity'], df['yolo'])
	plt.xlabel('humidity')
	plt.ylabel('ant count')
	plt.show()
	plt.clf()
	plt.scatter(df['LUX'], df['yolo'])
	plt.xlabel('LUX')
	plt.ylabel('ant count')
	plt.show()
	plt.clf()
	plt.scatter(df['time_stamp'].dt.hour, df['yolo'])
	plt.xlabel('time')
	plt.ylabel('ant count')
	
	plt.show()



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

#days_period = pd.Period(df.time_stamp.min(), freq='D')
days_period = pd.Period('2024-11-15', freq='D')

## get the sites where we have data for the period we are in interested in
sites = set(df.loc[(df.time_stamp.dt.day == days_period.start_time.day) & (df.time_stamp.dt.month == days_period.start_time.month)].site_id)
sites = list(sites)
sites = [3]

#plot_ant_counts_vs_variables_scatter()

for site in sites:
	#plot_data_for_a_day_range(days_period, 0, site)
	#plot_data_for_a_day_range(days_period, 10, site)
	scatter_plot_for_day_range(days_period, 22, site)
	#scatter_plot_encounters_for_day_range(days_period, 23, site)

#plot_data_by_hour()



# df2 = df[['temperature','time_stamp']].set_index('time_stamp').sort_index()

# ### this resamples
# hourly_resampled = (df[['temperature','time_stamp']].set_index('time_stamp')).resample('H').mean()

# series1 = pd.Series(hourly_resampled.reset_index(drop=True).temperature)
# series2 = pd.Series(df2.reset_index(drop=True).temperature)

# ## to check for NaNs
# series1.loc[series1.isna() == True]

#import ipdb;ipdb.set_trace()

# plot_data_by_hour()




