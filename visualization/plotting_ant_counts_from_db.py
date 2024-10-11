import sys
sys.path.append('../')
from mysql_dataset import database_helper
import matplotlib.pyplot as plt
import pandas as pd
"""

code to read counts and std dev from Counts table in the DB and plot ant counts against time per day, 
average ant count per site over multiple days, 


"""



def plot_data_for_a_day_range(start_day, number_of_days=0):
	hours = df.loc[df.time_stamp.dt.day == start_day.start_time.day]['time_stamp'].dt.hour
	temperature = df.loc[df.time_stamp.dt.day == start_day.start_time.day]['temperature']
	ant_counts = df.loc[df.time_stamp.dt.day == start_day.start_time.day]['blob_detection_average_count']
	ant_std_dev = df.loc[df.time_stamp.dt.day == start_day.start_time.day]['blob_detection_std_dev']

	fig, axes = plt.subplots(2, 1, figsize=(8, 10))
	fig.suptitle('Data for ' + str(number_of_days+1) + ' day(s)')

	alpha = 0.4

	if number_of_days == 0:
		## just one trace so plotting with std dev 
		alpha = 1
		axes[0].errorbar(hours, ant_counts, yerr=ant_std_dev , label=str(start_day.start_time.month) + '-' +str(start_day.start_time.day), color='b', alpha=alpha)
	else:
		## no std dev in these plots because then it would get too messy
		axes[0].errorbar(hours, ant_counts, label=str(start_day.start_time.month) + '-' +str(start_day.start_time.day), color='b', alpha=alpha)
	
	axes[1].plot(hours, temperature, label=str(start_day.start_time.month) + '-' + str(start_day.start_time.day), color='r', alpha=alpha)

	### for subsequent days if any
	for i in range(0, number_of_days):
		hours = df.loc[df.time_stamp.dt.day == (start_day+i).start_time.day]['time_stamp'].dt.hour
		temperature = df.loc[df.time_stamp.dt.day == (start_day+i).start_time.day]['temperature']
		ant_counts = df.loc[df.time_stamp.dt.day == (start_day+i).start_time.day]['blob_detection_average_count']

		axes[0].errorbar(hours, ant_counts, label=str((start_day+i).start_time.month) + '-' +str((start_day+i).start_time.day), color='b', alpha=alpha)
		axes[1].plot(hours, temperature, label=str((start_day+i).start_time.month) + '-' + str((start_day+i).start_time.day), color='r', alpha=alpha)

	axes[0].set_xticks(range(0,24)) 
	axes[0].set_ylabel('ant count')
	axes[0].set_xlabel('hour')

	axes[1].set_xticks(range(0,24)) 
	axes[1].set_ylabel('temperature')
	axes[1].set_xlabel('hour')

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
	plt.scatter(df['temperature'], df['blob_detection_average_count'])
	plt.show()
	plt.scatter(df['time_stamp'].dt.hour, df['blob_detection_average_count'])
	plt.show()



connection = database_helper.create_connection("localhost", "root", "master", "ant_colony_db")
cursor = connection.cursor()
query = f"""
	select Counts.video_id, Counts.blob_detection_average_count, Counts.blob_detection_std_dev, Videos.temperature, 
	Videos.humidity, Videos.LUX, Videos.time_stamp, Videos.site_id from Counts INNER JOIN Videos on Counts.video_id=Videos.video_id;
	"""
cursor.execute(query)
table_rows = cursor.fetchall()

df = pd.DataFrame(table_rows, columns=cursor.column_names)

## make sure the time_stamp column is in pandas datetime format
df["time_stamp"] = pd.to_datetime(df["time_stamp"])

## create a pandas period object that has the frequency of day, this allows me to say day + 1, day + 2, in order to get subsequent days 
## Note : we can also do this per hour using hour = pd.Period('2022-02-09 16:00:00', freq='H')
days_period = pd.Period(df.time_stamp.min(), freq='D')


#import ipdb; ipdb.set_trace()
plot_ant_counts_vs_variables_scatter()

plot_data_for_a_day_range(days_period + 1)
plot_data_for_a_day_range(days_period, 20)


plot_data_by_hour()



# df2 = df[['temperature','time_stamp']].set_index('time_stamp').sort_index()

# ### this resamples
# hourly_resampled = (df[['temperature','time_stamp']].set_index('time_stamp')).resample('H').mean()

# series1 = pd.Series(hourly_resampled.reset_index(drop=True).temperature)
# series2 = pd.Series(df2.reset_index(drop=True).temperature)

# ## to check for NaNs
# series1.loc[series1.isna() == True]

#import ipdb;ipdb.set_trace()

# plot_data_by_hour()




