import sys
sys.path.append('../')
from mysql_dataset import database_helper
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict
import itertools
from encounters import count_encounters_per_frame
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

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




def statistic_test_for_differences(away_values, toward_values, hour):
	## also perform statistic test (t-test) between the away and toward for every hour
	t_statistic, p_value = stats.ttest_ind(away_values, toward_values)
	print (f'hour {hour} t_stat {t_statistic} p value {p_value}')

	stat, p = stats.shapiro(away_values)
	if p <= 0.05:
		print(f'Shapiro test away data is not normally distributed for hour {hour}')
		print (f'man whitney U : {stats.mannwhitneyu(away_values, toward_values)}')

	stat, p = stats.shapiro(toward_values)
	if p <= 0.05:
		print(f'Shapiro test toward data is not normally distributed for hour {hour}')
		print (f'man whitney U : {stats.mannwhitneyu(away_values, toward_values)}')



def get_data_from_db(start_day, number_of_days=0, site_id = 1):
	df_per_site = df.loc[df.site_id == site_id]

	counts_away_per_hour_across_days = defaultdict(list)
	counts_toward_per_hour_across_days = defaultdict(list)
	counts_loitering_per_hour_across_days = defaultdict(list)

	counts_total = defaultdict(list)

	temperature = defaultdict(list)
	humidity = defaultdict(list)
	lux = defaultdict(list)


	for i in range(0, number_of_days):
		## these are all the videos for that day (should be 24 videos for a full day)
		df_per_site_per_day = df_per_site.loc[(df_per_site.time_stamp.dt.day == (start_day+i).start_time.day) & (df_per_site.time_stamp.dt.month == (start_day+i).start_time.month)]

		for index, video in df_per_site_per_day.iterrows():
			
			hour = video['time_stamp'].hour
			
			temperature[hour].append(video['temperature'])
			humidity[hour].append(video['humidity'])
			lux[hour].append(video['LUX'])
			
			detection_only_csv = video['herdnet_detection_only_csv']
			tracking_with_direction_csv = video['herdnet_tracking_with_direction_closest_boundary_method_csv']

			
			print (f' reading {tracking_with_direction_csv}')
			data = pd.read_csv(tracking_with_direction_csv)
			
			### away
			data_away = data[data.direction == 'away']
			if len(data_away) !=0:
				counts_away_per_hour_across_days[hour].append(len(data_away) / data['frame_number'].nunique())
			else:
				counts_away_per_hour_across_days[hour].append(0)

			### toward
			data_toward = data[data.direction == 'toward']
			if len(data_toward) != 0:
				counts_toward_per_hour_across_days[hour].append(len(data_toward) / data['frame_number'].nunique())
			else:
				counts_toward_per_hour_across_days[hour].append(0)

			### loitering/unknown
			data_loitering = data[data.direction == 'unknown']
			if len(data_loitering) != 0:
				counts_loitering_per_hour_across_days[hour].append(len(data_loitering) / data['frame_number'].nunique())
			else:
				counts_loitering_per_hour_across_days[hour].append(0)

			### total
			if len(data) !=0:
				counts_total[hour].append(len(data)/ data['frame_number'].nunique())
			else:
				counts_total[hour].append(0)
	
	return counts_away_per_hour_across_days, counts_toward_per_hour_across_days, counts_loitering_per_hour_across_days, counts_total, temperature, humidity, lux


def scatter_plot_for_day_range(start_day, number_of_days=0, site_id = 1):
	counts_away_per_hour_across_days, counts_toward_per_hour_across_days, counts_loitering_per_hour_across_days, counts_total, temperature, humidity, lux = get_data_from_db(start_day, number_of_days, site_id) 
	fig, axes = plt.subplots()
	x_list = []
	y_list_away, y_list_toward, y_list_loitering = [], [], []
	y_list_total = []

	y_away_err_lower, y_away_err_upper = [], []
	y_toward_err_lower, y_toward_err_upper = [], []
	y_loitering_err_lower, y_loitering_err_upper = [], []
	y_total_err_lower, y_total_err_upper = [], []

	bootstrapped_means_away, bootstrapped_means_toward, bootstrapped_means_loitering = [], [], []
	bootstrapped_means_total = []

	for h in range(24):
		## away
		away_values = counts_away_per_hour_across_days[h]
		y_list_away.append(away_values)
		bootstrapped_data = bootstrap(away_values, 10000)
		bootstrapped_means_away.append(np.mean(bootstrapped_data))
		conf_min, conf_max = confidence_interval(bootstrapped_data)
		y_away_err_lower.append(np.mean(bootstrapped_data) - conf_min)
		y_away_err_upper.append(conf_max - np.mean(bootstrapped_data))
		
		## toward
		toward_values = counts_toward_per_hour_across_days[h]
		y_list_toward.append(toward_values)
		bootstrapped_data = bootstrap(toward_values, 10000)
		conf_min, conf_max = confidence_interval(bootstrapped_data)
		bootstrapped_means_toward.append(np.mean(bootstrapped_data))
		y_toward_err_lower.append(np.mean(bootstrapped_data) - conf_min)
		y_toward_err_upper.append(conf_max - np.mean(bootstrapped_data))

		## loitering
		loitering_values = counts_loitering_per_hour_across_days[h]
		y_list_loitering.append(loitering_values)
		bootstrapped_data = bootstrap(loitering_values, 10000)
		conf_min, conf_max = confidence_interval(bootstrapped_data)
		bootstrapped_means_loitering.append(np.mean(bootstrapped_data))
		y_loitering_err_lower.append(np.mean(bootstrapped_data) - conf_min)
		y_loitering_err_upper.append(conf_max - np.mean(bootstrapped_data))

		## total
		total_counts = counts_total[h]
		y_list_total.append(total_counts)
		bootstrapped_data = bootstrap(total_counts, 10000)
		conf_min, conf_max = confidence_interval(bootstrapped_data)
		bootstrapped_means_total.append(np.mean(bootstrapped_data))
		y_total_err_lower.append(np.mean(bootstrapped_data) - conf_min)
		y_total_err_upper.append(conf_max - np.mean(bootstrapped_data))

		x_list.append([h]*len(away_values))

		statistic_test_for_differences(away_values, toward_values, h)
	
	x_axis = list(itertools.chain.from_iterable(x_list))

	## add some jitter
	x_axis_away = x_axis + np.random.uniform(low=0.10, high=0.10, size=len(x_axis))
	x_axis_toward = x_axis + np.random.uniform(low=-0.10, high=-0.10, size=len(x_axis))
	x_axis_loitering = x_axis + np.random.uniform(low=0, high=0, size=len(x_axis))
	
	#axes.scatter(x_axis_away, np.array(list(itertools.chain.from_iterable(y_list_away)))/np.array(list(itertools.chain.from_iterable(y_list_toward))), marker='.', c=[[0,1,0]], s=20)
	
	## total counts	
	# axes.scatter(x_axis, list(itertools.chain.from_iterable(y_list_total)), marker='.', c='k', s=20, alpha=0.3)
	# axes.plot(range(24), bootstrapped_means_total, c='k', alpha=0.3)
	# if number_of_days >1:	
	# 	axes.errorbar(range(24), bootstrapped_means_total, yerr=[y_total_err_lower, y_total_err_upper], fmt=".", c='k', ecolor='k', elinewidth=1, label='total counts')

	### away vs toward counts 
	axes.scatter(x_axis_away, list(itertools.chain.from_iterable(y_list_away)), marker='.', c='g', s=20, alpha=0.3)
	axes.errorbar(range(24) + np.random.uniform(low=0.10, high=0.10, size=24), bootstrapped_means_away, yerr=[y_away_err_lower, y_away_err_upper], fmt=".", c='g', ecolor='k', elinewidth=1, label='away')

	axes.scatter(x_axis_toward, list(itertools.chain.from_iterable(y_list_toward)), marker='.', c='r', s=20, alpha=0.3)
	axes.errorbar(range(24) + np.random.uniform(low=-0.10, high=-0.10, size=24), bootstrapped_means_toward, yerr=[y_toward_err_lower, y_toward_err_upper], fmt=".", c='r', ecolor='k', elinewidth=1, label='toward')

	axes.scatter(x_axis_loitering, list(itertools.chain.from_iterable(y_list_loitering)), marker='.', c='y', s=20, alpha=0.3)
	axes.errorbar(range(24) + np.random.uniform(low=0, high=0, size=24), bootstrapped_means_loitering, yerr=[y_loitering_err_lower, y_loitering_err_upper], fmt=".", c='y', ecolor='k', elinewidth=1, label='unknown')
	
	axes.legend()
	plt.title('Rain tree 11-15-2024 to 12-06-2024')
	plt.xticks(range(24))
	plt.xlabel('Hour')
	plt.ylabel('Mean Ant count')
	plt.show()


def scatter_plot_by_direction_aggregated(start_day, number_of_days=0, site_id=1):
	"""
	Plot (away - toward) difference for every hour across all days.
	X-axis: hours (0-23)
	Y-axis: (away - toward) difference
	Shows all individual values with alpha and bootstrapped mean with 95% CI.
	"""
	counts_away_per_hour_across_days, counts_toward_per_hour_across_days, counts_loitering_per_hour_across_days, counts_total, temperature, humidity, lux = get_data_from_db(start_day, number_of_days, site_id)
	
	# Calculate (away - toward) difference for each hour across all days
	difference_per_hour = {}  # {hour: [differences]}
	bootstrapped_means = []
	err_lower = []
	err_upper = []
	hours = []
	
	for h in range(24):
		away_values = counts_away_per_hour_across_days[h]
		toward_values = counts_toward_per_hour_across_days[h]
		
		# Calculate difference for each day at this hour
		# Ensure both lists have the same length
		min_len = min(len(away_values), len(toward_values))
		differences = [away_values[i] - toward_values[i] for i in range(min_len)]
		
		if len(differences) > 0:
			difference_per_hour[h] = differences
			hours.append(h)
			
			# Bootstrap with 10000 samples
			bootstrapped_data = bootstrap(differences, 10000)
			bootstrapped_mean = np.mean(bootstrapped_data)
			conf_min, conf_max = confidence_interval(bootstrapped_data)
			
			bootstrapped_means.append(bootstrapped_mean)
			err_lower.append(max(0, bootstrapped_mean - conf_min))
			err_upper.append(max(0, conf_max - bootstrapped_mean))
	
	# Create plot
	fig, axes = plt.subplots()
	
	# Plot all individual values with alpha and jitter
	for h in hours:
		differences = difference_per_hour[h]
		jitter = np.random.uniform(low=-0.15, high=0.15, size=len(differences))
		axes.scatter([h] * len(differences) + jitter, differences, 
		            marker='.', c='steelblue', s=20, alpha=0.3)
	
	# Plot bootstrapped means with error bars
	axes.errorbar(hours, bootstrapped_means, 
	             yerr=[err_lower, err_upper], 
	             fmt='o', c='steelblue', ecolor='k', elinewidth=2, capsize=5, capthick=2, 
	             markersize=8, label='Mean ± 95% CI', zorder=10)
	
	# Add horizontal line at y=0 (no difference)
	axes.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
	
	# Formatting
	axes.set_xticks(range(0, 24, 1))
	axes.set_xlabel('Hour of Day')
	axes.set_ylabel('(Away - Toward) Difference')
	
	# Get site name for title
	site_name = site_mapping.get(site_id, f'site_{site_id}')
	axes.set_title(f'{site_name.title()} Tree - (Away - Toward) Difference by Hour\n{start_day} to {start_day + number_of_days - 1}')
	
	axes.legend(loc='best')
	axes.grid(True, alpha=0.3)
	
	plt.tight_layout()
	plt.show()


def scatter_plot_cumulative_difference(start_day, number_of_days=0, site_id=1):
	"""
	Plot aggregated (away - toward) difference values across all hours and days.
	X-axis: 2 positions (away, toward) showing the difference values
	Y-axis: (away - toward) difference
	Shows all individual difference values with alpha and bootstrapped mean with 95% CI.
	"""
	counts_away_per_hour_across_days, counts_toward_per_hour_across_days, counts_loitering_per_hour_across_days, counts_total, temperature, humidity, lux = get_data_from_db(start_day, number_of_days, site_id)
	
	# Save counts data as pickle file
	import pickle
	import os
	site_name = site_mapping.get(site_id, f'site_{site_id}')
	end_day = start_day + number_of_days - 1
	bout_name = f"{site_name}_{str(start_day)}_{str(end_day)}"
	
	save_dir = '/home/tarun/Desktop/plots_for_committee_meeting/velocity_analysis'
	os.makedirs(save_dir, exist_ok=True)
	
	pickle_path = os.path.join(save_dir, f"counts_{bout_name}.pkl")
	with open(pickle_path, 'wb') as f:
		pickle.dump({
			'counts_away_per_hour_across_days': counts_away_per_hour_across_days,
			'counts_toward_per_hour_across_days': counts_toward_per_hour_across_days,
			'start_day': start_day,
			'number_of_days': number_of_days,
			'site_id': site_id
		}, f)
	print(f"✅ Saved counts data to {pickle_path}")
	
	# Calculate (away - toward) difference for each hour-day combination
	all_differences = []
	
	for h in range(24):
		away_values = counts_away_per_hour_across_days[h]
		toward_values = counts_toward_per_hour_across_days[h]
		
		# Calculate difference for each day at this hour
		min_len = min(len(away_values), len(toward_values))
		for i in range(min_len):
			diff = away_values[i] - toward_values[i]
			all_differences.append(diff)
	
	if len(all_differences) == 0:
		print("No data available for plotting")
		return
	
	# Bootstrap the aggregated differences
	bootstrapped_data = bootstrap(all_differences, 10000)
	bootstrapped_mean = np.mean(bootstrapped_data)
	conf_min, conf_max = confidence_interval(bootstrapped_data)
	err_lower = max(0, bootstrapped_mean - conf_min)
	err_upper = max(0, conf_max - bootstrapped_mean)

	print (f'bootstrapped mean is {bootstrapped_mean} and confidence interval is {conf_min} to {conf_max}')
	print (f'error lower is {err_lower} and error upper is {err_upper}')
	
	# Create plot
	fig, axes = plt.subplots()
	
	# X-axis position (single column)
	x_position = 0
	x_label = '(Away - Toward)'
	
	# Add jitter to x position for scatter plot
	jitter = np.random.uniform(low=-0.15, high=0.15, size=len(all_differences))
	
	# Plot all individual difference values with alpha
	axes.scatter(x_position + jitter, all_differences, marker='.', c='steelblue', s=20, alpha=0.3)
	
	# Plot bootstrapped mean with error bars
	axes.errorbar(x_position, bootstrapped_mean, 
	             yerr=[[err_lower], [err_upper]], 
	             fmt='o', c='steelblue', ecolor='k', elinewidth=2, capsize=5, capthick=2, 
	             markersize=8, label='Mean ± 95% CI', zorder=10)
	
	# Add horizontal line at y=0 (no difference)
	axes.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
	
	# Formatting
	axes.set_xticks([x_position])
	axes.set_xticklabels([x_label])
	axes.set_xlabel('Direction')
	axes.set_ylabel('(Away - Toward) Difference\n(aggregated across all hours and days)')
	
	# Get site name for title
	site_name = site_mapping.get(site_id, f'site_{site_id}')
	axes.set_title(f'{site_name.title()} Tree - Aggregated (Away - Toward) Difference\n{start_day} to {start_day + number_of_days - 1}')
	
	axes.legend(loc='best')
	axes.grid(True, alpha=0.3)
	
	plt.tight_layout()
	plt.show()


def plot_all_bouts_cumulative_difference(save_dir='/home/tarun/Desktop/plots_for_committee_meeting/velocity_analysis'):
	"""
	Read all pickle files saved by scatter_plot_cumulative_difference() and plot
	the aggregated (away - toward) difference values for all bouts on the same plot.
	
	X-axis: One position per bout (7 positions for 7 pickle files)
	Y-axis: (Away - Toward) difference
	Shows all individual difference values with alpha and bootstrapped mean with 95% CI for each bout.
	
	Parameters
	----------
	save_dir : str
		Directory containing the pickle files (default: '/home/tarun/Desktop/plots_for_committee_meeting/velocity_analysis')
	"""
	import pickle
	import os
	import glob
	
	# Find all pickle files matching the pattern
	pickle_pattern = os.path.join(save_dir, 'counts_*.pkl')
	pickle_files = sorted(glob.glob(pickle_pattern))
	

	
	# Store data for each bout
	bout_data = []
	
	# Load and process each pickle file
	for pickle_path in pickle_files:
		with open(pickle_path, 'rb') as f:
			data = pickle.load(f)
		
		counts_away_per_hour_across_days = data['counts_away_per_hour_across_days']
		counts_toward_per_hour_across_days = data['counts_toward_per_hour_across_days']
		start_day = data.get('start_day', 'Unknown')
		number_of_days = data.get('number_of_days', 0)
		site_id = data.get('site_id', 0)
		
		# Calculate (away - toward) difference for each hour-day combination
		all_differences = []
		
		for h in range(24):
			if h in counts_away_per_hour_across_days and h in counts_toward_per_hour_across_days:
				away_values = counts_away_per_hour_across_days[h]
				toward_values = counts_toward_per_hour_across_days[h]
				
				# Calculate difference for each day at this hour
				min_len = min(len(away_values), len(toward_values))
				for i in range(min_len):
					diff = away_values[i] - toward_values[i]
					all_differences.append(diff)
		
		if len(all_differences) == 0:
			print(f"Warning: No data in {os.path.basename(pickle_path)}")
			continue
		
		# Bootstrap the aggregated differences
		bootstrapped_data = bootstrap(all_differences, 10000)
		bootstrapped_mean = np.mean(bootstrapped_data)
		conf_min, conf_max = confidence_interval(bootstrapped_data)
		err_lower = max(0, bootstrapped_mean - conf_min)
		err_upper = max(0, conf_max - bootstrapped_mean)
		
		# Create bout label
		site_name = site_mapping.get(site_id, f'site_{site_id}')
		bout_label = f"{site_name}\n{start_day}"
		
		bout_data.append({
			'all_differences': all_differences,
			'bootstrapped_mean': bootstrapped_mean,
			'err_lower': err_lower,
			'err_upper': err_upper,
			'bout_label': bout_label,
			'site_id': site_id,
			'start_day': start_day
		})
		
		print(f"Processed {os.path.basename(pickle_path)}: {len(all_differences)} differences, mean={bootstrapped_mean:.4f}")
	
	if len(bout_data) == 0:
		print("No valid data to plot")
		return
	
	# Create plot
	fig, axes = plt.subplots(figsize=(max(8, len(bout_data) * 1.0), 6))
	
	# X-axis positions (one per bout)
	x_positions = np.arange(len(bout_data))
	
	# Plot each bout
	for idx, bout in enumerate(bout_data):
		x_pos = x_positions[idx]
		all_differences = bout['all_differences']
		bootstrapped_mean = bout['bootstrapped_mean']
		err_lower = bout['err_lower']
		err_upper = bout['err_upper']
		
		# Add jitter to x position for scatter plot
		jitter = np.random.uniform(low=-0.1, high=0.1, size=len(all_differences))
		
		# Plot all individual difference values in grey
		axes.scatter(x_pos + jitter, all_differences, marker='.', c='grey', s=20, alpha=0.3)
		
		# Plot bootstrapped mean with error bars in black
		axes.errorbar(x_pos, bootstrapped_mean, 
		             yerr=[[err_lower], [err_upper]], 
		             fmt='o', c='black', ecolor='black', elinewidth=1, capsize=5, capthick=1, 
		             markersize=8, zorder=10)
	
	# Add horizontal line at y=0 (no difference)
	axes.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=1)
	
	# Formatting
	axes.set_xticks(x_positions)
	axes.set_xticklabels([bout['bout_label'] for bout in bout_data], rotation=45, ha='right', fontsize=9)
	axes.set_xlabel('Bout', fontsize=11)
	axes.set_ylabel('(Away - Toward) Difference\n(aggregated across all hours and days)', fontsize=11)
	axes.set_title('Aggregated (Away - Toward) Difference Across All Bouts', fontsize=12, fontweight='bold')
	#axes.grid(True, alpha=0.3)
	
	plt.tight_layout()
	plt.show()


def linear_regression():
	'''
	  All data pooled together, linear regression of ant counts as independent variable, dependent variables being temperature, humidity, time etc.
	'''
	all_ant_counts = []
	all_temperatures = []
	all_humidity = []
	all_lux = []
	all_hours = []
	all_site_ids = []
	all_months = []

	data_collected = [(pd.Period('2024-08-22', freq='D'), 12, 3), (pd.Period('2024-10-03', freq='D'), 17, 3), (pd.Period('2024-11-15', freq='D'), 22, 3), (pd.Period('2024-08-01', freq='D'), 10, 1), (pd.Period('2024-10-22', freq='D'), 12, 1), (pd.Period('2024-08-01', freq='D'), 26, 2), (pd.Period('2024-08-26', freq='D'), 24, 2)]
	
	for (start_day, number_of_days, site_id) in data_collected:
		counts_away_per_hour_across_days, counts_toward_per_hour_across_days, counts_loitering_per_hour_across_days, counts_total, temperature, humidity, lux = get_data_from_db(start_day, number_of_days, site_id)
		
		for hour in range(24):
			all_ant_counts.extend(counts_total[hour])
			all_temperatures.extend(temperature[hour])
			all_humidity.extend(humidity[hour])
			all_lux.extend(lux[hour])
			all_hours.extend([hour]*len(counts_total[hour]))
			### below are our random effects (categorical variables)
			all_site_ids.extend([site_id]*len(counts_total[hour]))
			all_months.extend([start_day.month] * len(counts_total[hour]))


	df_linear_reg = pd.DataFrame({'ant_count': all_ant_counts, 'temperature': all_temperatures, 'humidity': all_humidity, 'lux':all_lux, 'time':all_hours, 'site':all_site_ids, 'month': all_months})
	
	df_linear_reg['site'] = df_linear_reg['site'].astype('category')
	df_linear_reg['month'] = df_linear_reg['month'].astype('category')
	
	df_linear_reg['log_ant_count'] = np.log(df_linear_reg['ant_count'])

	## convert time to cyclic variables to avoid jump between 23:00 and 0:00
	df_linear_reg['time_sin'] = np.sin(2 * np.pi * df_linear_reg['time'] / 24)
	df_linear_reg['time_cos'] = np.cos(2 * np.pi * df_linear_reg['time'] / 24)

	
	# model = smf.mixedlm("log_ant_count ~ time_sin + time_cos + temperature + humidity + lux", 
	# 	df_linear_reg, 
	# 	groups=df_linear_reg["site"])
	# 	#re_formula="~temperature")

	# result = model.fit(reml=False)
	# print(result.summary())


	### before fitting lets standardize predictors 
	predictors = ['time_sin', 'time_cos', 'temperature', 'humidity', 'lux']

	# Initialize scaler
	from sklearn.preprocessing import StandardScaler
	scaler = StandardScaler()
	# Fit scaler and transform predictors
	df_linear_reg_std = df_linear_reg.copy()
	df_linear_reg_std[predictors] = scaler.fit_transform(df_linear_reg_std[predictors])

	# Now fit mixed effects model with standardized predictors
	import statsmodels.formula.api as smf
	model = smf.mixedlm("log_ant_count ~ time_sin + time_cos + temperature + humidity + lux", df_linear_reg_std, groups=df_linear_reg_std["site"])
	result = model.fit(reml=False)
	print(result.summary())


	### calculate AIC/BIC
	log_likelihood = result.llf
	n1 = result.nobs
	k = result.df_modelwc + 1
	aic = 2*k - 2*log_likelihood
	bic = np.log(n1)*k - 2*log_likelihood
	print (f' AIC is {aic} and BIC is {bic}')


	### plot residuals
	plt.figure(figsize=(8,6))
	fitted = result.fittedvalues
	residuals = result.resid
	sns.scatterplot(x=fitted, y=residuals)
	plt.axhline(0, linestyle='--', color='red')
	plt.show()


	## check for correlations and multicollinearity
	predictors = df_linear_reg_std[['temperature', 'humidity', 'lux', 'time_sin', 'time_cos']]
	corr = predictors.corr()
	print (corr)
	X = sm.add_constant(predictors)
	# Calculate VIF for each predictor
	vif = pd.DataFrame()
	vif['Variable'] = X.columns
	vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

	print(vif)

	## plot model predictions against actual values
	fitted_values = result.fittedvalues
	actual_values = df_linear_reg_std['log_ant_count']
	sns.regplot(x=actual_values, y=fitted_values, scatter_kws={'alpha':0.5}, line_kws={'color':'orange'})
	plt.xlabel('Actual Values')
	plt.ylabel('Predicted Values')
	plt.title('Predicted vs Actual Values')
	plt.show()

	import ipdb;ipdb.set_trace()




def scatter_plot_encounters_for_day_range(start_day, number_of_days=0, site_id = 1):
	df_per_site = df.loc[df.site_id == site_id]

	encounters_away_per_hour_across_days = defaultdict(list)
	encounters_toward_per_hour_across_days = defaultdict(list)
	encounters_loitering_per_hour_across_days = defaultdict(list)


	for i in range(0, number_of_days):
		## these are all the videos for that day (should be 24 videos for a full day)
		df_per_site_per_day = df_per_site.loc[(df_per_site.time_stamp.dt.day == (start_day+i).start_time.day) & (df_per_site.time_stamp.dt.month == (start_day+i).start_time.month)]

		for index, video in df_per_site_per_day.iterrows():
			
			hour = video['time_stamp'].hour
			temperature = video['temperature']
			lux = video['LUX']
			

			detection_only_csv = video['herdnet_detection_only_csv']
			tracking_with_direction_csv = video['herdnet_tracking_with_direction_closest_boundary_method_csv']

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

			### loitering
			data_loitering = data[data.direction == 'unknown']
			if len(data_loitering) != 0:
				encounters = count_encounters_per_frame(data_loitering, threshold=5)		
				encounters_loitering_per_hour_across_days[hour].append(sum(encounters.values()) / data_toward['frame_number'].nunique())
			else:
				encounters_loitering_per_hour_across_days[hour].append(0)

	fig, axes = plt.subplots()
	x_list = []
	y_list_away, y_list_toward, y_list_loitering = [], [], []

	y_away_err_lower, y_away_err_upper = [], []
	y_toward_err_lower, y_toward_err_upper = [], []
	y_loitering_err_lower, y_loitering_err_upper = [], []
	
	bootstrapped_means_away, bootstrapped_means_toward, bootstrapped_means_loitering = [], [], []

	for h in range(24):
		## away
		away_values = encounters_away_per_hour_across_days[h]
		y_list_away.append(away_values)
		bootstrapped_data = bootstrap(away_values, 10000)
		bootstrapped_means_away.append(np.mean(bootstrapped_data))
		conf_min, conf_max = confidence_interval(bootstrapped_data)
		y_away_err_lower.append(np.mean(bootstrapped_data) - conf_min)
		y_away_err_upper.append(conf_max - np.mean(bootstrapped_data))
		
		## toward
		toward_values = encounters_toward_per_hour_across_days[h]
		y_list_toward.append(toward_values)
		bootstrapped_data = bootstrap(toward_values, 10000)
		conf_min, conf_max = confidence_interval(bootstrapped_data)
		bootstrapped_means_toward.append(np.mean(bootstrapped_data))
		y_toward_err_lower.append(np.mean(bootstrapped_data) - conf_min)
		y_toward_err_upper.append(conf_max - np.mean(bootstrapped_data))

		## loitering/unknown
		loitering_values = encounters_loitering_per_hour_across_days[h]
		y_list_loitering.append(loitering_values)
		bootstrapped_data = bootstrap(loitering_values, 10000)
		conf_min, conf_max = confidence_interval(bootstrapped_data)
		bootstrapped_means_loitering.append(np.mean(bootstrapped_data))
		y_loitering_err_lower.append(np.mean(bootstrapped_data) - conf_min)
		y_loitering_err_upper.append(conf_max - np.mean(bootstrapped_data))

		x_list.append([h]*len(away_values))

		statistic_test_for_differences(away_values, toward_values, h)


	
	x_axis = list(itertools.chain.from_iterable(x_list))

	## add some jitter
	x_axis_away = x_axis + np.random.uniform(low=0.10, high=0.10, size=len(x_axis))
	x_axis_toward = x_axis + np.random.uniform(low=-0.10, high=-0.10, size=len(x_axis))
	x_axis_loitering = x_axis + np.random.uniform(low=-0.10, high=-0.10, size=len(x_axis))
	
	#axes.scatter(x_axis_away, np.array(list(itertools.chain.from_iterable(y_list_away)))/np.array(list(itertools.chain.from_iterable(y_list_toward))), marker='.', c=[[0,1,0]], s=20)
	

	axes.scatter(x_axis_away, list(itertools.chain.from_iterable(y_list_away)), marker='.', c='g', s=20, alpha=0.3)
	axes.errorbar(range(24) + np.random.uniform(low=0.10, high=0.10, size=24), bootstrapped_means_away, yerr=[y_away_err_lower, y_away_err_upper], fmt=".", c='g', ecolor='k', elinewidth=1, label='away')

	axes.scatter(x_axis_toward, list(itertools.chain.from_iterable(y_list_toward)), marker='.', c='r', s=20, alpha=0.3)
	axes.errorbar(range(24) + np.random.uniform(low=-0.10, high=-0.10, size=24), bootstrapped_means_toward, yerr=[y_toward_err_lower, y_toward_err_upper], fmt=".", c='r', ecolor='k', elinewidth=1, label='toward')
	
	axes.scatter(x_axis_loitering, list(itertools.chain.from_iterable(y_list_loitering)), marker='.', c='y', s=20, alpha=0.3)
	axes.errorbar(range(24) + np.random.uniform(low=-0.10, high=-0.10, size=24), bootstrapped_means_loitering, yerr=[y_loitering_err_lower, y_loitering_err_upper], fmt=".", c='y', ecolor='k', elinewidth=1, label='loitering/unknown')

	axes.legend()
	plt.title('Beer tree 08-01-2024 to 08-10-2024')
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



parameters = {'axes.labelsize':8,'axes.titlesize':8, 'xtick.labelsize':8, 'font.family':"sans-serif", 'font.sans-serif':['Arial'], 'font.size':8, 'svg.fonttype':'none'}
plt.rcParams.update(parameters)


connection = database_helper.create_connection("localhost", "root", "master", "ant_colony_db")
cursor = connection.cursor()
query = f"""
	select Counts.video_id, Counts.yolo_detection_only_csv, Counts.yolo_tracking_with_direction_csv, Counts.herdnet_detection_only_csv, Counts.herdnet_tracking_with_direction_closest_boundary_method_csv,
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
days_period = pd.Period('2024-08-01', freq='D')

## get the sites where we have data for the period we are in interested in
sites = set(df.loc[(df.time_stamp.dt.day == days_period.start_time.day) & (df.time_stamp.dt.month == days_period.start_time.month)].site_id)
sites = list(sites)
sites = [1]


linear_regression()
plot_all_bouts_cumulative_difference()

for site in sites:
	#scatter_plot_for_day_range(days_period, 12, site)
	#scatter_plot_by_direction_aggregated(days_period, 10, site)
	scatter_plot_cumulative_difference(days_period, 10, site)
	#scatter_plot_encounters_for_day_range(days_period, 12, site)




# df2 = df[['temperature','time_stamp']].set_index('time_stamp').sort_index()

# ### this resamples
# hourly_resampled = (df[['temperature','time_stamp']].set_index('time_stamp')).resample('H').mean()

# series1 = pd.Series(hourly_resampled.reset_index(drop=True).temperature)
# series2 = pd.Series(df2.reset_index(drop=True).temperature)

# ## to check for NaNs
# series1.loc[series1.isna() == True]

#import ipdb;ipdb.set_trace()

# plot_data_by_hour()




