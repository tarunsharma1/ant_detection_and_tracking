
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

def count_encounters_per_frame(df, threshold=10):
    encounter_counts = {}

    for frame, frame_data in df.groupby("frame_number"):
        # Extract center positions
        coords = frame_data[["x_center", "y_center"]].values
        
        # Compute pairwise distances
        distances = squareform(pdist(coords))
        
        # Count pairs where distance is below threshold (excluding self-comparisons)
        encounter_matrix = (distances < threshold) & (distances > 0)
        encounter_count = np.sum(encounter_matrix) // 2  # Each pair is counted twice

        encounter_counts[frame] = encounter_count
        #print (f'frame : {frame}, num encounters : {encounter_count}')

    return encounter_counts



if __name__ == '__main__':
	path = '/media/tarun/Backup5TB/all_ant_data/beer-tree-08-01-2024_to_08-10-2024/2024-08-01_21_01_00'
	video = path + '/' + path.split('/')[-1] + '.mp4'

	tracking_csv = path + '/' + path.split('/')[-1] + '_yolo_tracking_with_direction.csv'

	df = pd.read_csv(tracking_csv)

	df["x_center"] = (df["x1"] + df["x2"]) / 2
	df["y_center"] = (df["y1"] + df["y2"]) / 2

	df_away = df[df['direction'] == 'away']
	df_toward = df[df['direction'] == 'toward']


	threshold = 10
	encounters_away = count_encounters_per_frame(df_away, threshold)
	encounters_toward = count_encounters_per_frame(df_toward, threshold)


	print (f'num encounters away: {sum(encounters_away.values())}')
	print (f'num encounters toward: {sum(encounters_toward.values())}')

