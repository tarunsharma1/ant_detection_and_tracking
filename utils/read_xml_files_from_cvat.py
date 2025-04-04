import xml.etree.cElementTree as ET
from collections import defaultdict

'''

Read annotation files downloaded from CVAT in two different formats: cvat for images 1.1, and cvat for videos

The cvat for images format is used for per frame detections:
	We read the xml files and convert it to a dictionary of points. Keys are entire path to video subfolder (video that was annotated) + _ + frame name,
	where frame name is subfolder_frame_number. Example of a key is '/media/tarun/Backup5TB/all_ant_data/shack-tree-diffuser-08-01-2024_to_08-22-2024/2024-08-22_03_01_01/2024-08-22_03_01_01_350.jpg'
	Value is a list of points [[x1,y1], [x2,y2], ...] which are the points of annotations (ants). 


The cvat for videos format will be used for tracking ants across a sequence.


'''



def xml_to_dict_cvat_for_images(xml_file, folder_where_annotated_video_came_from):
	tree = ET.parse(xml_file)
	root = tree.getroot()

	to_dict = {}

	## root[0] is <version> and root[1] is <meta>. After that each element is an image
	num_frames = len(root) - 2
	for frame_idx in range(2, num_frames+2):
		frame_name = root[frame_idx].get('name')
		
		to_dict[folder_where_annotated_video_came_from + frame_name] = []

		number_of_points = len(root[frame_idx])

		for point_idx in range(0, number_of_points):
			if root[frame_idx][point_idx].get('occluded') == '1':
				continue

			points = root[frame_idx][point_idx].get('points').split(',')
			point_x, point_y = [int(float(x)) for x in points]

			to_dict[folder_where_annotated_video_came_from + frame_name].append([point_x, point_y])

	return to_dict




def xml_to_dict_cvat_for_videos(xml_file):
	'''
	returns dictionary where keys are frame numbers and values are a list of lists containing [ant_id, x,y]
	while annotating on CVAT, if we annotate frames 250-280, the value of "frame" in the field will be 0-30. 
	

	'''
	tree = ET.parse(xml_file)
	root = tree.getroot()

	to_dict = defaultdict(list)

	## root[0] is <version> and root[1] is <meta>. After that each element is an ant track
	num_ant_ids = len(root) - 2
	for ant_id_idx in range(2, num_ant_ids+2):
		ant_id = int(root[ant_id_idx].get('id'))

		for point in root[ant_id_idx]:
			frame_number = int(point.get('frame'))

			if point.get('outside') == '1':
				continue
			point_x, point_y = [int(float(x)) for x in point.get('points').split(',')]

			to_dict[frame_number].append([ant_id, point_x, point_y])
	return to_dict










