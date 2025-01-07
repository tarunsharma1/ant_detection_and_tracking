
''' 
code to create xml file in CVAT 1.1 format in order to upload as annotations to CVAT
points are created as tracks and only on the first frame of a sequence I want to annotate. This is so that we can interpolate in CVAT and annotate an entire sequence.
'''
import xml.etree.cElementTree as ET


def create_xml_file_boilerplate(job_id="1264127"):

	root = ET.Element("annotations")
	ET.SubElement(root, "version").text = "1.1"
	
	meta = ET.SubElement(root, "meta")
	job = ET.SubElement(meta, "job")
	dumped = ET.SubElement(meta, "dumped").text = "2024-09-10 21:29:28.569551+00:00"

	_id = ET.SubElement(job, "id").text = job_id
	size = ET.SubElement(job, "size").text = "11"
	mode = ET.SubElement(job, "mode").text = "annotation"
	overlap = ET.SubElement(job, "overlap").text = "0"
	bugtracker = ET.SubElement(job, "bugtracker")
	created = ET.SubElement(job, "created").text = "2024-09-10 21:29:28.569551+00:00"
	updated = ET.SubElement(job, "updated").text = "2024-09-10 21:29:28.569551+00:00"
	subset = ET.SubElement(job, "subset").text = "default"
	start_frame = ET.SubElement(job, "start_frame").text = "0"
	stop_frame = ET.SubElement(job, "stop_frame").text = "10"
	frame_filter = ET.SubElement(job, "frame_filter")

	segments = ET.SubElement(job, "segments")
	segment = ET.SubElement(segments, "segment")
	segment_id = ET.SubElement(segment, "id").text = "1224831"
	start = ET.SubElement(segment, "start").text = "0"
	stop = ET.SubElement(segment, "stop").text = "10"
	url = ET.SubElement(segment, "url").text = "https://app.cvat.ai/api/jobs/" + job_id

	owner = ET.SubElement(job, "owner")
	username = ET.SubElement(owner, "username").text = "tarunsharma1"
	email = ET.SubElement(owner, "email").text = "tarunsuper@gmail.com"


	assignee = ET.SubElement(job, "assignee")

	labels = ET.SubElement(job, "labels")
	label = ET.SubElement(labels, "label")
	name = ET.SubElement(label, "name").text = "ant"
	color = ET.SubElement(label, "color").text = "#ff0000"
	type = ET.SubElement(label, "type").text = "points"
	attributes = ET.SubElement(label, "attributes").text=""

	return root


def add_tracks_to_xml_file(root,list_of_points, filename="bloby_annotation.xml"):
	### this method is for seeding annotations on CVAT by creating track annotations only for the first frame

	## list_of_points is a list of lists [[x1,y1], [x2,y2], [x3,y3]...] containing coords for blobs on the first frame

	for point in list_of_points:		
		x,y = point[0], point[1]

		#img = ET.SubElement(root, "image", id="0", name="frame-11.jpg", width="1920", height="1080")
		#points = ET.SubElement(img, "points", label="ant", source="manual", occluded="0", points="881.62,407.01", z_order="0")

		track = ET.SubElement(root, "track", id="0", label="ant", source="blob")
		points = ET.SubElement(track, "points", frame="0", keyframe="1", outside="0", occluded="0", points=str(x) + "," + str(y), z_order="0")

	tree = ET.ElementTree(root)
	tree.write(filename)


def add_img_and_points_to_xml_file(root, list_of_points, img_name, width, height, id, xml_filename):
	### this method is used when we are creating one XML for multiple patches of an image
	
	img = ET.SubElement(root, "image", id=str(id), name=img_name, width=str(width), height=str(height))
	for point in list_of_points:		
		x,y = point[0], point[1]
		
		points = ET.SubElement(img, "points", label="ant", source="manual", occluded="0", points=str(x) + "," + str(y), z_order="0")

	tree = ET.ElementTree(root)
	tree.write(xml_filename)
	


