
'''

takes in a list of points [[x1,y1], [x2,y2], ...] and converts it to a list of boxes [[x1_tl,y1_tl,x1_br,y1_br], [x2_tl,y2_tl,x2_br,y2_br], ...]
where tl and br are top left and bottom right of the box. Boxes are centered on the points and are of squares of side box_side unless they exceed the 
frame width or height.

'''
def convert_points_to_boxes(list_of_points, box_size, img_w=1920, img_h=1080):
	list_of_boxes = []
	for point in list_of_points:
		center_x, center_y = point
		center_x, center_y = int(center_x), int(center_y)
		top_left_x = max(center_x - int(box_size/2), 0)
		top_left_y = max(center_y - int(box_size/2), 0)
		bottom_right_x = min(center_x + int(box_size/2), img_w) 
		bottom_right_y = min(center_y + int(box_size/2), img_h)

		list_of_boxes.append([top_left_x, top_left_y, bottom_right_x, bottom_right_y])
	return list_of_boxes 
