import os
import traceback

import cv2
import xml.etree.ElementTree as ET
import numpy
import configparser
from ast import literal_eval
from operations import blur, rotate, perspective, flip, brightness, write_annotation, write_image, update_annotation, supress_bboxes
import copy

def read_annotations(file_path):
	tree = ET.parse(file_path)
	root = tree.getroot()
	bboxes = []
	i = 1
	while(1):
		if(i<len(root)):
			if(root[i].tag == "object"):
				if(root[i][0].text == 'none'):
					root[i][0].text == 'good'
				bboxes.append([float(root[i][2][0].text), float(root[i][2][1].text), float(root[i][2][2].text), float(root[i][2][3].text), 0])
				i+=1
		else:
			break
	return bboxes, tree


def call_augmentation_functions(aug_types, data_path, config):
	for file in os.listdir(data_path):
		try:
			if not file.endswith("jpeg"):
				continue
			#print("data path: ", data_path)
			#print("file: ", file)

			img = cv2.imread(os.path.join(data_path, file))
			###blur###
			if(aug_types['blur'] == 'Y'):
				ksize = config['BLUR_KSIZE']['ksize']
				img_ = blur(img, ksize)
				write_image(config['DATASET_PATH']['images_destination'], "blur_" + file, img_)

			###warp###
			if(aug_types['warp'] == 'Y'):
				if(config['WARP']['warp_x'] == 'Y'):
					e_x = literal_eval(config['WARP']['e_x'])
					for e in e_x:
						img_ = perspective(img, 1, e)
						write_image(config['DATASET_PATH']['images_destination'], "warp_x" + "-" + str(e) + "_" + file, img_)
						
				if(config['WARP']['warp_y'] == 'Y'):
					e_y =literal_eval(config['WARP']['e_y'])
					for e in e_y:
						img_ = perspective(img, 0, e)
						write_image(config['DATASET_PATH']['images_destination'], "warp_y" + "-" + str(e) + "_" + file, img_)

				if(config['WARP']['warp_x_y'] == 'Y'):
					e_x_y =literal_eval(config['WARP']['e_x_y'])
					for e in e_x_y:
						img_ = perspective(img, 2, e)
						write_image(config['DATASET_PATH']['images_destination'], "warp_x_y" + "-" + str(e) + "_" + file, img_)
					

			###rotate###
			if(aug_types['rotate'] == 'Y'):
				angles = numpy.arange(int(config['ROTATION_ANGLES']['min']), int(config['ROTATION_ANGLES']['max']), int(config['ROTATION_ANGLES']['interval']))
				index = numpy.argwhere(angles == 0)
				angles = numpy.delete(angles, index)
				for angle in angles:
					img_ = rotate(img, angle)

					if(angle<0):
						write_image(config['DATASET_PATH']['images_destination'], "m" + str(-angle) + "_" + file, img_)
					else:
						write_image(config['DATASET_PATH']['images_destination'], str(angle) + "_" + file, img_)
				
			###flip###
			if(aug_types['flip'] == 'Y'):
				img_ = flip(img)
				write_image(config['DATASET_PATH']['images_destination'], "flip_" + file, img_)
			###bright###
			if(aug_types['bright'] == 'Y'):
				img_ = brightness(img)
				write_image(config['DATASET_PATH']['images_destination'], "bright_" + file, img_)
		except:
			traceback.print_exc()