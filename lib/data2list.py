import os
import numpy as np
import cv2
import random
import xml.etree.ElementTree as ET
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
#% matplotlib inline

def make_datapath_list(rootpath):
	imgpath_tmp = os.path.join(rootpath,'images','%s.jpg')
	annpath_tmp = os.path.join(rootpath,'annotation','%s.xml')
#get_file_name
	train_id = os.path.join(rootpath+'segmentation/train.txt')
	val_id = os.path.join(rootpath+'segmentation/train.txt')

	train_img_list = list()
	train_anno_list = list()
	val_img_list = list()
	val_anno_list = list()

	for line in open(train_id):
		file_id = line.strip()#remove space
		img_path = (imgpath_tmp % file_id)
		anno_path = (annpath_tmp % file_id)
		train_img_list.append(img_path)
		train_anno_list.append(anno_path)

	for line in open(val_id):
		file_id = line.strip()#remove space
		img_path = (imgpath_tmp % file_id)
		anno_path = (annpath_tmp % file_id)
		val_img_list.append(img_path)
		val_anno_list.append(anno_path)
	
	return train_img_list,train_anno_list,val_img_list,val_anno_list

class Anno_xml2list(object):
	def __init__(self,classes):
		self.classes = classes
	
	def __call__(self,xml_path,width,height):
		result=[]
		xml = ET.parse(xml_path).getroot()
		for obj in xml.iter('object'):
			difficult = int(obj.find('difficult').text)
			if difficult:
				continue
			bndbox = [] #annotation's list for a obj.
			name = obj.find('name').text.lower().strip()#obj's name
			bbox = obj.find('bndbox')#inform bbox
			pts = ['xmin','ymin','xmax','ymax']
			for pt in (pts):
				pixel = int(bbox.find(pt).text)-1
				if pt == 'xmin' or pt == 'xmax':
					pixel /= width
				else:
					pixel /= height
				bndbox.append(pixel)
			label_idx = self.classes.index(name)
			bndbox.append(label_idx)
			result += [bndbox]
		return np.array(result)


