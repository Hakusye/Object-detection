import os.path
import numpy as np
import cv2
import random
import xml.etree.ElementTree as ET
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt

class DCDataset(data.Dataset):
	def __init__(self,img_list,anno_list,phase,transform,transform_anno):
		self.img_list = img_list
		self.anno_list = anno_list
		self.phase = phase
		self.transform = transform
		self.transform_anno = transform_anno

	def __len__(self):
		return len(self.img_list)

	def __getitem__(self,index):
		im,gt,h,w = self.pull_item(index)
		return im,gt

	def pull_item(self,index):
		img_path = self.img_list[index]
		img = cv2.imread(img_path)
		height,width,channels = img.shape
		#list in
		anno_path = self.anno_list[index]
		anno_list = self.transform_anno(anno_path,width,height)
		#prepare
		img,boxes,labels = self.transform(img,self.phase,anno_list[:,:4],anno_list[:,4])
		img = torch.from_numpy(img[:,:,(2,1,0)]).permute(2,0,1)
		#BBox with label -> numpy
		gt = np.hstack((boxes,np.expand_dims(labels,axis=1)))

		return img,gt,height,width



