import os,sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir("../lib/")
sys.path.append(os.path.curdir)
from data2list import *
from prepare_input import *
from dataset import *
from make_model import *
from multiboxloss import *
from predict import *

import cv2
import random
import xml.etree.ElementTree as ET
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

rootpath = "../data/"
train_img_list,train_anno_list,val_img_list,val_anno_list=make_datapath_list(rootpath)

DC_classes = ['dog','cat']
color_mean = (104,117,123)
input_size=300
train_dataset = DCDataset(train_img_list,train_anno_list,
						phase="train",transform=DataTransform(
						input_size,color_mean),
						transform_anno = Anno_xml2list(DC_classes))

val_dataset = DCDataset(val_img_list,val_anno_list,
						phase="val",transform=DataTransform(
						input_size,color_mean),
						transform_anno = Anno_xml2list(DC_classes))

ssd_cfg ={
		'num_classes': len(DC_classes)+1,
		'input_size' : input_size,
		'bbox_aspect_num' : [4,6,6,6,4,4],
		'feature_maps' : [38,19,10,5,3,1],#source size
		'steps': [8,16,31,64,100,300],
		'min_sizes': [30,60,111,162,213,264],
		'max_sizes': [60,111,162,213,264,315],
		'aspect_ratios': [[2],[2,3],[2,3],[2,3],[2],[2]],
}

net = SSD(phase="train",cfg=ssd_cfg)
net_weights = torch.load("../data/model/ssd300_10.pth")
#print(net_weights)
net.load_state_dict(net_weights)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
print("ok")

image_file_path = "../data/images/1.jpg"#image path
img = cv2.imread(image_file_path)
height,width,channels = img.shape
#plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#plt.show()

color_mean = (104,117,123)
input_size = 300
transform = DataTransform(input_size,color_mean)

phase = "val"
img_transformed, boxes, labels = transform(img, phase, "", "")
img = torch.from_numpy(img_transformed[:, :, (2, 1, 0)]).permute(2, 0, 1)

from utils.ssd_predict_show import SSDPredictShow
net.eval()
x = img.unsqueeze(0)
print(x.shape)
detections = net(x)
print(detections)
print(type(detections))
print(len(detections))
print(detections.shape)
ssd = SSDPredictShow(eval_categories=DC_classes,net=net)
ssd.show(image_file_path, data_confidence_level=0.6)




