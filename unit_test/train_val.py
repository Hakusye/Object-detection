import os,sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir("../lib/")
sys.path.append(os.path.curdir)
#print(os.listdir())
from data2list import *
from prepare_input import *
from dataset import *
from make_model import *
from multiboxloss import *
#from utils.ssd_model import *

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
#############################prepare#############################
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
#Dataloder
batch_size = 20
train_dataloader = data.DataLoader(train_dataset,batch_size=batch_size,
								shuffle=True,collate_fn=od_collate_fn)
val_dataloader = data.DataLoader(val_dataset,batch_size=batch_size,
								shuffle=False,collate_fn=od_collate_fn)
#dict_obj
dataloaders_dict = {"train":train_dataloader,"val":val_dataloader}

#make SSD
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
#make_SSD
net = SSD(phase="train",cfg=ssd_cfg)
#weight
#初期重み設定法がわからない
##vgg_weights =

def weights_init(m):
	if isinstance(m,nn.Conv2d):
		init.kaiming_normal_(m.weight.data)
		if m.bias is not None:
			nn.init.constant_(m.bias,0.0)

net.extras.apply(weights_init)
net.loc.apply(weights_init)
net.conf.apply(weights_init)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:",device)
print("ok")

##########################loss optimizer ###########################

criterion = MultiBoxLoss(jaccard_thresh=0.5,neg_pos=3,device=device)
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9,weight_decay=5e-4)

###########################trainning################################
from train_model import *
num_epochs = 10
train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)















