
import os,sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir("../lib/")
sys.path.append(os.path.curdir)
#print(os.listdir())
from data2list import *
from prepare_input import *
from dataset import *
from make_model import *
import cv2
import random
import xml.etree.ElementTree as ET
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
# checked make_darapath_list
rootpath = "../data/"
train_img_list,train_anno_list,val_img_list,val_anno_list = make_datapath_list(rootpath)
#print(train_anno_list[2])

#checked Anno_xml2list
voc_classes = ['cat','dog']
transform_anno = Anno_xml2list(voc_classes)
ind = 1
image_file_path = train_img_list[ind]
img = cv2.imread(image_file_path)
height,width,channels = img.shape
#print(transform_anno(train_anno_list[ind],width,height))

#checked DataTransform
img = cv2.imread(train_img_list[0])
height,width,channels = img.shape
## show original img
anno_list = transform_anno(train_anno_list[0],width,height)
#plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
#plt.show()
## show train img
transform = DataTransform(input_size=300,color_mean=(104,117,123))
img_transformed,boxes,labels = transform(img,"train",anno_list[:,:4],anno_list[:,4])
#plt.imshow(cv2.cvtColor(img_transformed,cv2.COLOR_BGR2RGB))
#plt.show()
## show val img
img_transformed,boxes,labels=transform(img,"val",anno_list[:, :4],anno_list[:,4])
#plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
#plt.show()

#checked dataset
input_size=300
color_mean=(104,117,123)
train_dataset = DCDataset(train_img_list,train_anno_list,phase="train",
								transform=DataTransform(
												input_size,color_mean),
								transform_anno=Anno_xml2list(voc_classes))

val_dataset = DCDataset(val_img_list,val_anno_list,phase="val",
								transform=DataTransform(
												input_size,color_mean),
								transform_anno=Anno_xml2list(voc_classes))

#make dataloder
batch_size = 4
train_dataloader = data.DataLoader(
		train_dataset, batch_size=batch_size, shuffle=True, collate_fn=od_collate_fn)
val_dataloader = data.DataLoader(
		val_dataset, batch_size=batch_size, shuffle=False, collate_fn=od_collate_fn)
#dataloder nito dict
dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}
ssd_cfg = {
    'num_classes': 3,  
    'input_size': 300,  
    'bbox_aspect_num': [4, 6, 6, 6, 4, 4], 
    'feature_maps': [38, 19, 10, 5, 3, 1],  
    'steps': [8, 16, 32, 64, 100, 300],   
    'min_sizes': [30, 60, 111, 162, 213, 264], 
    'max_sizes': [60, 111, 162, 213, 264, 315],  
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
}

# make DBox
dbox = DBox(ssd_cfg)
dbox_list = dbox.make_dbox_list()

print(pd.DataFrame(dbox_list.numpy()))


