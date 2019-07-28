#Imports
from comet_ml import Experiment
experiment = Experiment(api_key="YkPEmantOag1R1VOJmXz11hmt", parse_args=False, project_name='SegNet')

from os.path import join, basename, dirname
import tensorflow as tf
import shutil
import os
import sys
import argparse

import cv2
import numpy as np
import os
import time

from torch.utils.data import DataLoader, ConcatDataset, random_split#, SequentialSampler #yike: add SequentialSampler
import torch
import torchvision
import torchvision.transforms as transforms
from datasets import ArtPrint
from Model import *

home = os.environ['HOME']

#General Settings
parser = argparse.ArgumentParser()

# system basics
parser.add_argument("-name", default='debug', type=str, help="name of the log")
parser.add_argument("-gpu", default='0', type=str, help="gpu numbers")

parser.add_argument("-train", default=True, help="train the NN", action="store_true")

parser.add_argument("-transfer",default=False, help="test the NN", action="store_true")

parser.add_argument("-test",default=False, help="test the NN", action="store_true")

# image and logistic parameters 
parser.add_argument("-image_h", default=32, type=int, help='image height') #('image_h', "360", """ image height """) 32
parser.add_argument("-image_w", default=128, type=int, help='image width')#('image_w', "480", """ image width """)128
#parser.add_argument("-image_h", default=360, type=int, help='image height') 
#parser.add_argument("-image_w", default=480, type=int, help='image width')

parser.add_argument("-image_c", default=1, type=int, help='image channel')#('image_c', "3", """ image channel (RGB) """)
parser.add_argument("-num_class", default=2, type=int, help='total class number')

# training hyperparam
parser.add_argument("-batch_size", default=10, type=int, help='batch_size')
parser.add_argument("-lrInit", default=1e-3, type=int, help='initial lr')
parser.add_argument("-lrDrop1", default=10, type=int, help='step to drop lr by 10 first time') # not sure
parser.add_argument("-lrDrop2", default=1000, type=int, help='step to drop lr by 10 sexond time') # not sure
parser.add_argument('-max_epoch',default=2, type=int,help='max epoch numbers')



# file paths
parser.add_argument('-ckpt_root', default="/home/classify_copy/misc_test/ckpt", type=str,help= "dir to store ckpt") # log_dir !!!!!
parser.add_argument('-data_root', default="/home/classify_copy/misc_test/datasets", type=str, help=" root to any data folder ")
parser.add_argument('-urlTranferFrom', default="", type=str, help=" archived model url ")


#args = parser.parse_args()
args = parser.parse_known_args()[0]

name = args.name

experiment.set_name(name)
experiment.log_parameters(vars(args))

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

ckptroot = args.ckpt_root
args.ckptpath = join(ckptroot, name)
if args.name=='debug': shutil.rmtree(args.ckptpath, ignore_errors=True)
os.makedirs(args.ckptpath, exist_ok=True)

def checkArgs():
    if args.test:
        print('The model is set to Testing')
        print("Check point file: %s"%args.ckptpath)
        #print("CamVid testing dir: %s"%FLAGS.test_dir)
    elif args.transfer:
        print('The model is set to transfer learn from ckpt')
        print("Check point file: %s"%args.ckptpath)
        #print("CamVid Image dir: %s"%FLAGS.image_dir)
        #print("CamVid Val dir: %s"%FLAGS.val_dir)
    else:
        print('The model is set to Training')
        print("Max training Iteration: %d"%args.max_epoch)
        print("Initial lr: %f"%args.lrInit)
        print("First Drop Steps: %i"%args.lrDrop1)
        print("Second Drop Steps: %i"%args.lrDrop2)
        print("Data root: %s"%args.data_root)
        print("Check point file: %s"%args.ckptpath)
        #print("CamVid Val dir: %s"%FLAGS.val_dir)

    print("Batch Size: %d"%args.batch_size)
    
#main function
def main():    
    checkArgs()
    if args.train:
        transform_train = transforms.Compose([
          transforms.Lambda(lambda img: cv2.resize(img, (args.image_w,args.image_h), interpolation=cv2.INTER_CUBIC)),
          transforms.Lambda(lambda img: np.expand_dims(img,3) ),
          #transforms.Lambda(lambda img: add_artifacts(img,args)),
          #transforms.Lambda(lambda img: cv2.transpose(img))
        ])
        arprint=ArtPrint(args.data_root, transform=transform_train)
        concat=arprint
        idxTrain = int( len(arprint)*0.9)
        trainset, testset = random_split(concat, [idxTrain, len(concat)-idxTrain])
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True,num_workers=4)
        testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=False,num_workers=2)
        
        #weight gen
        pos_perc=sum(map(lambda x: np.sum(cv2.imread(x[1],cv2.IMREAD_GRAYSCALE)),trainset.dataset.samples))/(args.image_h*args.image_w*len(trainset.dataset.samples))
        neg_perc=1-pos_perc
        weight=np.array([pos_perc,neg_perc]) # just reverse
        print(weight)
        model=Model(args, experiment, loss_weight=weight, mustRestore=False)
        model.training(loader=trainloader, validateloader=testloader)
        
    else:
        pass # for now, leave it pass 

if __name__ =='__main__':
    main()
    
    
