import random
import numpy as np
import cv2
from glob import glob

import gzip
import pickle
import torch.utils.data as data
import os
from os.path import join, basename, dirname, exists
from utils import maybe_download
#from args import *
home = os.environ['HOME']

def read_text(path):
  with open(path,'r') as f:
    txt=f.read().strip('\n')
  return txt

class ArtPrint(data.Dataset):
  '''artifact printings dataset'''

  def __init__(self, root='/root/datasets', transform=None):

    self.transform=transform
    self.root = join(root,'artifact_images','artifact_images') #zip problem, sorry

    # download and put dataset in correct directory
    maybe_download('https://www.dropbox.com/s/gyod1hqau4a9lnj/artifact_images.zip?dl=0',
                   'artifact_images', root, 'folder')
    #if exists(join(self.root,'words.tgz')):
    #  if not exists(join(self.root, 'words')):
    #    os.makedirs(join(self.root, 'words'))
    #    os.system('tar xvzf '+join(self.root, 'words.tgz')+' --directory '+join(self.root, 'words'))
    #    os.system('rm '+join(self.root,'words.tgz'))

    # begin collecting all words in IAM dataset frm the words.txt summary file at the root of IAM directiory
    
    labelsFile = open(join(self.root,'databook.txt'))
    chars = set()
    self.samples = []
    #ct=0
    for line in labelsFile:
      #ct+=1
      # ignore comment line
      if not line or line[0] == '#':
        continue

      lineSplit = line.strip().split(' ')
      assert len(lineSplit) ==3

      #fileNameSplit = lineSplit[0].split('-')
      imgPath = lineSplit[0].replace('/root/datasets/artifact_images',self.root)
      # GT text are columns starting at 9
      labelPath = lineSplit[1].replace('/root/datasets/artifact_images',self.root)
      
      gt_text=lineSplit[2]

      # put sample into list
      # qyk exclude empty images
#      if '---' not in label: # qyk: data clean
#        img_test=cv2.imread(fileName, cv2.IMREAD_GRAYSCALE) #qyk: data clean
#        if not (img_test is None or np.min(img_test.shape) <= 1): #qyk: data clean
#            self.samples.append( (fileName, label) ) #qyk
      self.samples.append((imgPath,labelPath,gt_text))
      # makes list of characters
      chars = chars.union(set(list(gt_text)))  
    self.charList = sorted(list(chars))
      #if ct>=1000:
      #  break
  def __str__(self):
    return 'Artifact word image dataset. Data location: '+self.root+', Length: '+str(len(self.samples))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):

    label = cv2.imread(self.samples[idx][1],cv2.IMREAD_GRAYSCALE)
    # img = preprocess(cv2.imread(self.samples[i][0], cv2.IMREAD_GRAYSCALE),
    #                  args.imgsize, self.args, False, is_testing)
    img = cv2.imread(self.samples[idx][0], cv2.IMREAD_GRAYSCALE)
    gt_text=self.samples[idx][2]
    if self.transform:
      img = self.transform(img)
      label=self.transform(label)
    return img, label, gt_text

class ArtPrintNoIntsect(data.Dataset):
  '''artifact printings dataset - no intersect label'''

  def __init__(self, root='/root/datasets', transform=None):

    self.transform=transform
    self.root = join(root,'artifact_images_no_intersect','artifact_images_no_intersect') #zip problem, sorry

    # download and put dataset in correct directory
    maybe_download('https://www.dropbox.com/s/rogd4d5ilfm4g5e/artifact_images_no_intersect.zip?dl=0',
                   'artifact_images_no_intersect', root, 'folder')
    #if exists(join(self.root,'words.tgz')):
    #  if not exists(join(self.root, 'words')):
    #    os.makedirs(join(self.root, 'words'))
    #    os.system('tar xvzf '+join(self.root, 'words.tgz')+' --directory '+join(self.root, 'words'))
    #    os.system('rm '+join(self.root,'words.tgz'))

    # begin collecting all words in IAM dataset frm the words.txt summary file at the root of IAM directiory
    
    labelsFile = open(join(self.root,'databook.txt'))
    chars = set()
    self.samples = []
    #ct=0
    for line in labelsFile:
      #ct+=1
      # ignore comment line
      if not line or line[0] == '#':
        continue

      lineSplit = line.strip().split(' ')
      assert len(lineSplit) ==3

      #fileNameSplit = lineSplit[0].split('-')
      imgPath = lineSplit[0].replace('/root/datasets/artifact_images_no_intersect',self.root)
      # GT text are columns starting at 9
      labelPath = lineSplit[1].replace('/root/datasets/artifact_images_no_intersect',self.root)
      
      gt_text=lineSplit[2]

      # put sample into list
      # qyk exclude empty images
#      if '---' not in label: # qyk: data clean
#        img_test=cv2.imread(fileName, cv2.IMREAD_GRAYSCALE) #qyk: data clean
#        if not (img_test is None or np.min(img_test.shape) <= 1): #qyk: data clean
#            self.samples.append( (fileName, label) ) #qyk
      self.samples.append((imgPath,labelPath,gt_text))
      # makes list of characters
      #      chars = chars.union(set(list(label)))
      chars = chars.union(set(list(gt_text)))  
    self.charList = sorted(list(chars))
      #if ct>=10000:
        #break
  def __str__(self):
    return 'Artifact word image dataset - no intersect label. Data location: '+self.root+', Length: '+str(len(self.samples))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):

    label = cv2.imread(self.samples[idx][1],cv2.IMREAD_GRAYSCALE)
    # img = preprocess(cv2.imread(self.samples[i][0], cv2.IMREAD_GRAYSCALE),
    #                  args.imgsize, self.args, False, is_testing)
    img = cv2.imread(self.samples[idx][0], cv2.IMREAD_GRAYSCALE)
    gt_text=self.samples[idx][2]
    if self.transform:
      img = self.transform(img)
      label=self.transform(label)
    return img, label,gt_text

class ArtPrintNoIntsectBinary(data.Dataset):
  '''artifact printings dataset - no intersect label'''

  def __init__(self, root='/root/datasets', transform=None):

    self.transform=transform
    self.root = join(root,'artifact_images_no_intersect','artifact_images_no_intersect') #zip problem, sorry

    # download and put dataset in correct directory
    maybe_download('https://www.dropbox.com/s/rogd4d5ilfm4g5e/artifact_images_no_intersect.zip?dl=0',
                   'artifact_images_no_intersect', root, 'folder')
    #if exists(join(self.root,'words.tgz')):
    #  if not exists(join(self.root, 'words')):
    #    os.makedirs(join(self.root, 'words'))
    #    os.system('tar xvzf '+join(self.root, 'words.tgz')+' --directory '+join(self.root, 'words'))
    #    os.system('rm '+join(self.root,'words.tgz'))

    # begin collecting all words in IAM dataset frm the words.txt summary file at the root of IAM directiory
    
    labelsFile = open(join(self.root,'databook.txt'))
    #chars = set()
    self.samples = []
    #ct=0
    for line in labelsFile:
      #ct+=1
      # ignore comment line
      if not line or line[0] == '#':
        continue

      lineSplit = line.strip().split(' ')
      assert len(lineSplit) ==3

      #fileNameSplit = lineSplit[0].split('-')
      imgPath = lineSplit[0].replace('/root/datasets/artifact_images_no_intersect',self.root).replace('/images/','/images_bin/')
      # GT text are columns starting at 9
      labelPath = lineSplit[1].replace('/root/datasets/artifact_images_no_intersect',self.root)
      
      gt_text=lineSplit[2]

      # put sample into list
      # qyk exclude empty images
#      if '---' not in label: # qyk: data clean
#        img_test=cv2.imread(fileName, cv2.IMREAD_GRAYSCALE) #qyk: data clean
#        if not (img_test is None or np.min(img_test.shape) <= 1): #qyk: data clean
#            self.samples.append( (fileName, label) ) #qyk
      self.samples.append((imgPath,labelPath,gt_text))
      #chars = chars.union(set(list(gt_text)))  
      #self.charList = sorted(list(chars))
      # makes list of characters
      #chars = chars.union(set(list(label)))
      #self.charList = sorted(list(chars))
      #if ct>=10000:
        #break
  def __str__(self):
    return 'Artifact word image dataset - no intersect label. Data location: '+self.root+', Length: '+str(len(self.samples))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):

    label = cv2.imread(self.samples[idx][1],cv2.IMREAD_GRAYSCALE)
    # img = preprocess(cv2.imread(self.samples[i][0], cv2.IMREAD_GRAYSCALE),
    #                  args.imgsize, self.args, False, is_testing)
    img = cv2.imread(self.samples[idx][0], cv2.IMREAD_GRAYSCALE)
    gt_text=self.samples[idx][2]
    if self.transform:
      img = self.transform(img)
      label=self.transform(label)
    return img, label,gt_text

class ArtPrintNoIntsectBinary_20000(data.Dataset):
  '''artifact printings dataset - no intersect label'''

  def __init__(self, root='/root/datasets', transform=None):

    self.transform=transform
    self.root = join(root,'artifact_images_no_intersect','artifact_images_no_intersect') #zip problem, sorry

    # download and put dataset in correct directory
    maybe_download('https://www.dropbox.com/s/rogd4d5ilfm4g5e/artifact_images_no_intersect.zip?dl=0',
                   'artifact_images_no_intersect', root, 'folder')
    #if exists(join(self.root,'words.tgz')):
    #  if not exists(join(self.root, 'words')):
    #    os.makedirs(join(self.root, 'words'))
    #    os.system('tar xvzf '+join(self.root, 'words.tgz')+' --directory '+join(self.root, 'words'))
    #    os.system('rm '+join(self.root,'words.tgz'))

    # begin collecting all words in IAM dataset frm the words.txt summary file at the root of IAM directiory
    
    labelsFile = open(join(self.root,'databook.txt'))
    #chars = set()
    self.samples = []
    ct=0
    for line in labelsFile:
      ct+=1
      # ignore comment line
      if not line or line[0] == '#':
        continue

      lineSplit = line.strip().split(' ')
      assert len(lineSplit) ==3

      #fileNameSplit = lineSplit[0].split('-')
      imgPath = lineSplit[0].replace('/root/datasets/artifact_images_no_intersect',self.root).replace('/images/','/images_bin/')
      # GT text are columns starting at 9
      labelPath = lineSplit[1].replace('/root/datasets/artifact_images_no_intersect',self.root)
      
      gt_text=lineSplit[2]

      # put sample into list
      # qyk exclude empty images
#      if '---' not in label: # qyk: data clean
#        img_test=cv2.imread(fileName, cv2.IMREAD_GRAYSCALE) #qyk: data clean
#        if not (img_test is None or np.min(img_test.shape) <= 1): #qyk: data clean
#            self.samples.append( (fileName, label) ) #qyk
      self.samples.append((imgPath,labelPath,gt_text))
      #chars = chars.union(set(list(gt_text)))  
      #self.charList = sorted(list(chars))
      # makes list of characters
      #chars = chars.union(set(list(label)))
      #self.charList = sorted(list(chars))
      if ct>=20000:
        break
  def __str__(self):
    return 'Artifact word image dataset - no intersect label. Data location: '+self.root+', Length: '+str(len(self.samples))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):

    label = cv2.imread(self.samples[idx][1],cv2.IMREAD_GRAYSCALE)
    # img = preprocess(cv2.imread(self.samples[i][0], cv2.IMREAD_GRAYSCALE),
    #                  args.imgsize, self.args, False, is_testing)
    img = cv2.imread(self.samples[idx][0], cv2.IMREAD_GRAYSCALE)
    gt_text=self.samples[idx][2]
    if self.transform:
      img = self.transform(img)
      label=self.transform(label)
    return img, label,gt_text

class ArtPrintNoIntsectBinary_3000(data.Dataset):
  '''artifact printings dataset - no intersect label'''

  def __init__(self, root='/root/datasets', transform=None):

    self.transform=transform
    self.root = join(root,'artifact_images_no_intersect','artifact_images_no_intersect') #zip problem, sorry

    # download and put dataset in correct directory
    maybe_download('https://www.dropbox.com/s/rogd4d5ilfm4g5e/artifact_images_no_intersect.zip?dl=0',
                   'artifact_images_no_intersect', root, 'folder')
    #if exists(join(self.root,'words.tgz')):
    #  if not exists(join(self.root, 'words')):
    #    os.makedirs(join(self.root, 'words'))
    #    os.system('tar xvzf '+join(self.root, 'words.tgz')+' --directory '+join(self.root, 'words'))
    #    os.system('rm '+join(self.root,'words.tgz'))

    # begin collecting all words in IAM dataset frm the words.txt summary file at the root of IAM directiory
    
    labelsFile = open(join(self.root,'databook.txt'))
    #chars = set()
    self.samples = []
    ct=0
    for line in labelsFile:
      ct+=1
      # ignore comment line
      if not line or line[0] == '#':
        continue

      lineSplit = line.strip().split(' ')
      assert len(lineSplit) ==3

      #fileNameSplit = lineSplit[0].split('-')
      imgPath = lineSplit[0].replace('/root/datasets/artifact_images_no_intersect',self.root).replace('/images/','/images_bin/')
      # GT text are columns starting at 9
      labelPath = lineSplit[1].replace('/root/datasets/artifact_images_no_intersect',self.root)
      
      gt_text=lineSplit[2]

      # put sample into list
      # qyk exclude empty images
#      if '---' not in label: # qyk: data clean
#        img_test=cv2.imread(fileName, cv2.IMREAD_GRAYSCALE) #qyk: data clean
#        if not (img_test is None or np.min(img_test.shape) <= 1): #qyk: data clean
#            self.samples.append( (fileName, label) ) #qyk
      self.samples.append((imgPath,labelPath,gt_text))
      #chars = chars.union(set(list(gt_text)))  
      #self.charList = sorted(list(chars))
      # makes list of characters
      #chars = chars.union(set(list(label)))
      #self.charList = sorted(list(chars))
      if ct>=3000:
        break
  def __str__(self):
    return 'Artifact word image dataset - no intersect label. Data location: '+self.root+', Length: '+str(len(self.samples))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):

    label = cv2.imread(self.samples[idx][1],cv2.IMREAD_GRAYSCALE)
    # img = preprocess(cv2.imread(self.samples[i][0], cv2.IMREAD_GRAYSCALE),
    #                  args.imgsize, self.args, False, is_testing)
    img = cv2.imread(self.samples[idx][0], cv2.IMREAD_GRAYSCALE)
    gt_text=self.samples[idx][2]
    if self.transform:
      img = self.transform(img)
      label=self.transform(label)
    return img, label,gt_text


class ArtPrintNoIntsectLBW(data.Dataset):
  '''artifact printings dataset - no intersect label'''

  def __init__(self, root='/root/datasets', transform=None):

    self.transform=transform
    self.root = join(root,'artifact_images_noins_lbw') #zip problem, sorry

    # download and put dataset in correct directory
    #maybe_download('https://www.dropbox.com/s/rogd4d5ilfm4g5e/artifact_images_no_intersect.zip?dl=0',
                   #'artifact_images_no_intersect', root, 'folder')
    #if exists(join(self.root,'words.tgz')):
    #  if not exists(join(self.root, 'words')):
    #    os.makedirs(join(self.root, 'words'))
    #    os.system('tar xvzf '+join(self.root, 'words.tgz')+' --directory '+join(self.root, 'words'))
    #    os.system('rm '+join(self.root,'words.tgz'))

    # begin collecting all words in IAM dataset frm the words.txt summary file at the root of IAM directiory
    
    labelsFile = open(join(self.root,'databook.txt'))
    chars = set()
    self.samples = []
    #ct=0
    for line in labelsFile:
      #ct+=1
      # ignore comment line
      if not line or line[0] == '#':
        continue

      lineSplit = line.strip().split(' ')
      assert len(lineSplit) ==3

      #fileNameSplit = lineSplit[0].split('-')
      imgPath = lineSplit[0]#.replace('/root/datasets/artifact_images_no_intersect',self.root).replace('/images/','/images_bin/')
      # GT text are columns starting at 9
      labelPath = lineSplit[1]#.replace('/root/datasets/artifact_images_no_intersect',self.root)
      
      gt_text=lineSplit[2]

      # put sample into list
      # qyk exclude empty images
#      if '---' not in label: # qyk: data clean
#        img_test=cv2.imread(fileName, cv2.IMREAD_GRAYSCALE) #qyk: data clean
#        if not (img_test is None or np.min(img_test.shape) <= 1): #qyk: data clean
#            self.samples.append( (fileName, label) ) #qyk
      self.samples.append((imgPath,labelPath,gt_text))
      chars = chars.union(set(list(gt_text)))  
    self.charList = sorted(list(chars))
      # makes list of characters
      #      chars = chars.union(set(list(label)))
    #self.charList = sorted(list(chars))
      #if ct>=10000:
        #break

  def __str__(self):
    return 'Artifact word image dataset - no intersect label. Data location: '+self.root+', Length: '+str(len(self.samples))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):

    label = cv2.imread(self.samples[idx][1],cv2.IMREAD_GRAYSCALE)
    # img = preprocess(cv2.imread(self.samples[i][0], cv2.IMREAD_GRAYSCALE),
    #                  args.imgsize, self.args, False, is_testing)
    img = cv2.imread(self.samples[idx][0], cv2.IMREAD_GRAYSCALE)
    gt_text=self.samples[idx][2]
    if self.transform:
      img = self.transform(img)
      label=self.transform(label)
    return img, label,gt_text

class ArtPrintNoIntsectLBW_bpr_spr(data.Dataset):
  '''artifact printings dataset - no intersect label'''

  def __init__(self, root='/root/datasets', transform=None):

    self.transform=transform
    self.root = join(root,'artifact_images_noins_lbw_bprt_sprt') #zip problem, sorry

    # download and put dataset in correct directory
    #maybe_download('https://www.dropbox.com/s/rogd4d5ilfm4g5e/artifact_images_no_intersect.zip?dl=0',
                   #'artifact_images_no_intersect', root, 'folder')
    #if exists(join(self.root,'words.tgz')):
    #  if not exists(join(self.root, 'words')):
    #    os.makedirs(join(self.root, 'words'))
    #    os.system('tar xvzf '+join(self.root, 'words.tgz')+' --directory '+join(self.root, 'words'))
    #    os.system('rm '+join(self.root,'words.tgz'))

    # begin collecting all words in IAM dataset frm the words.txt summary file at the root of IAM directiory
    
    labelsFile = open(join(self.root,'databook.txt'))
    chars = set()
    self.samples = []
    #ct=0
    for line in labelsFile:
      #ct+=1
      # ignore comment line
      if not line or line[0] == '#':
        continue

      lineSplit = line.strip().split(' ')
      assert len(lineSplit) ==3

      #fileNameSplit = lineSplit[0].split('-')
      imgPath = lineSplit[0]#.replace('/root/datasets/artifact_images_no_intersect',self.root).replace('/images/','/images_bin/')
      # GT text are columns starting at 9
      labelPath = lineSplit[1]#.replace('/root/datasets/artifact_images_no_intersect',self.root)
      
      gt_text=lineSplit[2]

      # put sample into list
      # qyk exclude empty images
#      if '---' not in label: # qyk: data clean
#        img_test=cv2.imread(fileName, cv2.IMREAD_GRAYSCALE) #qyk: data clean
#        if not (img_test is None or np.min(img_test.shape) <= 1): #qyk: data clean
#            self.samples.append( (fileName, label) ) #qyk
      self.samples.append((imgPath,labelPath,gt_text))
      chars = chars.union(set(list(gt_text)))  
    self.charList = sorted(list(chars))
      # makes list of characters
      #      chars = chars.union(set(list(label)))
    #self.charList = sorted(list(chars))
      #if ct>=10000:
        #break

  def __str__(self):
    return 'Artifact word image dataset - no intersect label. Data location: '+self.root+', Length: '+str(len(self.samples))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):

    label = cv2.imread(self.samples[idx][1],cv2.IMREAD_GRAYSCALE)
    # img = preprocess(cv2.imread(self.samples[i][0], cv2.IMREAD_GRAYSCALE),
    #                  args.imgsize, self.args, False, is_testing)
    img = cv2.imread(self.samples[idx][0], cv2.IMREAD_GRAYSCALE)
    gt_text=self.samples[idx][2]
    if self.transform:
      img = self.transform(img)
      label=self.transform(label)
    return img, label,gt_text

class ArtPrintNoIntsectLBW_biameyd_sprt(data.Dataset):
  '''artifact printings dataset - no intersect label'''

  def __init__(self, root='/root/datasets', transform=None):

    self.transform=transform
    self.root = join(root,'artifact_images_noins_lbw_biameyd_sprt') #zip problem, sorry

    # download and put dataset in correct directory
    #maybe_download('https://www.dropbox.com/s/rogd4d5ilfm4g5e/artifact_images_no_intersect.zip?dl=0',
                   #'artifact_images_no_intersect', root, 'folder')
    #if exists(join(self.root,'words.tgz')):
    #  if not exists(join(self.root, 'words')):
    #    os.makedirs(join(self.root, 'words'))
    #    os.system('tar xvzf '+join(self.root, 'words.tgz')+' --directory '+join(self.root, 'words'))
    #    os.system('rm '+join(self.root,'words.tgz'))

    # begin collecting all words in IAM dataset frm the words.txt summary file at the root of IAM directiory
    
    labelsFile = open(join(self.root,'databook.txt'))
    chars = set()
    self.samples = []
    #ct=0
    for line in labelsFile:
      #ct+=1
      # ignore comment line
      if not line or line[0] == '#':
        continue

      lineSplit = line.strip().split(' ')
      assert len(lineSplit) ==3

      #fileNameSplit = lineSplit[0].split('-')
      imgPath = lineSplit[0]#.replace('/root/datasets/artifact_images_no_intersect',self.root).replace('/images/','/images_bin/')
      # GT text are columns starting at 9
      labelPath = lineSplit[1]#.replace('/root/datasets/artifact_images_no_intersect',self.root)
      
      gt_text=lineSplit[2]

      # put sample into list
      # qyk exclude empty images
#      if '---' not in label: # qyk: data clean
#        img_test=cv2.imread(fileName, cv2.IMREAD_GRAYSCALE) #qyk: data clean
#        if not (img_test is None or np.min(img_test.shape) <= 1): #qyk: data clean
#            self.samples.append( (fileName, label) ) #qyk
      self.samples.append((imgPath,labelPath,gt_text))
      chars = chars.union(set(list(gt_text)))  
    self.charList = sorted(list(chars))
      # makes list of characters
      #      chars = chars.union(set(list(label)))
    #self.charList = sorted(list(chars))
      #if ct>=10000:
        #break

  def __str__(self):
    return 'Artifact word image dataset - no intersect label. Data location: '+self.root+', Length: '+str(len(self.samples))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):

    label = cv2.imread(self.samples[idx][1],cv2.IMREAD_GRAYSCALE)
    # img = preprocess(cv2.imread(self.samples[i][0], cv2.IMREAD_GRAYSCALE),
    #                  args.imgsize, self.args, False, is_testing)
    img = cv2.imread(self.samples[idx][0], cv2.IMREAD_GRAYSCALE)
    gt_text=self.samples[idx][2]
    if self.transform:
      img = self.transform(img)
      label=self.transform(label)
    return img, label,gt_text

class ArtPrintNoIntsectLBW_biameyd_siameyd(data.Dataset):
  '''artifact printings dataset - no intersect label'''

  def __init__(self, root='/root/datasets', transform=None):

    self.transform=transform
    self.root = join(root,'artifact_images_noins_lbw_biameyd_siameyd') #zip problem, sorry

    # download and put dataset in correct directory
    #maybe_download('https://www.dropbox.com/s/rogd4d5ilfm4g5e/artifact_images_no_intersect.zip?dl=0',
                   #'artifact_images_no_intersect', root, 'folder')
    #if exists(join(self.root,'words.tgz')):
    #  if not exists(join(self.root, 'words')):
    #    os.makedirs(join(self.root, 'words'))
    #    os.system('tar xvzf '+join(self.root, 'words.tgz')+' --directory '+join(self.root, 'words'))
    #    os.system('rm '+join(self.root,'words.tgz'))

    # begin collecting all words in IAM dataset frm the words.txt summary file at the root of IAM directiory
    
    labelsFile = open(join(self.root,'databook.txt'))
    chars = set()
    self.samples = []
    #ct=0
    for line in labelsFile:
      #ct+=1
      # ignore comment line
      if not line or line[0] == '#':
        continue

      lineSplit = line.strip().split(' ')
      assert len(lineSplit) ==3

      #fileNameSplit = lineSplit[0].split('-')
      imgPath = lineSplit[0]#.replace('/root/datasets/artifact_images_no_intersect',self.root).replace('/images/','/images_bin/')
      # GT text are columns starting at 9
      labelPath = lineSplit[1]#.replace('/root/datasets/artifact_images_no_intersect',self.root)
      
      gt_text=lineSplit[2]

      # put sample into list
      # qyk exclude empty images
#      if '---' not in label: # qyk: data clean
#        img_test=cv2.imread(fileName, cv2.IMREAD_GRAYSCALE) #qyk: data clean
#        if not (img_test is None or np.min(img_test.shape) <= 1): #qyk: data clean
#            self.samples.append( (fileName, label) ) #qyk
      self.samples.append((imgPath,labelPath,gt_text))
      chars = chars.union(set(list(gt_text)))  
    self.charList = sorted(list(chars))
      # makes list of characters
      #      chars = chars.union(set(list(label)))
    #self.charList = sorted(list(chars))
      #if ct>=10000:
        #break

  def __str__(self):
    return 'Artifact word image dataset - no intersect label. Data location: '+self.root+', Length: '+str(len(self.samples))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):

    label = cv2.imread(self.samples[idx][1],cv2.IMREAD_GRAYSCALE)
    # img = preprocess(cv2.imread(self.samples[i][0], cv2.IMREAD_GRAYSCALE),
    #                  args.imgsize, self.args, False, is_testing)
    img = cv2.imread(self.samples[idx][0], cv2.IMREAD_GRAYSCALE)
    gt_text=self.samples[idx][2]
    if self.transform:
      img = self.transform(img)
      label=self.transform(label)
    return img, label,gt_text

'''

class REAL(data.Dataset):

  def __init__(self, root='/root/datasets', transform=None):

    self.transform = transform
    self.root = join(root, 'text_recognition')
    maybe_download(source_url='https://www.dropbox.com/s/n1pq94xu9kpur1a/text_recognition.zip?dl=0',filename='text_recognition', target_directory=root, filetype='zip') #'https://www.dropbox.com/s/cbhpy6clfi9a5lz/img_print_100000_clean.zip?dl=0'
    #yq patch delete unrecognized non-english samples in linux
    #os.system('find '+ root+' -maxdepth 1 -name "*.jpg" -type f -delete') find ./logs/examples -maxdepth 1 -name "*.log"
    #if exists(join(root, 'img_print_100000_en')): os.system('mv ' + join(root, 'img_print_100000_en') + ' ' + self.root)

    #folder_depth = 0
    allfiles = glob(join(self.root, 'imgs/' + '*.jpg'))
    #allfiles = [f for f in allfiles if len(basename(f))-4<=25 and len(basename(f))-4 >=1 and (not '#U' in f) and (not '---' in f)] # screen out non-recognized characters qyk
    labels = [read_text(f.replace('imgs','coord').replace('jpg','txt')) for f in allfiles]
    print('real all: '+str(len(labels)))
    all_samples = list(zip(allfiles, labels))
    self.samples= [ sample for sample in all_samples if len(sample[1])<30] 
    print('screened :'+str(len(self.samples)))
    # makes list of characters
    chars = set.union(*[set(l) for l in labels])
    self.charList = sorted(list(chars))

  def __len__(self):
    return len(self.samples)

  def __str__(self):
    return 'Printing dataset. Data location: ' + self.root + ', Length: ' + str(len(self))

  def __getitem__(self, idx):

    label = self.samples[idx][1]
    img = cv2.imread(self.samples[idx][0], cv2.IMREAD_GRAYSCALE)
    if self.transform:
      img = self.transform(img)

    return img, label

class IRS(data.Dataset):

  def __init__(self, root='/root/datasets', transform=None):

    self.transform = transform
    self.root = join(root, 'irs_handwriting')
    maybe_download(source_url='https://www.dropbox.com/s/54jarzcb0mju32d/img_cropped_irs.zip?dl=0', filename='irs_handwriting', target_directory=root, filetype='zip')
    if exists(join(root, 'img_cropped_irs')): os.system('mv '+join(root, 'img_cropped_irs')+' '+self.root)

    folder_depth = 2
    allfiles = glob(join(self.root, '**/'*folder_depth+'*.jpg'))
    labels = [basename(f)[:-4] for f in allfiles]
    #print(labels[0])
    self.samples = list(zip(allfiles, labels))
    # makes list of characters
    chars = set.union(*[set(l) for l in labels])
    self.charList = sorted(list(chars))

  def __len__(self):
    return len(self.samples)

  def __str__(self):
    return 'IRS dataset. Data location: '+self.root+', Length: '+str(len(self))

  def __getitem__(self, idx):

    label = self.samples[idx][1]
    img = cv2.imread(self.samples[idx][0], cv2.IMREAD_GRAYSCALE)
    if self.transform:
      img = self.transform(img)

    return img, label
'''
#-------------------------------HVBW-------------------------------------------------------------

class ArtPrintNoIntsectHVBW(data.Dataset):
  '''artifact printings dataset - no intersect label'''

  def __init__(self, root='/root/datasets', transform=None):

    self.transform=transform
    self.root = join(root,'artifact_images_noins_hvbw') #zip problem, sorry

    # download and put dataset in correct directory
    #maybe_download('https://www.dropbox.com/s/rogd4d5ilfm4g5e/artifact_images_no_intersect.zip?dl=0',
                   #'artifact_images_no_intersect', root, 'folder')
    #if exists(join(self.root,'words.tgz')):
    #  if not exists(join(self.root, 'words')):
    #    os.makedirs(join(self.root, 'words'))
    #    os.system('tar xvzf '+join(self.root, 'words.tgz')+' --directory '+join(self.root, 'words'))
    #    os.system('rm '+join(self.root,'words.tgz'))

    # begin collecting all words in IAM dataset frm the words.txt summary file at the root of IAM directiory
    
    labelsFile = open(join(self.root,'databook.txt'))
    chars = set()
    self.samples = []
    #ct=0
    for line in labelsFile:
      #ct+=1
      # ignore comment line
      if not line or line[0] == '#':
        continue

      lineSplit = line.strip().split(' ')
      assert len(lineSplit) ==3

      #fileNameSplit = lineSplit[0].split('-')
      imgPath = lineSplit[0]#.replace('/root/datasets/artifact_images_no_intersect',self.root).replace('/images/','/images_bin/')
      # GT text are columns starting at 9
      labelPath = lineSplit[1]#.replace('/root/datasets/artifact_images_no_intersect',self.root)
      
      gt_text=lineSplit[2]

      # put sample into list
      # qyk exclude empty images
#      if '---' not in label: # qyk: data clean
#        img_test=cv2.imread(fileName, cv2.IMREAD_GRAYSCALE) #qyk: data clean
#        if not (img_test is None or np.min(img_test.shape) <= 1): #qyk: data clean
#            self.samples.append( (fileName, label) ) #qyk
      self.samples.append((imgPath,labelPath,gt_text))
      chars = chars.union(set(list(gt_text)))  
    self.charList = sorted(list(chars))
      # makes list of characters
      #      chars = chars.union(set(list(label)))
    #self.charList = sorted(list(chars))
      #if ct>=10000:
        #break

  def __str__(self):
    return 'Artifact word image dataset - no intersect label. Data location: '+self.root+', Length: '+str(len(self.samples))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):

    label = cv2.imread(self.samples[idx][1],cv2.IMREAD_GRAYSCALE)
    # img = preprocess(cv2.imread(self.samples[i][0], cv2.IMREAD_GRAYSCALE),
    #                  args.imgsize, self.args, False, is_testing)
    img = cv2.imread(self.samples[idx][0], cv2.IMREAD_GRAYSCALE)
    gt_text=self.samples[idx][2]
    if self.transform:
      img = self.transform(img)
      label=self.transform(label)
    return img, label,gt_text

class ArtPrintNoIntsectHVBW_bpr_spr(data.Dataset):
  '''artifact printings dataset - no intersect label'''

  def __init__(self, root='/root/datasets', transform=None):

    self.transform=transform
    self.root = join(root,'artifact_images_noins_hvbw_bprt_sprt') #zip problem, sorry

    # download and put dataset in correct directory
    #maybe_download('https://www.dropbox.com/s/rogd4d5ilfm4g5e/artifact_images_no_intersect.zip?dl=0',
                   #'artifact_images_no_intersect', root, 'folder')
    #if exists(join(self.root,'words.tgz')):
    #  if not exists(join(self.root, 'words')):
    #    os.makedirs(join(self.root, 'words'))
    #    os.system('tar xvzf '+join(self.root, 'words.tgz')+' --directory '+join(self.root, 'words'))
    #    os.system('rm '+join(self.root,'words.tgz'))

    # begin collecting all words in IAM dataset frm the words.txt summary file at the root of IAM directiory
    
    labelsFile = open(join(self.root,'databook.txt'))
    chars = set()
    self.samples = []
    #ct=0
    for line in labelsFile:
      #ct+=1
      # ignore comment line
      if not line or line[0] == '#':
        continue

      lineSplit = line.strip().split(' ')
      assert len(lineSplit) ==3

      #fileNameSplit = lineSplit[0].split('-')
      imgPath = lineSplit[0]#.replace('/root/datasets/artifact_images_no_intersect',self.root).replace('/images/','/images_bin/')
      # GT text are columns starting at 9
      labelPath = lineSplit[1]#.replace('/root/datasets/artifact_images_no_intersect',self.root)
      
      gt_text=lineSplit[2]

      # put sample into list
      # qyk exclude empty images
#      if '---' not in label: # qyk: data clean
#        img_test=cv2.imread(fileName, cv2.IMREAD_GRAYSCALE) #qyk: data clean
#        if not (img_test is None or np.min(img_test.shape) <= 1): #qyk: data clean
#            self.samples.append( (fileName, label) ) #qyk
      self.samples.append((imgPath,labelPath,gt_text))
      chars = chars.union(set(list(gt_text)))  
    self.charList = sorted(list(chars))
      # makes list of characters
      #      chars = chars.union(set(list(label)))
    #self.charList = sorted(list(chars))
      #if ct>=10000:
        #break

  def __str__(self):
    return 'Artifact word image dataset - no intersect label. Data location: '+self.root+', Length: '+str(len(self.samples))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):

    label = cv2.imread(self.samples[idx][1],cv2.IMREAD_GRAYSCALE)
    # img = preprocess(cv2.imread(self.samples[i][0], cv2.IMREAD_GRAYSCALE),
    #                  args.imgsize, self.args, False, is_testing)
    img = cv2.imread(self.samples[idx][0], cv2.IMREAD_GRAYSCALE)
    gt_text=self.samples[idx][2]
    if self.transform:
      img = self.transform(img)
      label=self.transform(label)
    return img, label,gt_text

class ArtPrintNoIntsectHVBW_biameyd_sprt(data.Dataset):
  '''artifact printings dataset - no intersect label'''

  def __init__(self, root='/root/datasets', transform=None):

    self.transform=transform
    self.root = join(root,'artifact_images_noins_hvbw_biameyd_sprt') #zip problem, sorry

    # download and put dataset in correct directory
    #maybe_download('https://www.dropbox.com/s/rogd4d5ilfm4g5e/artifact_images_no_intersect.zip?dl=0',
                   #'artifact_images_no_intersect', root, 'folder')
    #if exists(join(self.root,'words.tgz')):
    #  if not exists(join(self.root, 'words')):
    #    os.makedirs(join(self.root, 'words'))
    #    os.system('tar xvzf '+join(self.root, 'words.tgz')+' --directory '+join(self.root, 'words'))
    #    os.system('rm '+join(self.root,'words.tgz'))

    # begin collecting all words in IAM dataset frm the words.txt summary file at the root of IAM directiory
    
    labelsFile = open(join(self.root,'databook.txt'))
    chars = set()
    self.samples = []
    #ct=0
    for line in labelsFile:
      #ct+=1
      # ignore comment line
      if not line or line[0] == '#':
        continue

      lineSplit = line.strip().split(' ')
      assert len(lineSplit) ==3

      #fileNameSplit = lineSplit[0].split('-')
      imgPath = lineSplit[0]#.replace('/root/datasets/artifact_images_no_intersect',self.root).replace('/images/','/images_bin/')
      # GT text are columns starting at 9
      labelPath = lineSplit[1]#.replace('/root/datasets/artifact_images_no_intersect',self.root)
      
      gt_text=lineSplit[2]

      # put sample into list
      # qyk exclude empty images
#      if '---' not in label: # qyk: data clean
#        img_test=cv2.imread(fileName, cv2.IMREAD_GRAYSCALE) #qyk: data clean
#        if not (img_test is None or np.min(img_test.shape) <= 1): #qyk: data clean
#            self.samples.append( (fileName, label) ) #qyk
      self.samples.append((imgPath,labelPath,gt_text))
      chars = chars.union(set(list(gt_text)))  
    self.charList = sorted(list(chars))
      # makes list of characters
      #      chars = chars.union(set(list(label)))
    #self.charList = sorted(list(chars))
      #if ct>=10000:
        #break

  def __str__(self):
    return 'Artifact word image dataset - no intersect label. Data location: '+self.root+', Length: '+str(len(self.samples))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):

    label = cv2.imread(self.samples[idx][1],cv2.IMREAD_GRAYSCALE)
    # img = preprocess(cv2.imread(self.samples[i][0], cv2.IMREAD_GRAYSCALE),
    #                  args.imgsize, self.args, False, is_testing)
    img = cv2.imread(self.samples[idx][0], cv2.IMREAD_GRAYSCALE)
    gt_text=self.samples[idx][2]
    if self.transform:
      img = self.transform(img)
      label=self.transform(label)
    return img, label, gt_text

class ArtPrintNoIntsectHVBW_biameyd_siameyd(data.Dataset):
  '''artifact printings dataset - no intersect label'''

  def __init__(self, root='/root/datasets', transform=None):

    self.transform=transform
    self.root = join(root,'artifact_images_noins_hvbw_biameyd_siameyd') #zip problem, sorry

    # download and put dataset in correct directory
    #maybe_download('https://www.dropbox.com/s/rogd4d5ilfm4g5e/artifact_images_no_intersect.zip?dl=0',
                   #'artifact_images_no_intersect', root, 'folder')
    #if exists(join(self.root,'words.tgz')):
    #  if not exists(join(self.root, 'words')):
    #    os.makedirs(join(self.root, 'words'))
    #    os.system('tar xvzf '+join(self.root, 'words.tgz')+' --directory '+join(self.root, 'words'))
    #    os.system('rm '+join(self.root,'words.tgz'))

    # begin collecting all words in IAM dataset frm the words.txt summary file at the root of IAM directiory
    
    labelsFile = open(join(self.root,'databook.txt'))
    chars = set()
    self.samples = []
    #ct=0
    for line in labelsFile:
      #ct+=1
      # ignore comment line
      if not line or line[0] == '#':
        continue

      lineSplit = line.strip().split(' ')
      assert len(lineSplit) ==3

      #fileNameSplit = lineSplit[0].split('-')
      imgPath = lineSplit[0]#.replace('/root/datasets/artifact_images_no_intersect',self.root).replace('/images/','/images_bin/')
      # GT text are columns starting at 9
      labelPath = lineSplit[1]#.replace('/root/datasets/artifact_images_no_intersect',self.root)
      
      gt_text=lineSplit[2]

      # put sample into list
      # qyk exclude empty images
#      if '---' not in label: # qyk: data clean
#        img_test=cv2.imread(fileName, cv2.IMREAD_GRAYSCALE) #qyk: data clean
#        if not (img_test is None or np.min(img_test.shape) <= 1): #qyk: data clean
#            self.samples.append( (fileName, label) ) #qyk
      self.samples.append((imgPath,labelPath,gt_text))
      chars = chars.union(set(list(gt_text)))  
    self.charList = sorted(list(chars))
      # makes list of characters
      #      chars = chars.union(set(list(label)))
    #self.charList = sorted(list(chars))
      #if ct>=10000:
        #break

  def __str__(self):
    return 'Artifact word image dataset - no intersect label. Data location: '+self.root+', Length: '+str(len(self.samples))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):

    label = cv2.imread(self.samples[idx][1],cv2.IMREAD_GRAYSCALE)
    # img = preprocess(cv2.imread(self.samples[i][0], cv2.IMREAD_GRAYSCALE),
    #                  args.imgsize, self.args, False, is_testing)
    img = cv2.imread(self.samples[idx][0], cv2.IMREAD_GRAYSCALE)
    gt_text=self.samples[idx][2]
    if self.transform:
      img = self.transform(img)
      label=self.transform(label)
    return img, label, gt_text

class ArtPrintNoIntsectHVBW_biameyd_siameyd_5000(data.Dataset):
  '''artifact printings dataset - no intersect label'''

  def __init__(self, root='/root/datasets', transform=None):

    self.transform=transform
    self.root = join(root,'artifact_images_noins_hvbw_biameyd_siameyd') #zip problem, sorry

    # download and put dataset in correct directory
    #maybe_download('https://www.dropbox.com/s/rogd4d5ilfm4g5e/artifact_images_no_intersect.zip?dl=0',
                   #'artifact_images_no_intersect', root, 'folder')
    #if exists(join(self.root,'words.tgz')):
    #  if not exists(join(self.root, 'words')):
    #    os.makedirs(join(self.root, 'words'))
    #    os.system('tar xvzf '+join(self.root, 'words.tgz')+' --directory '+join(self.root, 'words'))
    #    os.system('rm '+join(self.root,'words.tgz'))

    # begin collecting all words in IAM dataset frm the words.txt summary file at the root of IAM directiory
    
    
    gt_dict=dict(map(lambda x: (x[0],x[-1]),(map(lambda x: x.strip('\n').split(' '),open(join(self.root,'words.txt')).readlines()))))
    
    labelsFile = open(join(self.root,'databook.txt'))
    chars = set()
    self.samples = []
    ct=0
    for line in labelsFile:
      ct+=1
      # ignore comment line
      if not line or line[0] == '#':
        continue

      lineSplit = line.strip().split(' ')
      assert len(lineSplit) ==3

      #fileNameSplit = lineSplit[0].split('-')
      imgPath = lineSplit[0]#.replace('/root/datasets/artifact_images_no_intersect',self.root).replace('/images/','/images_bin/')
      # GT text are columns starting at 9
      labelPath = lineSplit[1]#.replace('/root/datasets/artifact_images_no_intersect',self.root)
      
      gt_text=lineSplit[2]
      
      if '-' in gt_text:
        gt_text=gt_dict[gt_text] 

      # put sample into list
      # qyk exclude empty images
#      if '---' not in label: # qyk: data clean
#        img_test=cv2.imread(fileName, cv2.IMREAD_GRAYSCALE) #qyk: data clean
#        if not (img_test is None or np.min(img_test.shape) <= 1): #qyk: data clean
#            self.samples.append( (fileName, label) ) #qyk
      self.samples.append((imgPath,labelPath,gt_text))
      chars = chars.union(set(list(gt_text)))  

      # makes list of characters
      #      chars = chars.union(set(list(label)))
    #self.charList = sorted(list(chars))
      if ct>=5000:
        break
    self.charList = sorted(list(chars))
  def __str__(self):
    return 'Artifact word image dataset - no intersect label. Data location: '+self.root+', Length: '+str(len(self.samples))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):

    label = cv2.imread(self.samples[idx][1],cv2.IMREAD_GRAYSCALE)
    # img = preprocess(cv2.imread(self.samples[i][0], cv2.IMREAD_GRAYSCALE),
    #                  args.imgsize, self.args, False, is_testing)
    img = cv2.imread(self.samples[idx][0], cv2.IMREAD_GRAYSCALE)
    gt_text=self.samples[idx][2]
    if self.transform:
      img = self.transform(img)
      label=self.transform(label)
    return img, label, gt_text


###########IRS Value Data###############################################
class ArtTypeWriter(data.Dataset):
  '''artifact printings dataset'''

  def __init__(self, root='/root/datasets', transform=None):

    self.transform=transform
    self.root = join(root,'art_gen_printing') #zip problem, sorry

    # download and put dataset in correct directory
    #maybe_download('https://www.dropbox.com/s/gyod1hqau4a9lnj/artifact_images.zip?dl=0',
    #               'artifact_images', root, 'folder')
    #if exists(join(self.root,'words.tgz')):
    #  if not exists(join(self.root, 'words')):
    #    os.makedirs(join(self.root, 'words'))
    #    os.system('tar xvzf '+join(self.root, 'words.tgz')+' --directory '+join(self.root, 'words'))
    #    os.system('rm '+join(self.root,'words.tgz'))

    # begin collecting all words in IAM dataset frm the words.txt summary file at the root of IAM directiory
    
    labelsFile = open(join(self.root,'databook.txt'))
    chars = set()
    self.samples = []
    #ct=0
    for line in labelsFile:
      #ct+=1
      # ignore comment line
      if not line or line[0] == '#':
        continue

      lineSplit = line.strip().split(' ')
      assert len(lineSplit) ==3

      #fileNameSplit = lineSplit[0].split('-')
      imgPath = lineSplit[0]#.replace('/root/datasets/artifact_images',self.root)
      # GT text are columns starting at 9
      labelPath = lineSplit[1]#.replace('/root/datasets/artifact_images',self.root)
      
      gt_text=lineSplit[2]

      # put sample into list
      # qyk exclude empty images
#      if '---' not in label: # qyk: data clean
#        img_test=cv2.imread(fileName, cv2.IMREAD_GRAYSCALE) #qyk: data clean
#        if not (img_test is None or np.min(img_test.shape) <= 1): #qyk: data clean
#            self.samples.append( (fileName, label) ) #qyk
      self.samples.append((imgPath,labelPath,gt_text))
      chars = chars.union(set(list(gt_text)))  
    self.charList = sorted(list(chars))
      # makes list of characters
      #      chars = chars.union(set(list(label)))
    #self.charList = sorted(list(chars))
      #if ct>=1000:
      #  break
  def __str__(self):
    return 'Artifact word image dataset. Data location: '+self.root+', Length: '+str(len(self.samples))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):

    label = cv2.imread(self.samples[idx][1],cv2.IMREAD_GRAYSCALE)
    # img = preprocess(cv2.imread(self.samples[i][0], cv2.IMREAD_GRAYSCALE),
    #                  args.imgsize, self.args, False, is_testing)
    img = cv2.imread(self.samples[idx][0], cv2.IMREAD_GRAYSCALE)
    gt_text=self.samples[idx][2]
    if self.transform:
      img = self.transform(img)
      label=self.transform(label)
    return img, label, gt_text

class ArtTypeWriterBin(data.Dataset):
  '''artifact printings dataset'''

  def __init__(self, root='/root/datasets', transform=None):

    self.transform=transform
    self.root = join(root,'art_gen_printing_bin') #zip problem, sorry

    # download and put dataset in correct directory
    #maybe_download('https://www.dropbox.com/s/gyod1hqau4a9lnj/artifact_images.zip?dl=0',
    #               'artifact_images', root, 'folder')
    #if exists(join(self.root,'words.tgz')):
    #  if not exists(join(self.root, 'words')):
    #    os.makedirs(join(self.root, 'words'))
    #    os.system('tar xvzf '+join(self.root, 'words.tgz')+' --directory '+join(self.root, 'words'))
    #    os.system('rm '+join(self.root,'words.tgz'))

    # begin collecting all words in IAM dataset frm the words.txt summary file at the root of IAM directiory
    
    labelsFile = open(join(self.root,'databook.txt'))
    chars = set()
    self.samples = []
    #ct=0
    for line in labelsFile:
      #ct+=1
      # ignore comment line
      if not line or line[0] == '#':
        continue

      lineSplit = line.strip().split(' ')
      assert len(lineSplit) ==3

      #fileNameSplit = lineSplit[0].split('-')
      imgPath = lineSplit[0]#.replace('/root/datasets/artifact_images',self.root)
      # GT text are columns starting at 9
      labelPath = lineSplit[1]#.replace('/root/datasets/artifact_images',self.root)
      
      gt_text=lineSplit[2]

      # put sample into list
      # qyk exclude empty images
#      if '---' not in label: # qyk: data clean
#        img_test=cv2.imread(fileName, cv2.IMREAD_GRAYSCALE) #qyk: data clean
#        if not (img_test is None or np.min(img_test.shape) <= 1): #qyk: data clean
#            self.samples.append( (fileName, label) ) #qyk
      self.samples.append((imgPath,labelPath,gt_text))
      chars = chars.union(set(list(gt_text)))  
    self.charList = sorted(list(chars))
      # makes list of characters
      #      chars = chars.union(set(list(label)))
    #self.charList = sorted(list(chars))
      #if ct>=1000:
      #  break
  def __str__(self):
    return 'Artifact word image dataset. Data location: '+self.root+', Length: '+str(len(self.samples))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):

    label = cv2.imread(self.samples[idx][1],cv2.IMREAD_GRAYSCALE)
    # img = preprocess(cv2.imread(self.samples[i][0], cv2.IMREAD_GRAYSCALE),
    #                  args.imgsize, self.args, False, is_testing)
    img = cv2.imread(self.samples[idx][0], cv2.IMREAD_GRAYSCALE)
    gt_text=self.samples[idx][2]
    if self.transform:
      img = self.transform(img)
      label=self.transform(label)
    return img, label, gt_text



#-------------For Recognition-------------------------------------------

class RecgArtPrintNoIntsectHVBW(data.Dataset):
  '''artifact printings dataset - no intersect label'''

  def __init__(self, root='/root/datasets', transform=None):

    self.transform=transform
    self.root = join(root,'artifact_images_noins_hvbw') #zip problem, sorry

    # download and put dataset in correct directory
    #maybe_download('https://www.dropbox.com/s/rogd4d5ilfm4g5e/artifact_images_no_intersect.zip?dl=0',
                   #'artifact_images_no_intersect', root, 'folder')
    #if exists(join(self.root,'words.tgz')):
    #  if not exists(join(self.root, 'words')):
    #    os.makedirs(join(self.root, 'words'))
    #    os.system('tar xvzf '+join(self.root, 'words.tgz')+' --directory '+join(self.root, 'words'))
    #    os.system('rm '+join(self.root,'words.tgz'))

    # begin collecting all words in IAM dataset frm the words.txt summary file at the root of IAM directiory
    
    labelsFile = open(join(self.root,'databook.txt'))
    chars = set()
    self.samples = []
    #ct=0
    for line in labelsFile:
      #ct+=1
      # ignore comment line
      if not line or line[0] == '#':
        continue

      lineSplit = line.strip().split(' ')
      assert len(lineSplit) ==3

      #fileNameSplit = lineSplit[0].split('-')
      imgPath = lineSplit[0]#.replace('/root/datasets/artifact_images_no_intersect',self.root).replace('/images/','/images_bin/')
      # GT text are columns starting at 9
      labelPath = lineSplit[1]#.replace('/root/datasets/artifact_images_no_intersect',self.root)
      
      gt_text=lineSplit[2]

      # put sample into list
      # qyk exclude empty images
#      if '---' not in label: # qyk: data clean
#        img_test=cv2.imread(fileName, cv2.IMREAD_GRAYSCALE) #qyk: data clean
#        if not (img_test is None or np.min(img_test.shape) <= 1): #qyk: data clean
#            self.samples.append( (fileName, label) ) #qyk
      self.samples.append((imgPath,labelPath,gt_text))
      chars = chars.union(set(list(gt_text)))  
    self.charList = sorted(list(chars))
      # makes list of characters
      #      chars = chars.union(set(list(label)))
    #self.charList = sorted(list(chars))
      #if ct>=10000:
        #break

  def __str__(self):
    return 'Artifact word image dataset - no intersect label. Data location: '+self.root+', Length: '+str(len(self.samples))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):

    ######label = cv2.imread(self.samples[idx][1],cv2.IMREAD_GRAYSCALE)
    # img = preprocess(cv2.imread(self.samples[i][0], cv2.IMREAD_GRAYSCALE),
    #                  args.imgsize, self.args, False, is_testing)
    img = cv2.imread(self.samples[idx][0], cv2.IMREAD_GRAYSCALE)
    gt=self.samples[idx][2]
    if self.transform:
      img = self.transform(img)
      #######label=self.transform(label)
    return img, gt #label

class IRSPRT(data.Dataset):

  def __init__(self, root='/root/datasets', transform=None):

    self.transform = transform
    self.root = join(root, 'sd02/1040_crops_fix')
    #maybe_download(source_url='https://www.dropbox.com/s/54jarzcb0mju32d/img_cropped_irs.zip?dl=0', filename='irs_handwriting', target_directory=root, filetype='zip')
    #if exists(join(root, 'img_cropped_irs')): os.system('mv '+join(root, 'img_cropped_irs')+' '+self.root)
    #todo: upload data and adjust
    folder_depth = 2
    allfiles = glob(join(self.root, '**/'*folder_depth+'*.png'))
    labels = [basename(f)[basename(f).find('_')+1:-4] for f in allfiles]
    #print(labels[0])
    self.samples = list(zip(allfiles, labels))
    # makes list of characters
    chars = set.union(*[set(l) for l in labels])
    self.charList = sorted(list(chars))

  def __len__(self):
    return len(self.samples)

  def __str__(self):
    return 'IRS dataset. Data location: '+self.root+', Length: '+str(len(self))

  def __getitem__(self, idx):

    label = self.samples[idx][1]
    img = cv2.imread(self.samples[idx][0], cv2.IMREAD_GRAYSCALE)
    if self.transform:
      img = self.transform(img)

    return img, label

class IRSPRT_3000(data.Dataset):

  def __init__(self, root='/root/datasets', transform=None):

    self.transform = transform
    self.root = join(root, 'sd02/1040_crops_fix')
    #maybe_download(source_url='https://www.dropbox.com/s/54jarzcb0mju32d/img_cropped_irs.zip?dl=0', filename='irs_handwriting', target_directory=root, filetype='zip')
    #if exists(join(root, 'img_cropped_irs')): os.system('mv '+join(root, 'img_cropped_irs')+' '+self.root)
    #todo: upload data and adjust
    folder_depth = 2
    allfiles = glob(join(self.root, '**/'*folder_depth+'*.png'))[:3001]
    labels = [basename(f)[basename(f).find('_')+1:-4] for f in allfiles]
    #print(labels[0])
    self.samples = list(zip(allfiles, labels))
    # makes list of characters
    chars = set.union(*[set(l) for l in labels])
    self.charList = sorted(list(chars))

  def __len__(self):
    return len(self.samples)

  def __str__(self):
    return 'IRS dataset. Data location: '+self.root+', Length: '+str(len(self))

  def __getitem__(self, idx):

    label = self.samples[idx][1]
    img = cv2.imread(self.samples[idx][0], cv2.IMREAD_GRAYSCALE)
    if self.transform:
      img = self.transform(img)

    return img, label


class IRSPRTWORD(data.Dataset):

  def __init__(self, root='/root/datasets', transform=None):

    self.transform = transform
    self.root = join(root, 'sd02/1040_word_crops')
    #maybe_download(source_url='https://www.dropbox.com/s/54jarzcb0mju32d/img_cropped_irs.zip?dl=0', filename='irs_handwriting', target_directory=root, filetype='zip')
    #if exists(join(root, 'img_cropped_irs')): os.system('mv '+join(root, 'img_cropped_irs')+' '+self.root)
    #todo: upload data and adjust
    folder_depth = 2
    allfiles = glob(join(self.root, '**/'*folder_depth+'*.png'))
    labels = [basename(f)[basename(f).find('_')+1:-4] for f in allfiles]
    #print(labels[0])
    self.samples = list(zip(allfiles, labels))
    # makes list of characters
    chars = set.union(*[set(l) for l in labels])
    self.charList = sorted(list(chars))

  def __len__(self):
    return len(self.samples)

  def __str__(self):
    return 'IRS dataset. Data location: '+self.root+', Length: '+str(len(self))

  def __getitem__(self, idx):

    label = self.samples[idx][1]
    img = cv2.imread(self.samples[idx][0], cv2.IMREAD_GRAYSCALE)
    if self.transform:
      img = self.transform(img)

    return img, label

class IRSManual(data.Dataset):

  def __init__(self, root='/root/datasets', transform=None):

    self.transform = transform
    self.root = join(root, 'sd02/testing_dataset')
    #maybe_download(source_url='https://www.dropbox.com/s/54jarzcb0mju32d/img_cropped_irs.zip?dl=0', filename='irs_handwriting', target_directory=root, filetype='zip')
    #if exists(join(root, 'img_cropped_irs')): os.system('mv '+join(root, 'img_cropped_irs')+' '+self.root)
    #todo: upload data and adjust
    folder_depth = 0
    allfiles = glob(join(self.root, '**/'*folder_depth+'*.JPG'))
    labels = [basename(f)[:-4] for f in allfiles] #[basename(f).find('_')+1:-4]
    #print(labels[0])
    self.samples = list(zip(allfiles, labels))
    # makes list of characters
    chars = set.union(*[set(l) for l in labels])
    self.charList = sorted(list(chars))

  def __len__(self):
    return len(self.samples)

  def __str__(self):
    return 'IRS dataset. Data location: '+self.root+', Length: '+str(len(self))

  def __getitem__(self, idx):

    label = self.samples[idx][1]
    img = cv2.imread(self.samples[idx][0], cv2.IMREAD_GRAYSCALE)
    if self.transform:
      img = self.transform(img)

    return img, label



if __name__=='__main__':
  artp=ArtPrint()
  leng=artp.__len__()
  print(leng)
  for idx in range(leng):
    img,label=artp.__getitem__(idx)
    if img.shape!=(32,128) or label.shape!=(32,128):
      print('-----')
      print(img.shape)
      print(label.shape)