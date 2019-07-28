import os
from os.path import join, basename, dirname
from numpy.random import choice, normal, rand, randint
from PIL import Image
import numpy as np
import cv2
from functools import reduce
import os
import sys
from matplotlib.pyplot import plot, imshow, show, colorbar
from glob import glob
home = os.environ['HOME']

htrAssetsRoot='/root/datasets/htr_assets/'
patchBoxesRoot = join(htrAssetsRoot, 'cropped_patches', 'nw_boxes-3')
patchHorizRoot = join(htrAssetsRoot, 'cropped_patches', 'nw_horizontal-2')
patchBoxesFiles = glob(join(patchBoxesRoot, '*.jpg'))
patchHorizFiles = glob(join(patchHorizRoot, '*.jpg'))

def remove_background(im, threshold):
  mask = im < threshold
  imMasked = im.copy()
  imMasked[mask] = 0
  return imMasked


def merge_patch(imBase, imPatch, centroid, threshold=100):
  '''Takes imPatch and superimpose on imBase at centroid. Returns modified image'''

  imBase, imPatch = 255 - imBase, 255 - imPatch  # invert images fro processing
  nrb, ncb = imBase.shape
  nrp, ncp = imPatch.shape
  #print(nrb)
  #print(ncb)
  # make white areas of imPatch transparent
  imPatchMasked = remove_background(imPatch, threshold)
  #print('imPatchMasked')
  #print(imPatchMasked.shape)
  # get difference of centroids between base and patch
  centroidPatch = np.array([int(dim / 2) for dim in imPatchMasked.shape])
  delta = np.array(centroid) - centroidPatch

  # add difference of centroids to the x,y position of patch
  cc, rr = np.meshgrid(np.arange(ncp), np.arange(nrp))
  #print(cc)
  #print(cc.shape)
  #print(rr.shape)
  rr = rr + int(delta[0])
  cc = cc + int(delta[1])

  # remove all parts of patch image that would expand base image
  keep = reduce(np.logical_and, [rr >= 0, rr < nrb, cc >= 0, cc < ncb])
  nrk, nck = np.max(rr[keep]) - np.min(rr[keep]) + 1, np.max(cc[keep]) - np.min(cc[keep]) + 1
  #print('nrk,nck')
  #print(nrk)
  #print(nck)
  #print(keep[0][665])
  #print(keep.shape)
  imPatchKeep = imPatchMasked[keep]
  #print(imPatchKeep.shape)
  #print(imPatchKeep)
  #print('**')
  # merge base and patch by taking maximum pixel at each position
  imMerge = imBase.copy()
  imBaseCrop = imBase.copy()
  imBaseCrop = imBaseCrop[rr[keep], cc[keep]]
  imMerge[rr[keep], cc[keep]] = np.maximum(imBaseCrop, imPatchKeep)
  imLabel=np.zeros(imMerge.shape)
  imLabel[rr[keep], cc[keep]] = np.int64(imPatchKeep>100) * np.int64(imBaseCrop<50) # yike: exclude mark area from Base 04/12/2019, threshold 50
  #print(rr[keep])
  #print(rr[keep].shape)
  #print(cc[keep])
  #print(cc[keep].shape)
  #print(imMerge.shape)
  #print('ee') 
  return 255 - imMerge, imLabel # invert back
  
def merge_patch_box_random(img, centroid_std=.05):
  imgSize = img.shape[::-1]
  imPatchFile = choice(patchBoxesFiles)
  imPatch = cv2.imread(imPatchFile, cv2.IMREAD_GRAYSCALE)
  imPatch = cv2.resize(imPatch, img.shape[::-1])
  imPatch = cv2.normalize(imPatch, None, np.min(img), np.max(img), norm_type=cv2.NORM_MINMAX)
  centroid = [imgSize[1] / 2 * (1 + normal(0, centroid_std)), imgSize[0] / 2 * (1 + normal(0, centroid_std))]
  return merge_patch(img, imPatch, centroid, threshold=50)


def merge_patch_horiz_random(img, centroid_std=.05):
  imgSize = img.shape[::-1]
  #imPatchFile = choice(patchBoxesFiles)
  imPatchFile = choice(patchHorizFiles)
  imPatch = cv2.imread(imPatchFile, cv2.IMREAD_GRAYSCALE)
  imPatch = cv2.resize(imPatch, None, fx=4, fy=1)
  imPatch = cv2.normalize(imPatch, None, np.min(img), np.max(img), norm_type=cv2.NORM_MINMAX)
  #print(imPatch.shape)
  centroid = [imgSize[1] * (.75 + normal(0, centroid_std)/2), imgSize[0] / 2 * (1 + normal(0, centroid_std))]
  #print(str(img.shape)+'!')
  #print(str(imPatch.shape)+'!')
  return merge_patch(img, imPatch, centroid, threshold=50)

#def add_artifacts(img,args): #yike: to be assessed
#  if not args.noartifact:
#    #cv2.imwrite('/root/Engagements/test/tst1_bf.jpg', img)
#    img= horizontal_stretch(img, minFactor=.5, maxFactor=1.5)
#    img = target_aspect_pad(img, targetRatio=args.imgsize[1] / args.imgsize[0])
#    img = keep_aspect_pad(img, maxFactor=1.1)

#    img = cv2.resize(img, tuple(args.imgsize), interpolation=cv2.INTER_CUBIC)

#    if rand() < .70:
#      img = merge_patch_box_random(img, centroid_std=.03)
#    else:
#      img = merge_patch_horiz_random(img, centroid_std=.05)
#    #cv2.imwrite('/root/Engagements/test/tst1_aft.jpg', img)
#  return img
  
if __name__=='__main__':
  orig_img_dir='/root/datasets/img_print_single/'
  targ_dir='/root/datasets/artifact_images_no_intersect/' # intersect won't be labeled as positive  04/12/2019
#  img_path=orig_img_dir+"'Til Death Do Us Part Car/Part.jpg"# later use glob
#  img=cv2.imread(img_path,0)  
#  img=cv2.resize(img, (128,32), interpolation=cv2.INTER_CUBIC)
  #img, imLabel = merge_patch_horiz_random(img, centroid_std=.05)
#  img, imLabel = merge_patch_box_random(img,centroid_std=.03)
  #print(img.shape) 
#  cv2.imwrite(targ_dir+'test.jpg',img)
#  cv2.imwrite(targ_dir+'tlabel.jpg',255-imLabel)
  if not os.path.exists(targ_dir):
    os.mkdir(targ_dir)
  if not os.path.exists(targ_dir+'images'):
    os.mkdir(targ_dir+'images')
  if not os.path.exists(targ_dir+'labels'):
    os.mkdir(targ_dir+'labels')
  orig_paths=glob(orig_img_dir+'**/**.jpg')
  with open(targ_dir+'databook.txt','w') as f:
    ct=0
    for path in orig_paths:
      print(ct)
      ct=ct+1
      img=cv2.imread(path,0)
      img=cv2.resize(img, (128,32), interpolation=cv2.INTER_CUBIC)
      nlst=path.replace(' ','_').split('/')
      gt=nlst[-1].split('.')[0]
      new_name='___'.join(nlst[-2:]) #use '#^#' as '/'
      #print(new_name)

      if rand() < .70:
        img_a,imLabel = merge_patch_box_random(img, centroid_std=.03)
      else:
        img_a,imLabel = merge_patch_horiz_random(img, centroid_std=.05)
    
      cv2.imwrite(targ_dir+'images/'+new_name,img_a)
      cv2.imwrite(targ_dir+'labels/'+new_name,imLabel)
      f.write(' '.join((targ_dir+'images/'+new_name,targ_dir+'labels/'+new_name,gt))+'\n')
      
      
    
    
  

