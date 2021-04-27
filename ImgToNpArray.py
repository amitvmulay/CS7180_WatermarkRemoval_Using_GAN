# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 11:10:37 2020

@author: amit
"""

import matplotlib.pyplot as plt
import os, time
import numpy as np 
from PIL import Image 
from numpy import asarray 


#dir_data      = "C:/Nothing Important/NEU/FA2020 - 6140 - Machine Learning/Dataset/img_align_celeba/"
dir_data_w      =  "/scratch/mulay.am/datasets/CLWD/test/Watermarked_image"
dir_data_g      =  "/scratch/mulay.am/datasets/CLWD/test/Watermark_free_image"
Ntrain        = 10000
Ntest         = 0
nm_imgs_w       = np.sort(os.listdir(dir_data_w))
nm_imgs_g       = np.sort(os.listdir(dir_data_g))
## name of the jpg files for training set
nm_imgs_train_w = nm_imgs_w[:Ntrain]
nm_imgs_train_g = nm_imgs_g[:Ntrain]
## name of the jpg files for the testing data
nm_imgs_test_w  = nm_imgs_w[Ntrain:Ntrain + Ntest]
nm_imgs_test_g  = nm_imgs_g[Ntrain:Ntrain + Ntest]
temp_image = plt.imread('/scratch/mulay.am/datasets/CLWD_1/train/Watermarked_image/1.jpg')
img_shape     = (temp_image.shape[0], temp_image.shape[1], 3)

def get_npdata(nm_imgs_train,dir_data):
    X_train = []
    for i, myid in enumerate(nm_imgs_train):
        img = Image.open(dir_data + "/" + myid)
        #img = img.resize((64,64))

        numpydata = asarray(img) 
        numpydata = numpydata/255.0
        
        X_train.append(numpydata)
    
    X_train = np.array(X_train)
    return(X_train)

X_train_w = get_npdata(nm_imgs_train_w,dir_data_w)
X_train_g = get_npdata(nm_imgs_train_g,dir_data_g)
print("X_train_w.shape = {}".format(X_train_w.shape))
print("X_train_g.shape = {}".format(X_train_g.shape))

#X_test_w  = get_npdata(nm_imgs_test_w,dir_data_w)
#X_test_g  = get_npdata(nm_imgs_test_g,dir_data_g)
#print("X_test_w.shape = {}".format(X_test_w.shape))
#print("X_test_g.shape = {}".format(X_test_g.shape))

np.save('/scratch/mulay.am/datasets/CLWD_1/X_test_w_10k', X_train_w)
np.save('/scratch/mulay.am/datasets/CLWD_1/X_test_g_10k', X_train_g)
#np.save('/scratch/mulay.am/datasets/CLWD_1/X_test_w_5n', X_test_w)
#np.save('/scratch/mulay.am/datasets/CLWD_1/X_test_g_5n', X_test_g)
