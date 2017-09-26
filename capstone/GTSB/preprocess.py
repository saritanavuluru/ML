import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import csv
import os

import keras
from keras.utils import np_utils
import random
from sklearn.utils import shuffle
from skimage import transform as transf
import cv2
import pickle
from skimage import data, img_as_float, exposure, io
from tqdm import tqdm
from scipy import misc


def conv_to_grayscale_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = np.expand_dims(img, axis=2)
    #img = np.resize(img, (img.shape[0], img.shape[1], 1))
    #print(img.shape)
    return  img

def conv_to_grayscale_data(data):
    
    """Convert to grayscale, histogram equalize, and expand dims"""
    
    #print(type(data))
    #print(data.shape)

    imgs = np.ndarray((data.shape[0], 32, 32, 1), dtype=np.uint8)
    for i, img in enumerate(data):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=2)
        #print(img.shape)

        imgs[i] = img
    #print(type(imgs))
    return imgs
    
def contrast_stretch_img(img):
  p2, p98 = np.percentile(img, (2,98))
  img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
  return img_rescale
  


def hist_equalize_img(img):

    img_eq = exposure.equalize_hist(img)

    return img_eq
    

def adaptive_equalize_img(img):
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
    return img_adapteq




#greyscale
# Convert to grayscale

def convert_to_grayscale(X_train,X_test, X_val):

    X_train_rgb = X_train
    X_val_rgb = X_val
    X_train_gry = np.sum(X_train/3, 
                     axis=3, keepdims=True)

    X_test_rgb = X_test
    X_test_gry = np.sum(X_test/3, axis=3, keepdims=True)
    X_val_gry = np.sum(X_val/3, axis=3, keepdims=True)
    
    print('X_train RGB shape:', X_train_rgb.shape)
    print('X_train Grayscale shape:', X_train_gry.shape)
    X_train = X_train_gry
    X_test = X_test_gry
    X_val = X_val_gry
    
    assert(X_train.shape[1:] == (32,32,1)), "The dimensions of the images are not 32 x 32 x 1."
    return X_train,X_test, X_val

def hist_equalize(dataset):
    hist_ds = []
    for img in  dataset:
        img_eq = exposure.equalize_hist(img)
        hist_ds.append(img_eq)
    return np.asarray(hist_ds) 


def hist_equalize_set(X_train,  X_test,  X_val):

    #X_train, X_test , X_val = convert_to_grayscale(X_train, X_test, X_val)
    
    #hist equalize
    print("Transformation : Histogram Equalization")
    X_train = hist_equalize(X_train)
    X_test = hist_equalize(X_test)
    X_val = hist_equalize(X_val)
    
     
   
    return X_train,  X_test,  X_val
    
def noralize(X_train, X_test, X_val)    :
    
    print("Transformation : Normalization")

    X_train = (X_train - X_train.mean()) / (np.max(X_train) - np.min(X_train))
    X_test = (X_test - X_test.mean()) / (np.max(X_test) - np.min(X_test))
    X_val = (X_val - X_val.mean())/(np.max(X_val)-np.min(X_val))
    return X_train, X_test, X_val

import matplotlib
matplotlib.rcParams['font.size'] = 8

def adaptive_equalize(data):
    #print(data[0].shape)
    #print(data[0].squeeze().shape)
    

    ae = []
    for i in tqdm(range(len(data))):
        #print(i)
        
        img = exposure.equalize_adapthist(data[i].squeeze(), clip_limit=0.03)
        #print(img.shape)
        #print("...")
        #print(type(img[0][0]))
        img = np.expand_dims(img, axis=2)
        #print(img.shape)
        ae.append(np.asarray(img))
    #print(len(ae))
    return np.asarray(ae)

def adaptive_equalize_set(train,test,val):
    print("Transformation : Adaptive Equalization")
    train = adaptive_equalize(train)
    test = adaptive_equalize(test)
    val = adaptive_equalize(val)
    return train,test,val

  
def contrast_stretch(data):
    #print(data.shape)
    
    cs = []
    for i in tqdm(range(len(data))):
        #print(i)
        p2, p98 = np.percentile(data[i].squeeze(), (2,98))

        img = exposure.rescale_intensity(data[i].squeeze(), in_range=(p2, p98))
        img = np.expand_dims(img, axis=2)
        #print(img.shape)
        cs.append(np.asarray(img))
    #print(len(d))
    return np.asarray(cs)



def contrast_stretch_set(train,test,val):
    print("Transformation : Contrast-Stretching")
    train = contrast_stretch(train)
    test = contrast_stretch(test)
    val = contrast_stretch(val)
    return train,test,val

  
#print(X_aug.squeeze().shape)
#a = contrast_stretch(np.asarray([img_d,img_b]))
#helper.plot_images(-1,a,2,"")



def data_augment(X,y,n_classes,n_gen_per_class,n_aug_per_class):

    print('Generating '+str(n_gen_per_class)+' number of images per class, selecting '+str(n_aug_per_class)+' number of images per class for  augmentation')
    
    datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    fill_mode='nearest',
   # horizontal_flip=True,
    )

    
    img_shape = [X.shape[1], X.shape[2], X.shape[3]]
    img_shape.insert(0,0)
    
    data_shape = np.asarray(img_shape)
    total_image_per_class = n_gen_per_class
    
    X_augmented =np.empty(data_shape)
    y_augmented =np.empty(0,dtype='uint8')
    


    print('Augmenting  Data...')
    for i in tqdm(range(n_classes)):
    #for i in tqdm(range(3)):
            #print(i)
            index = [y==i]
            images_for_i_class = X[y==i]
            y_i_class = y[y==i]
            X_augmented_i = np.empty(data_shape)
            #print(X_augmented_i.shape)
            y_augmented_i = np.empty(0,dtype='uint8')
            for X_b,y_b in datagen.flow(images_for_i_class, y_i_class, batch_size=len(y_i_class), seed=9345+i*37):            
                X_augmented_i = np.append(X_augmented_i, X_b, axis=0)
                y_augmented_i = np.append(y_augmented_i, y_b, axis=0)

                if len(X_augmented_i) >= total_image_per_class:
                    break
            X_augmented_i, y_augmented_i = shuffle(X_augmented_i, y_augmented_i, random_state=9345)
            X_augmented = np.append(X_augmented, X_augmented_i[:n_aug_per_class], axis=0)
            y_augmented = np.append(y_augmented, y_augmented_i[:n_aug_per_class], axis=0)     
    print("shufle")
    X_augmented, y_augmented = shuffle(X_augmented, y_augmented, random_state=9345)
    print("X_augmented shape: "+str(X_augmented.shape))
    print("y_augmented shape: "+str(y_augmented.shape))
    return X_augmented, y_augmented
    
    # Storing for checkpoint2
    #X_augmented = X_augmented.astype('float32')

    
