import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import csv
import os
import params
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
import loader

# Load from Disk

    

def preprocess_aug_combine_save(X_train,y_train,X_test,y_test,X_val,y_val):

    #Apply Transformations
    
    X_train, X_test,X_val= convert_to_grayscale(X_train, X_test,X_val)
    
    X_train_h, X_test_h, X_val_h = hist_equalize_set(X_train, X_test, X_val)
    X_train_hae ,X_test_hae, X_val_hae = adaptive_equalize_set(X_train_h, X_test_h, X_val_h)
    X_train_hcs,X_test_hcs,X_val_hcs = contrast_stretch_set(X_train_h,X_test_h,X_val_h)
    ##X_train_n, X_test_n,X_val_n = preprocess.noralize(X_train, X_test,X_val)

    #combine
    X_hset,y_hset = combine_datasets([(X_train_h,y_train),(X_train_hae,y_train),(X_train_hcs,y_train)])
    
    #augment
    X_aug_hset, y_aug_hset = data_augment(X_hset,y_hset,n_classes=params.n_classes,n_gen_per_class=params.n_generate,n_aug_per_class=params.n_select)

    X_aug_classes, y_aug_classes = data_augment_classes(X_hset,y_hset,[7,11,15,18,22,30,35,42],n_gen_per_class=5000,n_aug_per_class=400)

   
    #save preprocessed data
    loader.save_to_disk_6(X_train,y_train,X_test,y_test,X_val,y_val,False,"gray")
    print(y_val)
    loader.save_to_disk_6(X_train_h,y_train,X_test_h,y_test,X_val_h,y_val,False,"hist")
    print(y_val)
    loader.save_to_disk_6(X_train_hae,y_train,X_test_hae,y_test,X_val_hae,y_val,False,"hae")
    loader.save_to_disk_6(X_train_hcs,y_train,X_test_hcs,y_test,X_val_hcs,y_val,False,"hcs")
    loader.save_aug_all_to_disk(X_aug_hset,y_aug_hset,"hset_aug")
    
    
    #combine
    #X_aug_all,y_aug_all = combine_datasets([(X_all,y_all),(X_aug_all,y_aug_all),(X_aug_classes,y_aug_classes)])
    X_train,y_train = combine_datasets([(X_aug_hset,y_aug_hset),(X_hset,y_hset),(X_aug_classes,y_aug_classes)])
    
    
    #Use HAE for the test set and  the  validation set.
    
    X_train,y_train =shuffle( X_train,y_train,random_state=42)
    X_val,y_val = shuffle(X_val_hae,y_val,random_state=42)
    X_test,y_test = shuffle(X_test_hae,y_test,random_state=42)
     
    
    #return
    
    return X_train,y_train,X_test,y_test,X_val,y_val

    


def conv_to_grayscale_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = np.expand_dims(img, axis=2)
    #img = np.resize(img, (img.shape[0], img.shape[1], 1))
    #print(img.shape)
    return  img

def convert_to_grayscale_data(data):
    
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


def conv_to_grayscale_data(data):

    data_rgb = data
    
    data_gry = np.sum(data_rgb/3, 
                     axis=3, keepdims=True)

    
    assert(data_gry.shape[1:] == (32,32,1)), "The dimensions of the images are not 32 x 32 x 1."
    return data_gry


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

def combine_datasets(datasets):

    x=datasets[0][0]
    y = datasets[0][1]
    count=0 
    if type(x) is list:
            x = np.asarray(x)
            print(str(0)+' is a list')
    if type(y) is list:
            y = np.asarray(y)
        
    #print(x.shape)
    #print(y.shape)
    X_all=x
    y_all=y
    print('combining dataset '+str(0)+' of type '+str(type(x))+' and length '+str(len(x)))
            
    count = 0
    for ds in datasets:
        if count==0: #since we added the first one already
            count+=1
            continue
       
        #print(type(ds))
        x = ds[0]
        y=ds[1]
        print('combining dataset '+str(count)+' of type '+str(type(x))+' and length '+str(len(x)))
        if type(x) is list:
            x = np.asarray(x)
            print('Dataset '+str(count)+' is a list; converting to  ndarray')
        if type(y) is list:
            y = np.asarray(y)
        
        

        X_all=np.append(X_all,x,axis=0)
        y_all=np.append(y_all,y,axis=0)
        count+=1


   
    assert(len(X_all) == len(y_all))
    print('length of combined set : '+str(len(X_all)))
    return np.asarray(X_all),np.asarray(y_all)


def data_augment_classes(X,y,classes,n_gen_per_class,n_aug_per_class):

    print('Generating '+str(n_gen_per_class)+' number of images per class, selecting '+str(n_aug_per_class)+' number of images per class for  augmentation')
    print('No of classes:'+str(len(classes)))
    print('classes:'+str(classes))
    datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
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
    for i,classid in tqdm(enumerate(classes)):
    #for i in tqdm(range(3)):
            #print(i)
            index = [y==classid]
            images_for_classid = X[index]
            y_classid = y[y==classid]
            X_augmented_cid = np.empty(data_shape)
            #print(X_augmented_i.shape)
            y_augmented_cid = np.empty(0,dtype='uint8')
            for X_b,y_b in datagen.flow(images_for_classid, y_classid, batch_size=len(y_classid), seed=9345+i*37):            
                X_augmented_cid = np.append(X_augmented_cid, X_b, axis=0)
                y_augmented_cid = np.append(y_augmented_cid, y_b, axis=0)

                if len(X_augmented_cid) >= total_image_per_class:
                    break
            X_augmented_cid, y_augmented_cid = shuffle(X_augmented_cid, y_augmented_cid, random_state=9345)
            X_augmented = np.append(X_augmented, X_augmented_cid[:n_aug_per_class], axis=0)
            y_augmented = np.append(y_augmented, y_augmented_cid[:n_aug_per_class], axis=0)     
    print("shufle")
    X_augmented, y_augmented = shuffle(X_augmented, y_augmented, random_state=9345)
    print("X_augmented shape: "+str(X_augmented.shape))
    print("y_augmented shape: "+str(y_augmented.shape))
    return X_augmented, y_augmented
    
    # Storing for checkpoint2
    #X_augmented = X_augmented.astype('float32')
 
