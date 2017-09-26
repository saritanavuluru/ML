import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import csv
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

