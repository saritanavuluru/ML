import matplotlib.pyplot as plt
from keras.utils import np_utils
import numpy as np
from tqdm import tqdm
import math


def plot_image(image, nr, nc, i, label=""):
    """
    Plot a single image.
    If 'i' is greater than 0, then plot this image as 
    a subplot of a larger plot.
    """
    
    if i>0: 
        plt.subplot(nr, nc, i)
    else:
        plt.figure(figsize=(nr,nc))
        
    plt.xticks(())
    plt.yticks(())
    plt.xlabel(label)
    plt.tight_layout()
    plt.imshow(image, cmap="gray")

#plot images
def plot_images(img_class, images, ncol,desc):
    """
    Plot all of the images in the list 'class_images'
    """
        
    if type(images) is list:
        images = np.asarray(images)
        #print("input is a list, converting to ndarray")
    nimages = len(images)
    nrow = math.ceil(nimages/ncol)
    plt.figure(figsize=(3*nrow,ncol))
    if (img_class == -1):
        print("There are {} images in the {} dataset".format(nimages, desc))
    else:
        print("class {} has {} images in the {} dataset".format(img_class,nimages, desc))
    for image,i in zip(images, range(nimages)):
        #print(i)
        plot_image(image.squeeze(), nrow, ncol, i+1)
        



def plot_2_datasets(dataset1, dataset2, label1, label2, count=0):
    
        if count==0 :
            count = len(dataset1)
        
      
        nc = 2
        nr = count
        
        #plt.figure(figsize=(nr,nc))
        for i in range(count):
        
            plot_image(dataset1[i].squeeze(), nr, nc, 2*i+1,label1)
            
            image = dataset2[i].squeeze()
            plot_image(image, nr, nc, 2*i+2, label2)
            #print("image mean=", image.mean())




def get_dark_bright_imgs(imgs,mean,std):


    dark = []
    bright =[]
    dark_mean = []
    bright_mean = []
    #plot_images(11,class_images[0:100],10,"")
    for img in imgs:

        img_mean = np.mean(img)
        #print(img_mean)
        if (img_mean < mean):

            dark.append(img)
            dark_mean.append(img_mean)
        else:
            bright.append(img)
            bright_mean.append(img_mean)

    return dark,bright,dark_mean,bright_mean

def plot_equalize_color_gray(img_dark,img_dark_equal, img_dark_equal_gry, img_bright,img_bright_equal, img_bright_equal_gry,op):
    plot_image(img_dark, 2,4,1, "Original Dark")
    plot_image(img_dark_equal, 2,4,2,op+" Dark")
    plot_image(img_dark_gry, 2,4,3, "Original Dark Grey")

    plot_image(img_dark_equal_gry,2,4,4, op+" Dark Grey")
    plot_image(img_bright, 2,4,5, "Original Bright")
    plot_image(img_bright_equal,2,4,6, op+" Bright")
    plot_image(img_bright_gry, 2,4,7, "Original Bright Grey")

    plot_image(img_bright_equal_gry,2,4,8, op+" Bright Grey")

