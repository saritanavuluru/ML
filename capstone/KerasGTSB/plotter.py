import matplotlib.pyplot as plt
from keras.utils import np_utils
import numpy as np
from tqdm import tqdm
import math
import params
from math import sqrt
import seaborn as sns


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
        
def plot_first_images(first_images):
    
    nr = 10 ; nc = 10
    plt.figure(figsize=(nr,nc))
    for image,i in zip(first_images, range(len(first_images))):
        label = (str(i) + "\n"                                            # class
              + '('+str(sign_names['NumTrain'][i])+')')                              # no. of training samples
              
        plot_image(image.squeeze(), nr, nc, i+1, label)
    



def plot_2_datasets(dataset1, dataset2, label1, label2, count=0):
    
        if count==0 :
            count = len(dataset1)
        
      
        nc = 2
        nr = count
        
        plt.figure(figsize=(10,10))
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


def plot_top_n_preds(X_test_new,prob_new,sign_names,top_n):

    topn_preds = (np.argsort(-prob_new))[:,:top_n]
    topn_pred_names = sign_names[topn_preds]
    #print(prob_new.shape)
    for i in range(topn_preds.shape[0]):
        #plotter.plot_image(X_test_new[i].squeeze(),x.shape[0],2,2*(i)+1,"")
        indices = topn_preds[i,:]
        topn_probs = prob_new[i,[indices]][0]
        #print(top5_probs)
        names = topn_pred_names[i,:]
        #print(names)
        implot = plt.figure(figsize=(10,4))

        ax = implot.add_subplot(1,2,1)
        ax.grid(False)
        ax.axis('off')
        ax.imshow(X_test_new[i].squeeze(),cmap='gray')


        y_pos = np.arange(len(names))
        y_pos=y_pos[::-1]
        

        ax2 = implot.add_subplot(1,2,2)
        #print(top5_probs)
        ax2.barh(y_pos, topn_probs, align='center', color="purple")
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(names)
        #print(names)
        ax2.yaxis.set_ticks_position('right')
        ax2.set_xlabel('Top '+str(top_n)+' Probabilities')
        ax2.set_title('Classes')
        ax2.set_xlim([0, 1])

        plt.tight_layout()

        plt.show()
def plot_train_loss_accuracy(model_history):
    sns.set_style("whitegrid")
    #sns.set_context("poster")
    

    fig, (ax0,ax1) = plt.subplots(ncols=2,nrows=1,figsize=(20, 5), sharey=False)
    

    ax0.plot(model_history['acc'], 'r')  
    ax0.plot(model_history['val_acc'],'g')  
    ax0.set_title('Model (Train vs Validation) Accuracy')  
    ax0.set_ylabel('Accuracy',fontsize=14,fontweight='bold')  
    ax0.set_xlabel('Epoch',fontsize=14,fontweight='bold')  
    ax0.legend(['train', 'validation'], loc='lower right', fontsize=14)  
    

    ax1.plot(model_history['loss'],'r')  
    ax1.plot(model_history['val_loss'],'g')  
    ax1.set_title('Model (Train vs Validation) Loss')  
    ax1.set_ylabel('Loss',fontsize=14,fontweight='bold')  
    ax1.set_xlabel('Epoch',fontsize=14,fontweight='bold')  
    ax1.legend(['train', 'validation'], loc='upper right',fontsize=14)  
       
# modified from original Function by gcalmettes from http://stackoverflow.com/questions/11159436/multiple-figures-in-a-single-window
def plot_figures(figures, dims):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """
    ncols = dims[0]
    nrows = dims[1]
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind,figure in enumerate(figures):
        #print(ind)
        axeslist.ravel()[ind].imshow(figure, cmap=plt.gray())
       

    for ind in range(nrows*ncols):
        axeslist.ravel()[ind].set_axis_off()

    
    plt.show()


def get_dim(num):
    """
    Simple function to get the dimensions of a square-ish shape for plotting
    num images
    """

    s = sqrt(num)
    if round(s) < s:
        return (int(s), int(s)+1)
    else:
        return (int(s)+1, int(s)+1)

def compare_train_loss_accuracy(history1, history2,model_name1, model_name2):
    sns.set_style("whitegrid")
    sns.set_context("poster")
    
    fig, (ax0,ax1) = plt.subplots(ncols=2,nrows=1,figsize=(20, 5), sharey=False)
    

    ax0.plot(history1['acc'], 'r')  
    ax0.plot(history2['acc'],'b')  
    ax0.set_title('Compare Accuracy')  
    ax0.set_ylabel('Accuracy',fontsize=14,fontweight='bold')  
    ax0.set_xlabel('Epoch',fontsize=14,fontweight='bold')  
    ax0.legend([model_name1, model_name2 ], loc='lower right', fontsize=14)  
    
     # summarize history for loss  

    ax1.plot(history1['loss'],'r')  
    ax1.plot(history2['loss'],'b')  
    ax1.set_title('Compare Loss')  
    ax1.set_ylabel('Loss',fontsize=14,fontweight='bold')  
    ax1.set_xlabel('Epoch',fontsize=14,fontweight='bold')  
    ax1.legend([model_name1, model_name2 ], loc='upper right',fontsize=14)  

    
def plot_equalize_color_gray(img_dark,img_dark_equal, img_dark_equal_gry, img_bright,img_bright_equal, img_bright_equal_gry,op):
    plot_image(img_dark, 2,4,1, "Original Dark")
    plot_image(img_dark_equal, 2,4,2,op+" Dark")
    plot_image(img_dark_gry, 2,4,3, "Original Dark Grey")

    plot_image(img_dark_equal_gry,2,4,4, op+" Dark Grey")
    plot_image(img_bright, 2,4,5, "Original Bright")
    plot_image(img_bright_equal,2,4,6, op+" Bright")
    plot_image(img_bright_gry, 2,4,7, "Original Bright Grey")

    plot_image(img_bright_equal_gry,2,4,8, op+" Bright Grey")

