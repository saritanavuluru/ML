
import pandas as pd


def get_images_for_class(class_id,X,y):
   
    df = pd.DataFrame({'y':y})
    gg = df.groupby(by=y)
    #print(type(gg.groups))
    
    return X[gg.groups[class_id]]
    

def get_first_images(X,y):
    df = pd.DataFrame({'y':y})
    first_images=[]
    gg = df.groupby(by=y)
    #print(type(gg.groups))
    for key,val in gg.groups.items():
        #print("{} = {}".format(key, val[0]))
        first_images.append(val[0])
    return X[first_images]