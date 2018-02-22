import os
import csv
import numpy as np
from PIL import Image
import binary_loader


width = 256
height = 256


def label_loader(path_labels):
    ''' 
    #use as labels[id] = image-label
    '''
    

    #converting the labels.csv into a dictionary
    labels =[]
    files = []
    
    with open(path_labels, encoding='utf-8') as label_file:
        reader = csv.reader(label_file)
        next(reader)
        for row in reader:
            files.append(row[0]+'.jpg')
            labels.append(row[1])
    
    return labels,files

# def image_loader(path_train):
    
#     paths = os.listdir(path_train) #collecting all names of the images

#     #extracting image data into a numpy array
#     images = {}
#     id_order =[]
    
#     for i in paths: 
#         ied = i.split('.',1)[0]#removing .jpg from the end in path labels
#         id_order.append(str(ied))
#     for i in paths:
#         ied = i.split('.',1)[0]#removing .jpg from the end in path labels
#         image_info = (Image.open(path_train + "/" + str(i)))#importing all images   and converting all to numpy arrays
#         image_info = np.asarray(image_info.resize((width,height), Image.NEAREST))
#         image_info = image_info.reshape(image_info.shape[0]*image_info.shape[1]*3,1) #reshaping the images.
#         #image preprcessing step by subtracting mean and dividing by variance
#         images[ied] = (image_info - np.mean(image_info,axis=0))/np.var(image_info,axis=0) 

      
#     data = {'images':images,'id_order':id_order}




#     return data
            
    
