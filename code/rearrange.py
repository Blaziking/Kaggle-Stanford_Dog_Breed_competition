import shutil
import tensorflow as tf
import numpy as np
import os.path
import os
from sklearn.preprocessing import LabelEncoder
import csv


'''script for rearranging data for direct use for transfer learning

'''
home_dir = '/home/shubham/Desktop/projects/dog-breed-competition/'
train_dir = home_dir + 'data/train/'
test_dir = home_dir + 'data/test/'
labels_dir = home_dir + 'data/labels.csv'

def one_hot_encode(labels,depth=120):
    encoder = LabelEncoder() #to be later used for one_hot_encoding.
    encoder = encoder.fit(np.squeeze(labels))
    labels = encoder.transform(np.squeeze(labels)) #labels converted into numbers
    labels = tf.one_hot(labels,depth= depth) #labels represented via one_hot representation
    return labels

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

def rearrange_data(rearrange_dir):
    labels,files = label_loader(labels_dir)
    new_dirs = [rearrange_dir + label +'/' for label in labels]
    old_files_dir = [train_dir + image for image in files]
    new_files_dir = new_dirs
    for path in new_files_dir:
        os.makedirs(path,exist_ok=True)

    for src,dest in zip(old_files_dir,new_files_dir):
        shutil.copy(src,dest)
rearrange_data(home_dir + 'data/rearranged_data/')
print('works')    
