import shutil
import tensorflow as tf
import numpy as np
import os.path
import os
from sklearn.preprocessing import LabelEncoder
import csv

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
    old_files_dir = [train_dir + image+'/' for image in files]
    new_files_dir = [destination_directory + image+'/' for destination_directory,image in zip(new_dirs,files)]
    map(lambda path:os.makedirs(path,exist_ok = True),new_files_dir)
    map(lambda src,dest:shutil.copyfile(src,dest),old_files_dir,new_files_dir)

rearrange_data(home_dir + 'data/rearranged_data/')
print('works')    

def load_data(divider=9000):
    '''Arguments:
            divider -- point to divide training and validation dataset
   Returns:
            dictionary of dataset containing tf.data.Dataset type objects for validation and test set


    '''
    # Reads an image from a file, decodes it into a dense tensor, and resizes it to a shape of [batch_size,height,width,channels.
    
    def parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_image(image_string)
        image_normalized = tf.image.per_image_standardization(image_decoded)
        image_resized = tf.image.resize_image_with_crop_or_pad(image_normalized, 256, 256)
        return image_resized, label

    
    labels,files = label_loader(labels_dir)
    labels= one_hot_encode(labels)
    files = [train_dir + x for x in files] #adding home dir on each file name
    training_dataset = tf.constant(files[:divider])
    training_labels = labels[:divider]
    assert(training_dataset.get_shape() == training_labels.get_shape()[0])

    training_dataset = tf.data.Dataset.from_tensor_slices((training_dataset,training_labels))
    training_dataset = training_dataset.map(parse_function)
    validation_dataset = tf.constant(files[divider:])
    validation_labels = labels[divider:]
    assert(validation_dataset.get_shape() == validation_labels.get_shape()[0])

    validation_dataset = tf.data.Dataset.from_tensor_slices((validation_dataset,validation_labels))
    validation_dataset = validation_dataset.map(parse_function)

    dataset = {'training_dataset':training_dataset,'validation_dataset':validation_dataset}

    print('Data successfully imported')    
    return dataset




#run the following code to test it


# y = load_data()
# training_dataset = y['training_dataset']
# iterator = training_dataset.make_one_shot_iterator()
# with tf.Session() as sess:
#      next_element = iterator.get_next()
#      print((sess.run(next_element)))
#      print(np.array((sess.run(next_element)))[0].shape)
#      print((sess.run(next_element))[0].dtype)

