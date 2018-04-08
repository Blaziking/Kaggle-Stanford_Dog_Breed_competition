import label_image
import os
import csv

home_dir =os.path.dirname( os.path.dirname(os.path.realpath('__file__'))) + '/'

images = os.listdir(home_dir + '/data/test/')

images_dir = [image for image in images]

with open('test_lables.csv', 'w') as csvfile:
     labels = label_image.load_labels(label_file = home_dir + 'data/transfer_modeldata/output_labels.txt')
     labels.insert(0,'id')
     print(labels)
     spamwriter = csv.writer(csvfile, delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
     spamwriter.writerow(labels)

for image in images_dir:
    label_image.main(model_file =  home_dir + 'data/transfer_modeldata/output_graph.pb',file_dir = home_dir + 'data/test/' + image,  label_file = home_dir+ 'data/transfer_modeldata/output_labels.txt',     input_height = 299,input_width = 299,     input_mean = 0,input_std = 255,input_layer = 'Mul',output_layer = 'final_result')

    
