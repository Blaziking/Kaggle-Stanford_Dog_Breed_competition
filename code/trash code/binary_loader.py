import dataset
import pickle 
import numpy as np
import os


"""###########################################################################

to do : 
1. club both the create_pickle and read_pickle under the same function
###########################################################################
"""
script_dir = os.path.dirname(__file__)



def create_pickle(pickle_dir, data):  #give it the file_name of the pickle and the data to be pickled
    
    with open(pickle_dir,'wb') as f:
        pickle_file =   pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done Pickling")
    return pickle_file
    

def read_pickle(pickle_dir):
    
    with open(pickle_dir,'rb') as f:
         loaded_obj = pickle.load(f)
    
    print("Done reading pickle file")
    return loaded_obj
            
