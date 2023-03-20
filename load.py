import os
import cv2
import pickle
import math
import numpy as np
# import tensorflow as tf
import util.fc4_augmentation
from glob import glob
from dataset import ImageRecord
import sys
sys.path.append(os.path.abspath(".") + './')


def load_paths( data_dir, data_name, folds):
    paths = []
    for fold in folds:
        paths += glob(os.path.join('./',data_dir, data_name, str(fold), '*.pkl'))
        print('INFO:Loading dataset from "%s"...' % os.path.join(data_dir, data_name, str(fold)))
    return paths


def read_data(path):
    
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(type(data[0]))
    image, illum, cc24 = data[0].img.astype(np.float32),\
                            data[0].illum.astype(np.float32),\
                            data[0].cc24s.astype(np.float32),
    return image, illum, cc24


if __name__ == '__main__':
    dir = 'data'
    name = 'gehler'
    fold = [1]
    path = load_paths(dir, name, fold)
    image, illum, cc24 = read_data(path[0])
    print('image',image.shape) 
    print('illum',illum.shape)
    print('cc24', cc24.shape)
        
        