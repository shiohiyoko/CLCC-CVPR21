import os
import cv2
import pickle
import math
import numpy as np
import tensorflow as tf
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
    image, illum, cc24 = data["image"].astype(np.float32),\
                            data["illum"].astype(np.float32),\
                            data["cc24"].astype(np.float32),
    return image, illum, cc24


if __name__ == '__main__':
    dir = 'data'
    name = 'gehler'
    fold = [0]
    path = load_paths(dir, name, fold)
    image, illum, cc24 = read_data(path)
    print('image',image) 
    # illum, cc24)
        
        