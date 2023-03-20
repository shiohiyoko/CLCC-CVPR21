import os
import cv2
import pickle
import math
import numpy as np
import tensorflow as tf
import util.fc4_augmentation
from glob import glob
from dataset import ImageRecord
        
class Dataloader:
    def __init__(self,
                 data_dir,
                 data_name,
                 folds,
                 batch_size,
                 is_training):
        """ Refactored from FC4 data_provider.py 
        """

        self.paths = self.load_paths(data_dir, data_name, folds)
        self.data_count = len(self.paths)
        self.batch_size = batch_size
        self.is_training = is_training
        
        if self.is_training:
            self.batch_size = batch_size
            self.preprocess = self.preprocess_train
        else:
            self.batch_size = 1 # Test different [h,w]. Don't batch.
            self.preprocess = self.preprocess_test
            
        data = [self.paths]
        dataset = tf.data.Dataset.zip(tuple([tf.data.Dataset.from_tensor_slices(x) for x in data]))
        dataset = dataset.map(self.preprocess, num_parallel_calls=16)
        dataset = dataset.shuffle(buffer_size=len(data[0])) if self.is_training else dataset
        dataset = dataset.repeat()
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.batch_size * 4)

        self.data = data
        self.dataset = dataset
        self.iterator = tf.compat.v1.data.make_one_shot_iterator(self.dataset)
        self.get_batch = self.iterator.get_next()
    
    def load_paths(self, data_dir, data_name, folds):
        paths = []
        for fold in folds:
            paths += glob(os.path.join('./',data_dir, data_name, str(fold), '*.pkl'))
            print('INFO:Loading dataset from "%s"...' % os.path.join(data_dir, data_name, str(fold)))
        return paths
    
    def read_data(self, path):
        def _func(path):
            
            with open(path, 'rb') as f:
                data = pickle.load(f)
            image, illum, cc24 = data.img.astype(np.float32),\
                                 data.illum.astype(np.float32),\
                                 data.cc24s.astype(np.float32)
            return image, illum, cc24
            
        return tf.compat.v1.py_func(_func, [path], [tf.float32, tf.float32, tf.float32], stateful=False)
            
    def preprocess_test(self, path):
        def _func(image, illum, cc24):
            image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
            return image.astype(np.float32), illum.astype(np.float32), cc24.astype(np.float32)
        
        image, illum, cc24 = self.read_data(path)
        return tf.compat.v1.py_func(_func, [image, illum, cc24], [tf.float32, tf.float32, tf.float32], stateful=False)

    def preprocess_train(self, path):
        def _func(image, illum, cc24):
            cc24 = cc24[...,::-1].reshape(-1,3) # ***BGR => RGB***
            assert cc24.shape[0] == 24, 'Color checker should be 24 colors.'

            # Check dataset GT is consistenct with the one in CC24.
            # And also check CC24 is RGB (same as illum).
            norm_cc24 = (cc24 / np.linalg.norm(cc24, axis=-1, keepdims=True))
            errors = np.abs(norm_cc24[...,1]/norm_cc24[...,0] - illum[...,1]/illum[...,0]) + \
                        np.abs(norm_cc24[...,1]/norm_cc24[...,2] - illum[...,1]/illum[...,2])        
            gt_idx = np.argmin(errors)

            try:
                assert gt_idx in [23,22,21,20,19,18], ' Gray indices '
                cc24[gt_idx] = illum # Replace with dataset GT.
            except Exception as e:
                print(e)
                pass
                # print('Image "%s": Color checker GT is not a valid idx (%d != [18-23]).' % (fn, gt_idx))

            new_image, new_illum, new_cc24 = util.fc4_augmentation.augment(image, illum, cc24)

            return new_image.astype(np.float32), new_illum.astype(np.float32), new_cc24.astype(np.float32)
        
        image, illum, cc24 = self.read_data(path)
        return tf.compat.v1.py_func(_func, [image, illum, cc24], [tf.float32, tf.float32, tf.float32], stateful=False)