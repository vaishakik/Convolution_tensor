import time
import math
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import dataset
import cv2

from sklearn.metrics import confusion_matrix
from datetime import timedelta

# Convolutional Layer 1.
filter_size1 = 3 
num_filters1 = 32

# Convolutional Layer 2.
filter_size2 = 3
num_filters2 = 32

# Convolutional Layer 3.
filter_size3 = 3
num_filters3 = 64

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.

# Number of color channels for the images: 1 channel for gray-scale.
num_channels = 3

# image dimensions (only squares for now)
img_size = 128

# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# class info
classes = ['BM', 'CH','NA','PI','ST']
num_classes = len(classes)

# batch size
batch_size = 8

# validation split
validation_size = .20

# how long to wait after validation loss stops improving before terminating training
early_stopping = True  # use None if you don't want to implement early stoping

train_path = 'DL_data_New/train/train/'
test_path = 'DL_data_New/test/'
checkpoint_dir = "models/"
model_path = "/tmp/model.ckpt"
data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)
test_images, test_ids = dataset.read_test_set(test_path, img_size)
session = tf.Session()
#session.run(tf.initialize_all_variables())
#saver = tf.train.import_meta_graph('/tmp/model.ckpt.meta')
#saver.restore(session, model_path)
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('model11.meta')
saver.restore(session,tf.train.latest_checkpoint('./'))
graph = tf.get_default_graph()
    #y_pred_cls = graph.get_operation_by_name('final:0')
y_pred_cls = graph.get_tensor_by_name('y_pred:0')    
x = graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")
    ##
init_op = tf.initialize_all_variables()
    
session.run(init_op)
#session.run(tf.initialize_all_variables())
test_cat = cv2.imread('1.jpg')
test_cat = cv2.resize(test_cat, (img_size, img_size), cv2.INTER_LINEAR) / 255

preview_cat = plt.imshow(test_cat.reshape(img_size, img_size, num_channels))
test_dog = cv2.imread('3.jpg')
test_dog = cv2.resize(test_dog, (img_size, img_size), cv2.INTER_LINEAR) / 255
preview_dog = plt.imshow(test_dog.reshape(img_size, img_size, num_channels))
def sample_prediction(test_im):
    
    feed_dict_test = {
        x: test_im.reshape(1, img_size_flat),
        y_true: np.array([[0, 1, 2, 3, 4]])
    }

    test_pred = session.run(y_pred_cls, feed_dict=feed_dict_test)
    print (test_pred)
    return classes[test_pred[0]]
#sample_prediction(test_cat)
print("Predicted class for test_chariot: {}".format(sample_prediction(test_cat)))
print("Predicted class for test_pillars: {}".format(sample_prediction(test_dog)))
