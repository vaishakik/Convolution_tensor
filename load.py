import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


img_size = 192
img_size_flat=  img_size * img_size * 3
classes = ['BM', 'CH','PI']
model_file_name = 'savedModel/cnn_5class.meta'

model_folder = 'savedModel/'

sess=tf.Session()    

saver = tf.train.import_meta_graph(model_file_name)
  
saver.restore(sess,tf.train.latest_checkpoint(model_folder))
    
graph = tf.get_default_graph()
    
    #y_pred_cls = graph.get_operation_by_name('final:0')
y_pred_cls = graph.get_tensor_by_name('y_pred:0')
    
x = graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")
    ##
#init_op = tf.global_variables_initializer()
    
#sess.run(init_op)

test_cat = cv2.imread('3.jpg')
test_cat = cv2.resize(test_cat, (img_size, img_size), cv2.INTER_LINEAR) / 255





feed_dict_test = {
    x: test_cat.reshape(1, img_size_flat),
        #y_true: np.array([[0, 0,0]])       #3
    y_true: np.array([[1,2,3]])  #5
           # y_true: np.array([[0,0]]) 
    #    y_true: np.array([[0, 0,0,0]])    #4
    }
test_pred = sess.run(y_pred_cls, feed_dict=feed_dict_test)
print(test_pred)
index_max = np.argmax(test_pred)
#index_min = min(xrange(len(test_pred)), key=test_pred.__getitem__)
#print(classes[test_pred[0]])
print(classes[index_max])

