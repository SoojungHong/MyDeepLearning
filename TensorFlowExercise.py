#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 22:17:38 2018

@author: soojunghong
"""

import tensorflow as tf

x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2 

# TensorFlow session take care placing operations onto devices such as CPU, GPU and running them
sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)
sess.close()

# to simplify instead of calling run() all the time
with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
    print(result)
    
# instead of initializing variable all the time, use global variable initializer
init = tf.global_variables_initializer() 

with tf.Session() as sess:
    init.run() 
    result = f.eval()    
    print(result)

# inside Jupyter or in Python shell, you can use InteractiveSession    
sess = tf.InteractiveSession() # automatically set itself as default and close 
init.run()
result = f.eval()
print(result)    

# managing the graph
x1 = tf.Variable(1)
x1.graph is tf.get_default_graph()

# creating multiple independent graph
graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2)
    
x2.graph is graph

x2.graph is tf.get_default_graph()

# Life cycle of Node value 
w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3 

with tf.Session() as sess:
    print(y.eval())
    print(z.eval())


# to avoid multiple evaluation of w and x 
with tf.Session() as sess:
    y_val, z_val = sess.run([y, z])
    print(y_val)
    print(z_val)    


# Linear Regression with TensorFlow 
import numpy as np
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
housing.data
housing.data.shape
m, n = housing.data.shape     
housing_data_plus_bias = np.c_[np.ones((m,1)), housing.data] # np.c_ means : concatenate  

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name="y")
XT = tf.transpose(X)
theta = tf.matmul(tf.matul(tf.matrix_inverse(tf.matmul(XT,X)), XT), y)

with tf.Session() as sess:
    theta_value = theta.eval()