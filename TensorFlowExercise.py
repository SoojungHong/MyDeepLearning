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
import tensorflow as tf    
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
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT,X)), XT), y)

with tf.Session() as sess:
    theta_value = theta.eval()
    print(theta_value)
    

# Manually computing the gradients 
tf.reset_default_graph()    
n_epochs = 1000
learning_rate = 0.01

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m,1)), scaled_housing_data]

print(scaled_housing_data_plus_bias.mean(axis=0)) # axis = 0 row 
print(scaled_housing_data_plus_bias.mean(axis=1)) # axis = 1 column 
print(scaled_housing_data_plus_bias.mean()) # axis = 1 column 
print(scaled_housing_data_plus_bias.shape) # axis = 1 column 

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n+1,1], -1.0, 1.0, seed = 42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
gradients = 2/m * tf.matmul(tf.transpose(X), error)
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE=", mse.eval())
        sess.run(training_op)
    
    best_theta = theta.eval()    

print("Best theta: ")
print(best_theta) 

# using autodiff 
tf.reset_default_graph()

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
gradients = tf.gradients(mse, [theta])[0]
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)
    
    best_theta = theta.eval()

print("Best theta:")
print(best_theta)    

# using Gradient Descent Optimizer
tf.reset_default_graph()
n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n+1,1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)
    
    best_theta = theta.eval()

print("Best theta:")
print(best_theta)    