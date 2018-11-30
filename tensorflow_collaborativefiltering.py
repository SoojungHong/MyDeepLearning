# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 16:44:48 2018

@author : shong
@about : 
"""

#====================
# import libraries
#====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import random


#====================
# data and header
#====================
CHICAGO_DATA = 'C:/Users/shong/Documents/data/chicago1year_train_clean_newclients_dedupe_chain_center_click_restaurants_15112018.tsv'
CHICAGO_HEADER = ['cookie', 'ppid', 'result_provider', 'click_time', 'query_string', 'query_language', 'categories', 'place_name', 'lat', 'lon', 'click_order', 'chain_info', 'unknown_col1', 'unknown_col2']


#=============
# functions 
#=============
def readData(dataset, header):
    file_path = dataset
    headers = header
    data = pd.read_csv(file_path, sep='\t', names=headers, error_bad_lines=False)
    return data 


def initDataWithHeader(dataset, header): 
    data = readData(dataset, header)
    data['rating'] = 1
    data['cookie_int'] = pd.factorize(data.cookie)[0]
    data['ppid_int'] = pd.factorize(data.ppid)[0]
    return data


def getUserItemMatrix(DATA, HEADER): 
    data = pd.read_csv(DATA, sep='\t', names=HEADER, error_bad_lines=False)

    data['rating'] = 1
    data['cookie_int'] = pd.factorize(data.cookie)[0]
    data['ppid_int'] = pd.factorize(data.ppid)[0]

    print(data)     
   
    
def get_occurences(records):
    #used to filter records based on how many click a certain user or ppid recieved
    occurs = {}
    for record in records:
        if record not in occurs:
            occurs[record] = 1
        else:
            occurs[record] += 1

    return occurs



#================
# experiment 
#================    
data = initDataWithHeader(CHICAGO_DATA, CHICAGO_HEADER)
data.cookie_int.unique() # 0 ~ 5817
data.ppid_int.unique() # 0 ~ 5456
item_user_data = csr_matrix((data.rating, (data.ppid_int, data.cookie_int)))

# for each cell in matrix 
num_ppid = len(data.ppid_int.unique()) # 5457
num_cookie = len(data.cookie_int.unique()) # 5818

data_all = []
features = [] # features : (ppid_id, cookie_id)
label_all = []
""" # this is very slow!! 
for p_idx in range(num_ppid): 
    for c_idx in range(num_cookie): 
        features.append(p_idx)
        features.append(c_idx)
        print(features) # avoid print - time consuming! 
        data_all.append(features)
        
        if (item_user_data[p_idx, c_idx] > 1):
            label_all.append(1)
        else: 
            label_all.append(0)
        
        features = [] # init 
""" 
for p_idx in range(num_ppid): 
    for c_idx in range(num_cookie): 
        features.append(p_idx)
        features.append(c_idx)
        data_all.append(features)
        
        if (item_user_data[p_idx, c_idx] > 1):
            label_all.append(1)
        else: 
            label_all.append(0)
               
        features = []

# TODO : instead of using nested for loop, use other type of matrix -> https://cmdlinetips.com/2018/03/sparse-matrices-in-python-with-scipy/


# make train and test set
percentage_test=0.2
num_testset = int(np.ceil(percentage_test*len(label_all))) 
num_testset #6349766
testset = random.sample(data_all, num_testset)

testset_label = []
for testIdx in range(len(testset)): 
    testset_label.append(item_user_data[testset[testIdx][0], testset[testIdx][1]])

# remove test data from data_all and label_all 
for i in range(len(testset)):
    if(testset_label[i] > 0): 
        test_index = data_all.index(testset[i])
        del label_all[test_index]
        data_all.remove(testset[i])

# convert list to array 
import numpy as np
training_data = np.asarray(data_all)
training_label = np.asarray(label_all)
test_data = np.asarray(testset)
test_label = np.asarray(testset_label)

import tensorflow as tf 
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(training_data)
print(feature_columns)
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[300, 100], n_classes=2, feature_columns=feature_columns)
dnn_clf
dnn_clf.fit(x=training_data, y=training_label, batch_size=50, steps=40000)

from sklearn.metrics import accuracy_score
label_pred = list(dnn_clf.predict(test_data))
accuracy = accuracy_score(test_label, label_pred)
accuracy  #0.99

from sklearn.metrics import log_loss
y_pred_prob = list(dnn_clf.predict_proba(test_data))
eval = log_loss(test_label, y_pred_prob)

# print prediction 
print (np.unique(y_pred_prob))
