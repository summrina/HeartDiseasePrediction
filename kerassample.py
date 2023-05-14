
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 12:13:46 2019

@author: DELL
"""


# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras import backend as K
import gc
import time
import pandas as pd
from keras.layers import Dense
import numpy as np
#from opt_aiNet_fs_H import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
# load the dataset#
def kerasaccfunc(Ab, dataset,size):
    
#dataset = pd.read_csv('Heart new.csv', delimiter=',')
    #split into input (X) and output (y) variables
    result_array = np.array([])
    for i in Ab:
            #i=2000
            X = dataset.iloc[:,0:i]
            Y = dataset.iloc[:,-1]
            # define the keras model
            print('Loading data...')
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0, shuffle=True);
            #kf = KFold(n_splits=10)
            model = Sequential();
            model.add(Dense(100, input_dim=i, activation='relu'));
            #model.add(Dense(50, activation='relu'));
            model.add(Dense(1, activation='sigmoid'));
            # compile the keras model
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']);
            # fit the keras model on the dataset
            model.summary()
            wait = input("PRESS ENTER TO CONTINUE.")
            model.fit(X_train, Y_train, validation_split=0.30, epochs=20, batch_size=5);
            # evaluate the keras model
    #        start_time = time.time()
     #       print(start_time-start_time)
            accuracy = model.evaluate(X_test, Y_test);
            #model.predict()
            K.clear_session()
            gc.collect();
            result_array = np.append(result_array, accuracy[1]);
            print(model.metrics_names)
            print(accuracy)
            print(i)
            wait = input("PRESS ENTER TO CONTINUE.")
    print((result_array))
        #return result_array
    
