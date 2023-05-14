
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
import matplotlib.pyplot as plt
from keras.layers import Dense
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
#from opt_aiNet_fs_H import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import seaborn as sns
# load the dataset
def kerasaccfunc(Ab, dataset,size):
    
    #dataset = loadtxt('Hotel-Bow.csv', delimiter=',')
    # split into input (X) and output (y) variables
    result_array = np.array([])
    for i in Ab:
        X = dataset[:,1:i+1]
        Y = dataset[:,0]
        # define the keras model
        print('Loading data...')
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0);
        #kf = KFold(n_splits=10)
        model = Sequential();
        model.add(Dense(100, input_dim=i, activation='relu'));
        model.add(Dense(50, activation='relu'));
        model.add(Dense(1, activation='sigmoid'));
        # compile the keras model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']);
        # fit the keras model on the dataset
        model.summary()
        wait = input("PRESS ENTER TO CONTINUE.")
        history=model.fit(X_train, Y_train, validation_split=0.30, epochs=20, batch_size=50);
        # evaluate the keras model
#        start_time = time.time()
 #       print(start_time-start_time)
        accuracy = model.evaluate(X_test, Y_test);
        y_pred=model.predict(X_test)
        #print(y_pred);
        cf_matrix=confusion_matrix(Y_test,np.rint(y_pred));
        fig = plt.figure()
        plt.matshow(cf_matrix)
        plt.title('Confusion Matrix for COVID 19')
        plt.colorbar()
        plt.ylabel('True Label')
        plt.xlabel('Predicated Label')
        plt.savefig('confusion_matrix.png', dpi=500)
        #plt.matshow(cf_matrix)
        #sns.heatmap(cf_matrix, annot=True)
        plt.figure(figsize = (10,7))
        a=sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')
        fig=a.get_figure()
        fig.savefig('cscm.png', dpi=500) #model.predict()
        print(classification_report(Y_test,np.rint(y_pred))) 
        K.clear_session()
        gc.collect();
        result_array = np.append(result_array, accuracy[1]);
        print(model.metrics_names)
        print(accuracy)
        print(i)
        wait = input("PRESS ENTER TO CONTINUE.")
    #print((result_array))
    return cf_matrix, result_array

