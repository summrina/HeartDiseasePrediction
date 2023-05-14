
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
import Plot_3D
from keras.layers import Dense
import pylab as pl
import numpy as np
#from opt_aiNet_fs_H import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# load the dataset
def mySVM(Ab, dataset,size):
    
    #dataset = loadtxt('Hotel-Bow.csv', delimiter=',')
    # split into input (X) and output (y) variables
    result_array = np.array([])
    for i in Ab:
        i=67500
        X = dataset[:,1:i+1]
        Y = dataset[:,0]
        # define the keras model
        print('Loading data...')
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0);
        #kf = KFold(n_splits=10)
        svclassifier = SVC(kernel='rbf' )
        stime=time.time()
        clf=svclassifier.fit(X_train, Y_train)
        y_pred = svclassifier.predict(X_test)
        #model.fit(X_train, Y_train, validation_split=0.30, epochs=20, batch_size=50);
        # evaluate the keras model
        etime = time.time()
        print("time"+str(etime-stime))
        print(confusion_matrix(Y_test,y_pred))
        print(classification_report(Y_test,y_pred))       
        print("fs"+str(i))
        #Plot_3D.Plot_3D(X, X_test,Y_test,clf)
        wait = input("PRESS ENTER TO CONTINUE.")
    #print((result_array))
        svclassifier = SVC(kernel='linear' )
        stime=time.time()
        clf=svclassifier.fit(X_train, Y_train)
        y_pred = svclassifier.predict(X_test)
        #model.fit(X_train, Y_train, validation_split=0.30, epochs=20, batch_size=50);
        # evaluate the keras model
        etime = time.time()
        print("time"+str(etime-stime))
        print(confusion_matrix(Y_test,y_pred))
        print(classification_report(Y_test,y_pred))       
        print("fs"+str(i))
        #Plot_3D.Plot_3D(X, X_test,Y_test,clf)
        wait = input("PRESS ENTER TO CONTINUE.")
        svclassifier = SVC(kernel='poly' )
        stime=time.time()
        clf=svclassifier.fit(X_train, Y_train)
        y_pred = svclassifier.predict(X_test)
        #model.fit(X_train, Y_train, validation_split=0.30, epochs=20, batch_size=50);
        # evaluate the keras model
        etime = time.time()
        print("time"+str(etime-stime))
        print(confusion_matrix(Y_test,y_pred))
        print(classification_report(Y_test,y_pred))       
        print("fs"+str(i))
        #Plot_3D.Plot_3D(X, X_test,Y_test,clf)
        wait = input("PRESS ENTER TO CONTINUE.")
 
    return result_array

