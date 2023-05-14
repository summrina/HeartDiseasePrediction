id# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 16:58:23 2019

@author: hp
"""

import statsmodels.api as sm
import pandas as pd

import matplotlib.pyplot as plt
#from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
#from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
#import statsmodels.api as sm
import numpy as np

def LR(Ab, dataset,size):
    
    #dataset = loadtxt('Hotel-Bow.csv', delimiter=',')
    # split into input (X) and output (y) variables
    result_array = np.array([])
    for i in Ab:
        
        X = dataset.iloc[:,1:i+1]
        Y = dataset.iloc[:,0]
        x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)
        reg= LogisticRegression(solver='saga')
        reg.fit(x_train,y_train)
        Y_pred=reg.predict(x_test)
        #ols ordinary least square value diff ko kam karta ha
        #ols=sm.add_constant(X)
        #summary of ols
        #results=sm.OLS(Y,X).fit()
        #matrix = confusion_matrix(y_test, Y_pred)
        #print(matrix)
        report = classification_report(y_test, Y_pred)
        print(report)
        print("feature"+str(i))
        accuracy=accuracy_score(y_test, Y_pred)
        #print(accuracy)
        result_array = np.append(result_array, accuracy);
        print(accuracy)
    #print(i)
    #wait = input("PRESS ENTER TO CONTINUE.")
    #print((result_array))
    return result_array

    