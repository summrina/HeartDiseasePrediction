'''This example demonstrates the use of Convolution1D for text classification.
Gets to 0.89 test accuracy after 2 epochs.
90s/epoch on Intel i5 2.4Ghz CPU.
10s/epoch on Tesla K40 GPU.
'''
from __future__ import print_function
import tensorflow as  tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras import backend as K
import gc
#from keras.datasets import imdb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from numpy import loadtxt
##SoftMax Layer######
#data1 = np.load('input.npy')
#print(data1.shape)
#data2 = np.load('output.npy')
#print(data2.shape)
#A = pd.read_csv('C:/MRI 2019/brca_metabric/data_clinical_patient_updated.csv', delimiter=',')
    # split into input (X) and output (y) variables
#A = pd.read_csv('C:/MRI 2019/brca_metabric/out_updated.csv', delimiter=',')
#data_CNA_transposed
#A = pd.read_csv('D:\Python courses\Opt-aiNet/Hotel-Bow.csv', delimiter=',')
 
max_features = 200 
#16382 #5000 #8724  
#dataset=A.to_numpy()
#X = dataset[:,1:201]
#Y = dataset[:,0]
# set parameters:
maxlen = max_features
batch_size = 28
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 10

def mLSTM(Ab, dataset, size):
    
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
        # bulid model
        model = Sequential()
        model.add(LSTM(128, input_length=i, activation='relu', return_sequences=True))
        model.add(Dropout(0.2))
        
        model.add(LSTM(128, activation='relu'))
        model.add(Dropout(0.2))
        
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        
        model.add(Dense(10, activation='softmax'))
        opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)
        # compile the keras model
        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt,
             metrics=['accuracy'])
        #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']);
        # fit the keras model on the dataset
        model.fit(X_train, Y_train, validation_split=0.30, epochs=10, batch_size=10);
        # evaluate the keras model
        accuracy = model.evaluate(X_test, Y_test);
        K.clear_session()
        gc.collect();
        result_array = np.append(result_array, accuracy[1]);
        print(model.metrics_names)
        print(accuracy)
        print(i)
   # wait = input("PRESS ENTER TO CONTINUE.")
    #print((result_array))
    return result_array                  
#plt.plot(epochs, y_test)
# plt.axis(epochs)
# plt.show()
