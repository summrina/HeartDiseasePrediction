'''This example demonstrates the use of Convolution1D for text classification.
Gets to 0.89 test accuracy after 2 epochs.
90s/epoch on Intel i5 2.4Ghz CPU.
10s/epoch on Tesla K40 GPU.
'''
from __future__ import print_function
from keras.callbacks import ReduceLROnPlateau
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
batch_size = 100
embedding_dims = 15
filters = 5
kernel_size = 5
hidden_dims = 100
epochs = 5

def CNNaccfunc(Ab, dataset, size):
    
    #dataset = loadtxt('Hotel-Bow.csv', delimiter=',')
    # split into input (X) and output (y) variables
    result_array = np.array([])
    for i in Ab:
        X = dataset[:,1:i+1]
        Y = dataset[:,0]
        max_features=i
        print(i);
        print(max_features)
        maxlen=max_features
        print(maxlen)
        print('Loading data...')
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0, shuffle=True)
        #X_train, X_test, Y_train, Y_test =train_test_split(X,
         #                                     Y, 
          #                                    test_size=0.1,
           #                                   random_state=2,
            #                                  shuffle=True,
             #                                 stratify=Y
              #                               )
        #print(X_train, X_test, Y_train, Y_test)
        kf = KFold(n_splits=10)
        print(len(X_train), 'train sequences')
        print(len(X_test), 'test sequences')

        print('Pad sequences (samples x time)')
        #X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
        #X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
        X_train = np.expand_dims(X_train, axis=2)
        X_test = np.expand_dims(X_test, axis=2)
        print('x_train shape:', X_train.shape)
        print('x_test shape:', X_test.shape)

        print('Build model...')
        model = Sequential()
        
        model.add(Conv1D(filters=10, kernel_size=5,
                        input_shape=(max_features,1), kernel_initializer='uniform',
                        activation='relu'))
# model.add(Flatten(input_shape=(45, 300)))
#model.add(Flatten(input_shape=(400,300)))
# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
# model.add(Embedding(max_features,
#                     embedding_dims,
#                     input_length=maxlen))

        model.add(Dropout(0.2))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
        model.add(Conv1D(filters=200,
                 kernel_size=5,
                 padding='valid',
                 activation='relu',
                 strides=1))
        model.add(Conv1D(filters=32,
                 kernel_size=5,
                 padding='valid',
                 activation='relu',
                 strides=1))
# we use
# we use max pooling:
        model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
        model.add(Dense(hidden_dims))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))

        # We project onto a single unit output layer, and squash it with a sigmoid:
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        lr_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                 patience=10, 
                                 verbose=1, 
                                 factor=0.5, 
                                 min_lr=0.00001)
        model.summary();
        model.fit(X_train, Y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(X_test, Y_test));#, callbacks=[lr_reduction])
       
        accuracy = model.evaluate(X_test, Y_test);
        K.clear_session()
        gc.collect();
        result_array = np.append(result_array, accuracy[1]);
        print(accuracy[1])
        print(i)
        
        wait = input("PRESS ENTER TO CONTINUE.")
    #print((result_array))
    return result_array                  
#plt.plot(epochs, y_test)
# plt.axis(epochs)
# plt.show()
