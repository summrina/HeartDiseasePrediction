from __future__ import print_function
import numpy as np
from sklearn.metrics import roc_auc_score
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split#
#A = pd.read_csv('data_expression_median_finalized_DNN.csv', delimiter=',')
#

# set parameters:
max_features = size=150
maxlen = 100
batch_size = 100
def BLSTM(Ab, dataset, size):
    wait = input("PRESS")
    #dataset = loadtxt('Hotel-Bow.csv', delimiter=',')
    # split into input (X) and output (y) variables
    result_array = np.array([])
    for i in Ab:
        X = dataset[:,1:i+1]
        Y = dataset[:,0]
        print(X.shape)
        print(Y.shape)
        wait = input("PRESS") 
        max_features=i
        print(i);
        print(max_features)
        maxlen=max_features
        print(maxlen)
        print('Loading data...')
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
        print(len(x_train), 'train sequences')
        print(len(x_test), 'test sequences')
        print('Pad sequences (samples x time)')
        #x_train = np.expand_dims(x_train, axis=2)
        #x_test = np.expand_dims(x_test, axis=2)
        print('x_train shape:', x_train.shape)
        print('x_test shape:', x_test.shape)
        print('Build model...')
        model = Sequential()
        model.add(Embedding(max_features, 128, input_length=maxlen))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        
        # try using different optimizers and different optimizer configs
        model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
        
        print('Train...')
        
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=20,
                  validation_data=(x_test, y_test))
        
        _, accuracy = model.evaluate(x_test, y_test)
        #y_pred_val=model.predict(x_test)
        
        #roc_val = roc_auc_score(y_test, y_pred_val)
        #print(roc_val)
        print('Accuracy: %.2f' % (accuracy * 100))
        K.clear_session()
        gc.collect();
        result_array = np.append(result_array, accuracy);
        print(accuracy)
        print(i)
   # wait = input("PRESS ENTER TO CONTINUE.")
    #print((result_array))
    return result_array