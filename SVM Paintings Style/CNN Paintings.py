# -*- coding: utf-8 -*-
"""
Created on Mon May 22 18:06:21 2017

@author: Artem
"""

import csv
import pickle
import random
import keras
import time
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np


def separate(database):
    y = []
    x = []
    y_train = []
    x_train = []
    part = len(database) //100 *10
    cnt=0
    for point in database:
        if len(point[0])==1:
            if cnt>part:
                y.append(point[0])
                x.append(point[1:])
            else:
                y_train.append(point[0])
                x_train.append(point[1:])
        else:
            pass
        cnt+=1 
            
            
    #print(y)
    return x, y, x_train, y_train

def CNN(X_test, Y_test, X_train, Y_train):
    batch_size = 32 
    nb_epoch = 10
    nb_classes = 11
# input image dimensions
    img_rows, img_cols = 32, 32
    # the CIFAR10 images are RGB
    img_channels = 3
    
    
    seed = 7
    np.random.seed(seed)
    
    
    model = Sequential()
    
    ### PLEASE PUT YOUR CODE HERE!
    ### Add Convolution, Activation and Pooling layers, compile your model, and fit it.
    
    #Add convolutional layer with with 32 filters. output tensor format is 32,3,3
    model.add(Convolution2D(32, 3, 3, border_mode='valid', activation='relu',
                                input_shape=(32,32,3)))
    
    #reducing dimensionality of the features
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    
    model.add(Convolution2D(128, 3, 3, border_mode='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))
    
    
    model.add(Dropout(0.2))
    #Drop random 20% of weights
    
    model.add(Flatten())
    #convert the tensor to 1D vector
    
    model.add(Dense(128))
    #another layer which is fully connnected, changes the dimensions of vector
    
    model.add(Dropout(0.2))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    





    ## Compoling the model
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
    '''
    X_train =  X_train.astype('float32')
    Y_train = Y_train.astype('float32')
    X_train /= 255
    Y_train /= 255
    '''
    model.compile(loss='categorical_crossentropy', optimizer= opt,shuffle=True, metrics=['accuracy'])
    c, r = Y_train.shape
    Y_train.reshape(c, r)
    # kfold = StratifiedKFold(n_splits = 10, shuffle=True)
    # results = cross_val_score(model, X_train, Y_train, cv = kfold)
    # print(results.mean)
    
    model_hist = model.fit(X_train, Y_train, 
                           batch_size=batch_size,
          verbose=1,
          nb_epoch=nb_epoch,
          validation_data=(X_test, Y_test))

    plot_model_history(model_hist)
    score = model.evaluate(X_test, Y_test,
                        verbose=0,
                      )
    print('Test loss:', score[0])
    print('Test accuracy:', score[1]*100, '%')
   
def main():
    '''
    database=[]
    with open('data.csv', newline='') as f:
        reader = csv.reader(f, delimiter = ',')
        for row in reader: 
            new =[]
            newitem = ''
            for entry in row:
                if entry != ',':
                    newitem += entry
                else:
                    new.append(newitem)
                    newitem=''
            if len(newitem) != 0:
                database.append(new)
    random.shuffle(database, random.random)
    data = open('database.txt', 'wb')
    print(database)
    pickle.dump(database, data)
    '''
    
    
    datao = open('database.txt', 'rb')
    
    f = pickle.load(datao)
    '''
    for line in range(len(f)):
        print(line)
        print('_____')
        print(f[line][0])
    '''
    print(len(f))
    x, y, x_train, y_train = separate(f)
    x = np.asarray(x)
    y = np.asarray(y)
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    CNN(x, y, x_train, y_train)
main()