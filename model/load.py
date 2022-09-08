import numpy as np
import keras.models
# from scipy.misc import imread, imresize,imshow
import tensorflow as tf

from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

def init():

    input_shape = (28, 28, 1)
    model = Sequential()
    model.add(Conv2D(30, kernel_size=(3,3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) 
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation=tf.nn.softmax))
    
    #load woeights into new model
    model.load_weights("weights.h5")
    print("Loaded Model from disk")

    #compile and evaluate loaded model
    model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
    #loss,accuracy = model.evaluate(X_test,y_test)
    #print('loss:', loss)
    #print('accuracy:', accuracy)
    graph = tf.compat.v1.get_default_graph()


    return model, graph