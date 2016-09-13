import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense

from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten
)
from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D,
    AveragePooling2D,
    #GlobalPooling2D,
    ZeroPadding2D
)

def get_simple_model(rows, cols, channels=3):
    inputs = Input((channels, rows, cols))
    
    conv1 = Convolution2D(32, 3, 3, border_mode='same', init='he_normal', activation='relu')(inputs)
    max1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    avg1 = AveragePooling2D(pool_size=(2, 2))(conv1)
    sum1 = merge([(max1), avg1], mode='sum', concat_axis=1)
    drop1 = Dropout(0.3)(sum1)
    
    conv2 = Convolution2D(64, 3, 3, border_mode='same', init='he_normal', activation='relu')(drop1)
    max2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    avg2 = AveragePooling2D(pool_size=(2, 2))(conv2)
    sum2 = merge([(max2), avg2], mode='sum', concat_axis=1)
    drop2 = Dropout(0.3)(sum2)
    
    conv3 = Convolution2D(128, 3, 3, border_mode='same', init='he_normal', activation='relu')(drop2)
    max3 = MaxPooling2D(pool_size=(8, 8))(conv3)
    avg3 = AveragePooling2D(pool_size=(8, 8))(conv3)
    sum3 = merge([(max3), avg3], mode='sum', concat_axis=1)
    drop3 = Dropout(0.3)(sum3)
    

    flat = Flatten()(drop3)
    dense1 = Dense(10)(flat)
    out = Activation('softmax')(dense1)
    
    model = Model(input=inputs, output=out)

    model.compile(Adam(lr=5e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def get_VGG16_classifier(rows, cols, channels=3):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(channels, rows, cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    model.load_weights('weights/vgg16_weights.h5')

    print('VGG 16 model loaded.')

    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=1e-4, momentum=0.9)

    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def get_VGG16_finetune(rows, cols, channels=3):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(channels, rows, cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', trainable=False))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.load_weights('weights/tuned_classifier_weights.h5')

    print('Tuned classifier model loaded.')


    sgd = SGD(lr=1e-4, momentum=0.9)

    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model