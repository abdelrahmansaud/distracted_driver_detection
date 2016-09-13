import numpy as np
import os
import glob
import cv2
import pickle
import datetime
import pandas as pd
import time
import shutil
import warnings
warnings.filterwarnings("ignore")
np.random.seed(2016)
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from keras.models import Sequential

from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.utils import np_utils
from keras.regularizers import l2, l1l2
from keras.layers.advanced_activations import ELU
from keras.layers.core import Dropout

from models import get_simple_model, get_VGG16_classifier, get_VGG16_finetune
from utils import *
from cfg import *

import h5py

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


def train_model(_model, path):
    train_datagen = ImageDataGenerator(rescale=1./255)

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        'train_src',
        target_size=(rows, cols),
        batch_size=batch_size,
        class_mode='categorical')

    val_generator = val_datagen.flow_from_directory(
        'val',
        target_size=(rows, cols),
        shuffle=False,
        batch_size=batch_size,
        class_mode='categorical')

    
    
    # Count of train/val images to be used by generator
    count_of_train = get_nb_files('train_src')
    count_of_val = get_nb_files('val')
        
    history = None
    model = _model
    
    # Continue training
    #if os.path.isfile(path):
    #    model.load_weights(path)
        
    try:
        if not os.path.isfile(path):
            callbacks = [
                # Wait for number of epochs before stopping training
                EarlyStopping(monitor='val_loss', patience=3, verbose=1),
                ModelCheckpoint(path, monitor='val_loss', save_best_only=True, verbose=0)
            ]

            history = model.fit_generator(
                train_generator,
                samples_per_epoch=int(count_of_train * 1),
                nb_epoch=nb_epoch,
                validation_data=val_generator,
                callbacks=callbacks,
                nb_val_samples=count_of_val)

            df = pd.DataFrame({'loss': history.history['loss'], 
                               'val_loss': history.history['val_loss']})
            df.to_csv('history/vgg16.csv')


        # Load best model
        if os.path.isfile(path):
            model.load_weights(path)

    # Keyboard interruptions can be used to pass folds
    except KeyboardInterrupt:
        print('Passing fold.')
        
    return model, history