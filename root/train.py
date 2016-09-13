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
from training import *
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
from keras.layers.normalization import BatchNormalization




def train_and_predict(nfolds=10):

    # Cross validation by driver
    #kf = KFold(len(driver_list), n_folds=nfolds, shuffle=True, random_state=random_state)
    num_fold = 0
    
    
    
    # Predictions of all folds
    all_predictions = []
    all_train_predictions = []
    
    # Make sure images are merged before start
    merge_back()
    
    #driver_list_valid = ['p042', 'p051']
    driver_list_valid = ['p022', 'p021']
    train_val_split(driver_list_valid)
    
    model, history = train_model(get_VGG16_classifier(rows, cols, 3),
                                    'cache/tuning_vgg16.h5')
    model.save_weights('weights/tuned_classifier_weights.h5')    
    
    #for train_drivers, test_drivers in kf:

    # Split data by driver for cross validation
    #driver_list_valid = [driver_list[i] for i in test_drivers]
    train_val_split(driver_list_valid)

    # Count of test images to be used by generator
    count_of_test = get_nb_files('test')

    num_fold += 1
    #print('Starting fold {} / {}'.format(num_fold, nfolds))

    # Path to save best model of each fold
    model_path = os.path.join('cache', 'vgg16_model.h5')

    model, history = train_model(get_VGG16_finetune(rows, cols, 3),
                                 model_path)

    # Merge images back into one folder
    merge_back()
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        'test',
        target_size=(rows, cols),
        shuffle=False,
        batch_size=batch_size,
        # Return data only, no labels
        class_mode=None)


    # Predict on testing set
    print('Predicting on test set')
    predictions = model.predict_generator(test_generator, count_of_test)
    #all_predictions.append(predictions)
    
    X_train, y_train, X_train_id = load_all_train(rows, cols)
    print('Predicting on train set')
    train_pred = model.predict(X_train, batch_size=batch_size, verbose=0)
    #all_train_predictions.append(train_pred)


    # Create ensemble of folds predictions
    #train_folds = merge_several_folds_mean(all_train_predictions, nfolds)
    save_train_preds(train_pred, y_train, X_train_id)
    
    #folds_ensemble = merge_several_folds_mean(all_predictions, nfolds)
    create_submission(predictions, get_test_ids(), nfolds)

if __name__ == '__main__':
    train_and_predict(13)
