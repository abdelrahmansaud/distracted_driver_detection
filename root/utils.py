import numpy as np
import os
import glob
import cv2
import pickle
import datetime
import pandas as pd
import time
import shutil

from keras.utils import np_utils


# Create submission file
def create_submission(predictions, test_id, folds):
    # Create dataframe of predictions with specified column headers
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir('submissions'):
        os.mkdir('submissions')
    # Use number of folds and time as unique identifier of submissions
    suffix = 'folds_' + str(folds) + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('submissions', 'submission_' + suffix + '.csv')
    result1.to_csv(sub_file, index=False)

# Create ensemble of folds
def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()


def load_all_train(rows, cols):
    """ Load all training data 
        return data, labels, file names"""
    X_train = []
    X_train_id = []
    y_train = []

    print('Read all train images')
    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join('train_src', 'c' + str(j), '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            raw = cv2.imread(fl)
            img = cv2.resize(raw, (cols, rows), interpolation=cv2.INTER_LINEAR)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(j)
            
            
    X_train = np.array(X_train, dtype=np.uint8)
    y_train = np.array(y_train, dtype=np.uint8)
        
    X_train = X_train.transpose((0, 3, 1, 2))

    y_train = np_utils.to_categorical(y_train, 10)
    X_train = X_train.astype('float32')
    X_train /= 255

    return X_train, y_train, X_train_id

def save_train_preds(train_folds, y_train, X_train_id):
    X = [np.argmax(i) for i in train_folds]
    Y = [np.argmax(i) for i in y_train]
    df = pd.DataFrame({'X': X, 'Y': Y, 'img': X_train_id})
    df.to_csv('results/result.csv')

def train_val_split(selected):
    """ Split the data into train val on HDD
        selected driver IDs passed as argument
        this is used due to memory limitation
        to do the splitting in memory"""
    
    dr = dict()
    path = os.path.join('driver_imgs_list.csv')
    # selected = ['p072', 'p042', 'p014', 'p039']
    print('Read drivers data')
    print("Validation drivers: ", selected)
    dv = pd.read_csv(path)
    
    for j in range(10):
        print('Working on folder c{}'.format(j))
        all_path = os.path.join('train_src', 'c' + str(j), '*.jpg')
        files = glob.glob(all_path)
        tmp = dv['img'][(dv['subject'].isin(selected)) & (dv['classname'] == 'c' + str(j))].tolist()
        for fl in files:
            flbase = os.path.basename(fl)
            if flbase in tmp:
                new_path = os.path.join('val', 'c' + str(j), flbase)
                shutil.move(fl, new_path)

def merge_back():
    """ Merge the splitted drivers data back into one folder. """
    
    for j in range(10):
        print('Working on folder c{}'.format(j))
        val_path = os.path.join('val', 'c' + str(j), '*.jpg')
        files = glob.glob(val_path)
        for fl in files:
            flbase = os.path.basename(fl)
            new_path = os.path.join('train_src', 'c' + str(j), flbase)
            shutil.move(fl, new_path)
    print "Done Merging Back!"  

def get_nb_files(folder):
    """ Get number of files found in folder specified """
    count = sum([len(files) for r, d, files in os.walk("/home/aero/Kaggle/State_farm_drivers/" + folder)])
    return count

def get_test_ids():
    """ Get test IDs to be used for creating submission file"""
    files = os.listdir("/home/aero/Kaggle/State_farm_drivers/test/test_all")
    return files