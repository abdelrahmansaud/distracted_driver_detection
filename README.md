# Distracted Driver Detection

Please refer to project's [report](Report.pdf) for details about the project

Software used

- Theano
- Keras
- OpenBlas
- cv2
- Cuda
- cudnn
- h5py
- pandas
- numpy

## Instructions for running the model

#### Create the below folder structure
- cache
- history
- results
- submissions
- test
  - test_all
- train
- val
- weights

#### Download dataset
Dataset download link
https://www.kaggle.com/c/state-farm-distracted-driver-detection/data

1. Extract train and test compressed files which sould extract them into train and test folders
2. Rename train folder to train_src
3. Copy test images from test folder to test_all

#### Download pre-trained model
1. Download VGG16 weights from
https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing
2. move the weights to weights folder and rename the file to vgg16_weights.h5

## Note
The current version will train the model on two stages

1. Use the pre-trained weights for all the layers except fully connected layers, this is train the layers with randomly initialized weights
2. Train the model again but this time including the last convolutional block
