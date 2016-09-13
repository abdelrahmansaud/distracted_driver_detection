# Distracted Driver Detection

Please refer to project's [report](Report.pdf) for details about the project

Software used

Theano,
Keras,
OpenBlas,
cv2,
Cuda,
cudnn,
h5py,
pandas,
numpy

## Instructions for running the model
Dataset download link
https://www.kaggle.com/c/state-farm-distracted-driver-detection/data

Extract train and test compressed files which sould extract them into train and test folders
rename train folder to train_src
copy test images from test folder to test_all

Download VGG16 weights from
https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing
move the weights to weights folder and rename the file to vgg16_weights.h5
