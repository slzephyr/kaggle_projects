# pretrained models for dog breed classification
'''
Xception
VGG16
VGG19
ResNet50
InceptionV3
InceptionResNetV2
MobileNet
DenseNet
NASNet
MobileNetV2
'''

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.optimizers import SGD, Adam

import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from shutil import copy2
import os

# load and preprocessing data
path = '/home/lei/Downloads/kaggle_datasets/dog_breed/'
dir_train = '/home/lei/Downloads/kaggle_datasets/dog_breed/train/'
dir_test = '/home/lei/Downloads/kaggle_datasets/dog_breed/test/'
labels = pd.read_csv('/home/lei/Downloads/kaggle_datasets/dog_breed/labels.csv')
submission = pd.read_csv('/home/lei/Downloads/kaggle_datasets/dog_breed/sample_submission.csv')

#  too large of whole dataset, out of memory
targets_series = pd.Series(labels['breed'])
one_hot = pd.get_dummies(targets_series, sparse=True)
one_hot_labels = np.asarray(one_hot)

'''
# separate training data into train and validation sets
# each breed has its own subfolder, in order for flow_from_directory
valid_size = 0.3
TrainID = []  # id of train image
ValidID = []  # id of validation image
np.random.seed(42)
for breed in np.unique(labels['breed'].values):
    sub = labels[labels['breed'].str.match(breed)]
    N = np.arange(sub.shape[0])
    np.random.shuffle(N)
    numTrain = np.round((1-valid_size)*sub.shape[0]).astype('int16')
    TrainID  = sub['id'].values[N[:numTrain]].tolist()
    ValidID = sub['id'].values[N[numTrain:]].tolist()

    if not os.path.exists(path+'splitTrain/'+breed):
        os.makedirs(path+'splitTrain/'+breed)
    for id in TrainID:
        copy2(dir_train + id + '.jpg', path + 'splitTrain/'+breed)
    if not os.path.exists(path+'splitVali/'+breed):
        os.makedirs(path+'splitVali/'+breed)
    for id in ValidID:
        copy2(dir_train+id+'.jpg', path +'splitVali/'+breed)
'''

'''
# load all the images into memory, but it is too large
im_size = [224, 224]
x_train = []
y_train = []
x_test = []
# training data
i = 0
for f in tqdm(labels['id'].values):
    img = image.load_img((dir_train + '%s.jpg' % f), target_size=im_size)
    img = image.img_to_array(img)
    label = one_hot_labels[i]
    x_train.append(img)
    y_train.append(label)
    i += 1
# testing data
#for f in tqdm(submission['id'].values):
#    img = image.load_img((dir_test + '%s.jpg' % f), target_size=im_size)
#    img = image.img_to_array(img)
#    x_test.append(img)

#x_train = np.array(x_train, np.float16)/255.
#y_train = np.array(y_train)
#x_test = np.array(x_test, np.float16)/255.

# split training data
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.3, random_state=123)
'''

img_size = 50
# pretrained model
#base_model = ResNet50(weights='imagenet', include_top=False, input_shape=[img_size, img_size,3])
base_model = VGG16(weights='imagenet', include_top=False, input_shape=[img_size,img_size,3])
# train only the top layers (which were randomly initialized)
for layer in base_model.layers:
    layer.trainable = False
# add new layers on top
x = base_model.output
x = Flatten()(x)
#x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(120, activation='softmax')(x)
# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

learning_rate = 1e-3
adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
model.summary()

batch_size = 16
epochs = 5

# data augmentation
train_datagen = image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = image.ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers, and indefinitely generate batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        path+'splitTrain',  # this is the target directory
        target_size=(img_size, img_size),  # all images will be resized to img_size X img_size
        batch_size=batch_size,
        class_mode='categorical')

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        path+'splitVali',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical')
# generator for test data
test_generator = test_datagen.flow_from_directory(
        dir_test,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical')

model.fit_generator(
        train_generator,
        steps_per_epoch=7143 // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=3079 // batch_size,
        verbose=2)
model.save_weights('vgg16_1.h5')  # always save your weights after training or during training


preds = model.predict_generator(test_generator)


sub = pd.DataFrame(preds)
# Set column names to those generated by the one-hot encoding earlier
col_names = one_hot.columns.values
sub.columns = col_names
# Insert the column id from the sample_submission at the start of the data frame
sub.insert(0, 'id', submission['id'])
sub.head(5)
