# classify the stacked band patches with pre-trained CNN built in Keras

#import sys
#sys.path
#sys.path.append('/home/mcisaac/Dropbox/Professional Learning/7.DNN/deep-learning-models-keras')
#sys.path.insert(0,'/home/mcisaac/Dropbox/Professional Learning/7.DNN/deep-learning-models-keras')


# load modules

#from keras.applications.vgg16 import VGG16
#from keras.applications.vgg19 import VGG19
#from keras.applications.resnet50 import ResNet50
#from keras.applications.inception_v3 import InceptionV3
#from keras.applications.xception import Xception

#from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras import regularizers
from resnet50 import *
from keras.preprocessing import image
from keras.models import Sequential,Model
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import SGD,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.utils import np_utils

import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
import pandas as pd

import matplotlib.pylab as plt
import math


# Load data
data_loc = "~/Downloads/Kaggle_data/iceberg_classifier/"
train = pd.read_json(data_loc + "train.json")
test = pd.read_json(data_loc + "test.json")
mean = np.mean([angle for angle in train['inc_angle'] if angle != 'na'])
# replace na with the mean angle
train.inc_angle = train.inc_angle.replace('na', mean)
test.inc_angle = test.inc_angle.replace('na', mean)
# train.inc_angle = train.inc_angle.replace('na', 0)
# train.inc_angle = train.inc_angle.astype(float).fillna(0.0)

# Train data
# normalize the data
x_band1 = np.array([np.array(band).astype(np.float32)/math.cos(math.radians(angle)) for band, angle in zip(train["band_1"], train["inc_angle"])])
x_band2 = np.array([np.array(band).astype(np.float32)/math.cos(math.radians(angle)) for band, angle in zip(train["band_2"], train["inc_angle"])])

# subtract mean
x_band1 = x_band1 - np.mean(x_band1, 1, keepdims=True)
x_band2 = x_band2 - np.mean(x_band2, 1, keepdims=True)

# reshape
x_band1 = x_band1.reshape(-1, 75, 75)
x_band2 = x_band2.reshape(-1, 75, 75)

# x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
# x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
X_train = np.concatenate([x_band1[:, :, :, np.newaxis]
                             , x_band2[:, :, :, np.newaxis]
                             ], axis=-1)  # ((x_band1+x_band2)/2)[:, :, :, np.newaxis]

X_angle_train = np.array(train.inc_angle)
y_train = np.array(train["is_iceberg"])

# Test data
x_band1 = np.array([np.array(band).astype(np.float32)/math.cos(math.radians(angle)) for band, angle in zip(test["band_1"], test["inc_angle"])])
x_band2 = np.array([np.array(band).astype(np.float32)/math.cos(math.radians(angle)) for band, angle in zip(test["band_2"], test["inc_angle"])])

# subtract mean
x_band1 = x_band1 - np.mean(x_band1, 1, keepdims=True)
x_band2 = x_band2 - np.mean(x_band2, 1, keepdims=True)

# reshape
x_band1 = x_band1.reshape(-1, 75, 75)
x_band2 = x_band2.reshape(-1, 75, 75)

# x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
# x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
X_test = np.concatenate([x_band1[:, :, :, np.newaxis]
                            , x_band2[:, :, :, np.newaxis]
                            ], axis=-1)  # ((x_band1+x_band2)/2)[:, :, :, np.newaxis]
X_angle_test = np.array(test.inc_angle)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=123, train_size=0.8)

# ============ load pre-trianed cnn model ======================================
input_shape = (75,75,2)

#base_model = VGG16(weights='imagenet',include_top=False,input_shape=input_shape)
base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape)

top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(1,activation='sigmoid'))

# add the model on top of the convolutional base
model = Model(input=base_model.input, output=top_model(base_model.output))

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
'''
for layer in model.layers[:-4]:
    layer.trainable = False
'''
model.summary()
# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
#sgd = SGD(lr=1e-4,decay=1e-6, momentum=0.9,nesterov=True)
adam = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
model.compile(loss='binary_crossentropy',
              optimizer= adam,
              metrics=['accuracy'])
#keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# prepare data augmentation configuration
datagen = ImageDataGenerator(
#    samplewise_center=True,
    shear_range=0.2,
    zoom_range=0.2,
#    rotation_range=360,
    horizontal_flip=True,
    vertical_flip=True
    )

test_datagen = ImageDataGenerator(#samplewise_center=True
                                    )

batch_size = 16
epoch = 100
train_generator = datagen.flow(
    X_train,y_train,
    batch_size=batch_size)

# validation_generator = test_datagen.flow(
#    X_vali,y_vali,
 #   batch_size=batch_size)

# fine-tune the model
model.fit_generator(
    train_generator,
    steps_per_epoch=X_train.shape[0]/batch_size,epochs=epoch,
    #validation_data=validation_generator,
    validation_data=(X_valid,y_valid),
    verbose=2,
    validation_steps=X_valid.shape[0]/batch_size,
    #callbacks=[early_stopping_monitor],
    #workers = 12,
#    use_multiprocessing=True   # it is already in parallel computing
)

fig1 = plt.figure()
ax1 = fig1.add_subplot(121)
plt.hold(True)
ax1.set_title('Loss')
ax1.plot(model.history.history['loss'], 'r')
ax1.plot(model.history.history['val_loss'], 'b')
ax1.legend(['train', 'validation'])
ax2 = fig1.add_subplot(122)
ax2.set_title('accuracy')
ax2.plot(model.history.history['acc'], 'r')
ax2.plot(model.history.history['val_acc'], 'b')
ax2.legend(['train', 'validation'])
plt.show()
plt.hold(False)


# loss value & metrics values in score
#score = model.evaluate(X_test, y_test, batch_size=batch_size)
prediction = model.predict(X_test, batch_size=batch_size, verbose=2)

submission = pd.DataFrame({'id': test["id"], 'is_iceberg': prediction.reshape((prediction.shape[0]))})
submission.to_csv("./submission.csv", index=False)

a=0

