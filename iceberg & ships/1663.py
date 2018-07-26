# create a new CNN to classify the stacked patches

import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
#from sklearn.model_selection import train_test_split
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras import backend as K

from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score

import matplotlib.pylab as plt
from keras import regularizers
import pandas as pd
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
# calibrate with incidence angle, divide by square of cos(theta)
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


batch_size = 16
epochs = 100

# first Conv
model = Sequential()
model.add(Conv2D(16, kernel_size=(3,3),activation='relu',input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# second Conv
model.add(Conv2D(64, kernel_size=(3,3),activation='relu'))
model.add(Conv2D(128, kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

'''
# third Conv
model.add(Conv2D(128, kernel_size=(3,3),activation='relu'))
model.add(Conv2D(128, kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
'''

# fully connected
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.summary()

#sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
adm = Adam(lr=1e-4/2, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
model.compile(optimizer=adm,loss='binary_crossentropy',metrics=['accuracy'])

# prepare data augmentation configuration
datagen = ImageDataGenerator(
 #   samplewise_center=True,
    shear_range=0.2,
   zoom_range=0.2,
#    rotation_range=360,
    horizontal_flip=True,
    vertical_flip=True
    )
test_datagen = ImageDataGenerator(
   # horizontal_flip=True,
   # vertical_flip=True
    )

train_generator = datagen.flow(
    X_train,y_train,
    batch_size=batch_size)
# test_generator = test_datagen.flow(X_test,y_test,batch_size=batch_size)
#vali_generator = test_datagen.flow(X_vali,y_vali,batch_size=batch_size)

# fine-tune the model
#early_stopping_monitor = EarlyStopping(patience=3)
model.fit_generator(
    train_generator,
    steps_per_epoch=X_train.shape[0]/batch_size,
    epochs=epochs,
    verbose=2,
    validation_data=(X_valid,y_valid),
    #validation_data=vali_generator,
    validation_steps=X_valid.shape[0]/batch_size,
    #callbacks=[early_stopping_monitor],

)


fig1 = plt.figure()
ax1 = fig1.add_subplot(121)
plt.hold(True)
ax1.set_title('Loss')
ax1.plot(model.model.history.history['loss'], 'r')
ax1.plot(model.model.history.history['val_loss'], 'b')
ax1.legend(['train', 'validation'])
ax2 = fig1.add_subplot(122)
ax2.set_title('accuracy')
ax2.plot(model.model.history.history['acc'], 'r')
ax2.plot(model.model.history.history['val_acc'], 'b')
ax2.legend(['train', 'validation'])
plt.show()
plt.hold(False)

prediction = model.predict(X_test,batch_size=batch_size,verbose=2)

submission = pd.DataFrame({'id': test["id"], 'is_iceberg': prediction.reshape((prediction.shape[0]))})
submission.to_csv("./submission_keras2.csv", index=False)

