# adapted from noobhound

from matplotlib import pyplot
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

import numpy as np # linear algebra
#np.random.seed(666)
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

#Load data
data_loc = "~/Downloads/Kaggle_data/iceberg_classifier/"
train = pd.read_json(data_loc+"train.json")
test = pd.read_json(data_loc+"test.json")
mean = np.mean([angle for angle in train['inc_angle'] if angle!='na'])
# replace na with the mean angle
train.inc_angle = train.inc_angle.replace('na', mean)
test.inc_angle = test.inc_angle.replace('na', mean)
#train.inc_angle = train.inc_angle.replace('na', 0)
#train.inc_angle = train.inc_angle.astype(float).fillna(0.0)

# Train data
# normalize the data
x_band1 = np.array([np.array(band).astype(np.float32) for band in train["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32) for band in train["band_2"]])

# subtract mean
x_band1 = x_band1 - np.mean(x_band1,1,keepdims=True)
x_band2 = x_band2 - np.mean(x_band2,1,keepdims=True)

# reshape
x_band1 = x_band1.reshape(-1,75,75)
x_band2 = x_band2.reshape(-1,75,75)

#x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
#x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
X_train = np.concatenate([x_band1[:, :, :, np.newaxis]
                          , x_band2[:, :, :, np.newaxis]
                         , ], axis=-1)  #((x_band1+x_band2)/2)[:, :, :, np.newaxis]


X_angle_train = np.array(train.inc_angle)
y_train = np.array(train["is_iceberg"])

# Test data
x_band1 = np.array([np.array(band).astype(np.float32) for band in test["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32) for band in test["band_2"]])

# subtract mean
x_band1 = x_band1 - np.mean(x_band1,1,keepdims=True)
x_band2 = x_band2 - np.mean(x_band2,1,keepdims=True)

# reshape
x_band1 = x_band1.reshape(-1,75,75)
x_band2 = x_band2.reshape(-1,75,75)

#x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
#x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
X_test = np.concatenate([x_band1[:, :, :, np.newaxis]
                          , x_band2[:, :, :, np.newaxis]
                         , ], axis=-1)  #((x_band1+x_band2)/2)[:, :, :, np.newaxis]
X_angle_test = np.array(test.inc_angle)


X_train, X_valid, X_angle_train, X_angle_valid, y_train, y_valid = train_test_split(X_train
                    , X_angle_train, y_train, random_state=123, train_size=0.8)

#def get_callbacks(filepath, patience=2):
#    es = EarlyStopping('val_loss', patience=patience, mode="min")
#    msave = ModelCheckpoint(filepath, save_best_only=True)
#    return [es, msave]

def get_model():
    bn_model = 0
    p_activation = "elu"
    input_1 = Input(shape=(75, 75, 2), name="X_1")
    input_2 = Input(shape=[1], name="angle")

    #img_1 = Conv2D(16, kernel_size=(3, 3), activation=p_activation)((BatchNormalization(momentum=bn_model))(input_1))
    img_1 = Conv2D(16, kernel_size=(3, 3), activation=p_activation)(input_1)
    img_1 = Conv2D(16, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = MaxPooling2D((2, 2))(img_1)
    #img_1 = Dropout(0.2)(img_1)
    img_1 = Conv2D(32, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = Conv2D(32, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = MaxPooling2D((2, 2))(img_1)
    #img_1 = Dropout(0.2)(img_1)
    img_1 = Conv2D(64, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = Conv2D(64, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = MaxPooling2D((2, 2))(img_1)
    img_1 = Dropout(0.5)(img_1)
    img_1 = Conv2D(128, kernel_size=(3, 3), activation=p_activation)(img_1)
    img_1 = MaxPooling2D((2, 2))(img_1)
    img_1 = Dropout(0.5)(img_1)
    img_1 = GlobalMaxPooling2D()(img_1)
    '''
    #img_2 = Conv2D(128, kernel_size=(3, 3), activation=p_activation)((BatchNormalization(momentum=bn_model))(input_1))
    img_2 = Conv2D(128, kernel_size=(3, 3), activation=p_activation)(input_1)
    img_2 = MaxPooling2D((2, 2))(img_2)
    img_2 = Dropout(0.2)(img_2)
    img_2 = GlobalMaxPooling2D()(img_2)
    '''
    #img_concat = (Concatenate()([img_1, img_2, BatchNormalization(momentum=bn_model)(input_2)]))
    #img_concat = (Concatenate()([img_1, img_2]))

    #dense_ayer = Dropout(0.5)(BatchNormalization(momentum=bn_model)(Dense(256, activation=p_activation)(img_concat)))
    #dense_ayer = Dropout(0.5)(BatchNormalization(momentum=bn_model)(Dense(64, activation=p_activation)(dense_ayer)))
    dense_ayer = Dropout(0.5)(Dense(256, activation=p_activation)(img_1))
    dense_ayer = Dropout(0.5)(Dense(256, activation=p_activation)(dense_ayer))

    output = Dense(1, activation="sigmoid")(dense_ayer)

    model = Model([input_1], output)
    optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


model = get_model()
model.summary()


#file_path = ".model_weights.hdf5"
#callbacks = get_callbacks(filepath=file_path, patience=5)

batch_size = 16
epochs=20


model = get_model()
model.fit([X_train], y_train, epochs=epochs
          , validation_data=([X_valid], y_valid)
         , batch_size=batch_size
         , verbose =2,
          #callbacks=callbacks
          )

#model.load_weights(filepath=file_path)

# plot out the loss and acc
import matplotlib.pyplot as plt
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

'''
print("Train evaluate:")
print(model.evaluate([X_train, X_angle_train], y_train, verbose=1, batch_size=200))
print("####################")
print("validation evaluate:")
print(model.evaluate([X_valid, X_angle_valid], y_valid, verbose=1, batch_size=200))
'''

prediction = model.predict([X_test], verbose=1, batch_size=200)

submission = pd.DataFrame({'id': test["id"], 'is_iceberg': prediction.reshape((prediction.shape[0]))})
submission.head(10)

submission.to_csv("./submission.csv", index=False)