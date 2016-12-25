import json
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Input, Activation, Conv2D, Flatten, Dropout, MaxPooling2D, Lambda
from keras.utils import np_utils



training_file = 'data/driving_log.p'
testing_file = 'data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)


features = np.array([x[0] for x in train])
labels = np.array([x[1]  for x in train])


X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.1, random_state=0)

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.,
          input_shape=(20, 64, 3),
          output_shape=(20, 64, 3)))
model.add(Conv2D(8, 2, 2, subsample=(2, 2), border_mode="same", activation='relu'))
model.add(Conv2D(2, 2, 2, subsample=(2, 2), border_mode="same", activation='relu'))
model.add(Flatten())
model.add(Dense(640))
model.add(Dense(1))
model.summary()
model.compile(loss='mean_squared_error',
             optimizer='Adam')
history = model.fit(X_train, y_train,
                   batch_size=640, nb_epoch=12,
                   verbose=1, validation_data=(X_val, y_val))

model.save_weights('model.h5')

with open('model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)

