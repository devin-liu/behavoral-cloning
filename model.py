import json
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Input, Activation, Conv2D, Flatten, Dropout, MaxPooling2D, Lambda
from keras.utils import np_utils


# Load the pickled image and steering angle data
training_file = 'data/driving_log.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)

# The pickled training file is stored in format [feature, label]
features = np.array([x[0] for x in train])
labels = np.array([x[1]  for x in train])

# Split the pickled data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.1, random_state=0)


model = Sequential() # Initialize the Keras Set
model.add(Lambda(lambda x: x/127.5 - 1., # Take the input shape (20,64,3) image and normalize with lambda layer
          input_shape=(20, 64, 3),
          output_shape=(20, 64, 3)))
model.add(Conv2D(8, 2, 2, subsample=(2, 2), border_mode="same", activation='relu')) # 8 Filter convolution
model.add(Conv2D(2, 2, 2, subsample=(2, 2), border_mode="same", activation='relu')) # 2 Filter convolution
model.add(Flatten()) # Prep for dense layer
model.add(Dense(640)) # Wide dense layer for more granular steering angles
model.add(Dropout(0.5)) # Dropout to prevent overfitting
model.add(Dense(1)) # Final dense layer for steering angle prediction
model.summary() # Prints summary of model

# Compile and run the model
model.compile(loss='mean_squared_error',
             optimizer='Adam')
history = model.fit(X_train, y_train,
                   batch_size=640, nb_epoch=16,
                   verbose=1, validation_data=(X_val, y_val))

# Prep and save the model weights
model.save_weights('model.h5')

with open('model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)

