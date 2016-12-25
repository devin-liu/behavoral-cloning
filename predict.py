import json
from keras.models import model_from_json
import numpy as np
from scipy.misc import imread


model_file = open('model.json', 'r')
model_file = json.load(model_file)
model = model_from_json(model_file)

def load_photo(image_location):
  image = imread(image_location).astype(np.float32)
  image = image - np.mean(image)
  return np.array(image)


original_shape=160*320*3
photo = load_photo('IMG/center_2016_12_11_08_07_49_945.jpg')
test_image = photo.reshape(-1, original_shape)


print(model.predict(test_image))
