from scipy.misc import imread, imresize, face
import numpy as np

def preprocess_image(image_array):
  # Cropping
  image_array = image_array[60:160, 0:320]
  # Resize
  image_array = imresize(image_array, (20, 64))
  return image_array


def normalize(image_array):
  # image_array -= np.mean(image_array, axis = 0)
  mean = image_array.mean()
  # X /= np.std(X, axis = 0)
  image_array = np.subtract(image_array, mean)
  return image_array

