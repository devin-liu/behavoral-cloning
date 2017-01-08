import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from preprocess import preprocess_image, normalize
from scipy.misc import imread, imresize

training_file = 'data/driving_log.p'
testing_file = 'data/test.p'


driving_log_csv_file = 'data/driving_log.csv'
center_images = np.genfromtxt(driving_log_csv_file, delimiter=",", usecols=(0,3), converters={0: lambda s: s}, skip_header=1)
left_images = np.genfromtxt(driving_log_csv_file, delimiter=",", usecols=(1,3), converters={1: lambda s: s}, skip_header=1)
right_images = np.genfromtxt(driving_log_csv_file, delimiter=",", usecols=(2,3), converters={2: lambda s: s}, skip_header=1)


def replaceWithPhoto(np_row):
  image_location = np_row[0].decode('UTF-8').strip()
  image = imread(image_location).astype(np.float32)
  image = preprocess_image(image)
  image = normalize(image)
  imgplot = plt.imshow(image)
  plt.show()
  return np.array([image, np_row[1]])

def flip(np_row):
  image = np_row[0]
  return np.array([np.fliplr(image), np_row[1] * - 1])

def getImages(images):
  for image in images:
    yield image


# open(training_file, 'wb').close()

for image in getImages(left_images[437:439]):
  image = replaceWithPhoto(image)
  flipped_image = flip(image)
  # image.dump(open( training_file, "ab" ))
  # flipped_image.dump(open( training_file, "ab" ))

for image in getImages(center_images[437:439]):
  image = replaceWithPhoto(image)
  flipped_image = flip(image)
  # image.dump(open( training_file, "ab" ))
  # flipped_image.dump(open( training_file, "ab" ))

for image in getImages(right_images[437:439]):
  image = replaceWithPhoto(image)
  flipped_image = flip(image)
  # image.dump(open( training_file, "ab" ))
  # flipped_image.dump(open( training_file, "ab" ))
