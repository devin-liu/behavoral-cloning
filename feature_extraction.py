import pickle
import random
import numpy as np
from sklearn.model_selection import train_test_split
from preprocess import preprocess_image, normalize
from scipy.misc import imread, imresize
from keras import preprocessing


training_file = 'data/driving_log.p'
testing_file = 'data/test.p'


driving_log_csv_file = 'data/driving_log.csv'
driving_log = np.genfromtxt(driving_log_csv_file, delimiter=",", usecols=(0,3), converters={0: lambda s: s}, skip_header=1)


def replaceWithPhoto(np_row):
  image_location = np_row[0].decode('UTF-8')
  image = imread('/Users/devin/behavioral-cloning/data/'+image_location).astype(np.float32)
  image = preprocess_image(image)
  image = normalize(image)
  return np.array([image, np_row[1]])

def flip(np_row):
  image = np_row[0]
  return np.array([np.fliplr(image), np_row[1] * - 1])


photoReplacer = np.vectorize(replaceWithPhoto)
driving_log = photoReplacer(driving_log)

photoFlipper = np.vectorize(flip)
flipped_driving_log = photoFlipper(driving_log)

driving_log = np.hstack((driving_log, flipped_driving_log))


driving_log = np.array(driving_log).dump(open( training_file, "w+b" ))
