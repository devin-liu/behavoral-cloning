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
driving_log = np.genfromtxt(driving_log_csv_file, delimiter=",", usecols=(0,3), converters={0: lambda s: s}, skip_header=1)


def replaceWithPhoto(np_row):
  image_location = np_row[0].decode('UTF-8')
  image = imread('/Users/devin/behavioral-cloning/data/'+image_location, flatten=1).astype(np.float32)
  image = preprocess_image(image)
  # image = normalize(image)
  imgplot = plt.imshow(image)
  plt.show()
  if random.random() < 0.5:
    return np.array([np.fliplr(image), np_row[1] * -1])
  else:
    return np.array([image, np_row[1]])


def flip(np_row):
  # print(np_row)
  # np_row[0] = np.fliplr(np_row[0])
  return np_row


photoReplacer = np.vectorize(replaceWithPhoto)
photoFlipper = np.vectorize(flip)

batch = photoReplacer(driving_log[0])
print(flip(batch))
# flipped_driving_log = photoFlipper(batch)


# batched_logs = np.split(driving_log[0], 14)
# batched_logs = np.split(driving_log[0], 1)
