import random
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

BATCH_SIZE = 32

# Extract samples from csv
def extract_csv(csvfile, path):
    lines = []
    with open(csvfile) as f:
        reader = csv.reader(f)
        for line in reader:
            lines.append(line)
    samples = []
    for i in range(1, len(lines)):
        a = float(lines[i][3])
        center_img = lines[i][0].strip().replace('\\', '/').split('/')[-1]
        left_img = lines[i][1].strip().replace('\\', '/').split('/')[-1]
        right_img = lines[i][2].strip().replace('\\', '/').split('/')[-1]
        samples.append([path + center_img, a, 0])
        samples.append([path + left_img, a + 0.2, 0])
        samples.append([path + right_img, a - 0.2, 0])
        # Augment - Flip
        samples.append([path + center_img, a * -1.0, 1])
        samples.append([path + left_img, (a + 0.2) * -1.0, 1])
        samples.append([path + right_img, (a - 0.2) * -1.0, 1])

    return samples

def extract_samples():
    s1 = extract_csv('../data/raw/driving_log.csv', '../data/raw/IMG/')
    s2 = extract_csv('../data/center/driving_log.csv', '../data/center/IMG/')
    s3 = extract_csv('../data/curve/driving_log.csv', '../data/curve/IMG/')
    return s1 + s2 + s3

samples = extract_samples()
train_samples, valid_samples = train_test_split(samples, test_size=0.2)
print('====> Train samples number:', len(train_samples))
print('====> Valid samples number:', len(valid_samples))

# Make samples number divided by BATCH_SIZE 
train_samples = random.sample(train_samples, len(train_samples) - len(train_samples) % BATCH_SIZE)
valid_samples = random.sample(valid_samples, len(valid_samples) - len(valid_samples) % BATCH_SIZE)
print('====> Train samples number(shorted):', len(train_samples))
print('====> Valid samples number(shorted):', len(valid_samples))

def generator(samples, batch_size=32):
    num_samples = len(samples)
    # Loop forever so the generator never terminates
    while 1:
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset + batch_size]
            
            images, angles = [], []
            for sample in batch_samples:
                img = cv2.imread(sample[0])
                a = sample[1]
                if sample[2] == 1:
                    img = cv2.flip(img, 1) 
                images.append(img)
                angles.append(a)
            
            yield shuffle(np.array(images), np.array(angles))

train_generator = generator(train_samples, batch_size=BATCH_SIZE)
valid_generator = generator(valid_samples, batch_size=BATCH_SIZE)

from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

# Nvidia Net
def NvidiaNet():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (2, 2))))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))

    model.add(Flatten())

    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    return model, 'nvidia.h5'

# Simple Net （Not used!）
def AlexNet():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Flatten())
    model.add(Dense(1))
    return model, 'alexnet.h5'

# LeNet
def LeNet():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (2, 2))))
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model, 'lenet.h5'

# Build the model
model, mfile = NvidiaNet()

# Train the model
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), \
    validation_data=valid_generator, nb_val_samples=len(valid_samples), nb_epoch=3)

model.save(mfile)
