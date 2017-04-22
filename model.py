import csv
import cv2
import sklearn
import numpy as np
import random
import math

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Activation
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Cropping2D
from keras.layers import Dropout
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils.visualize_util import plot

from keras.models import Model
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

def preprocess_image(img):

    preprocessed_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    return preprocessed_img

def augment_sample(img, steering):

    augmented_image = cv2.flip(img, 1)
    augmented_steering = steering * -1.0
    return augmented_image, augmented_steering

def load_sample(sample):

    images = []
    steerings = []

    filename_center = sample[0]
    filename_left = sample[1]
    filename_right = sample[2]

    image_center = cv2.imread(filename_center)
    image_left = cv2.imread(filename_left)
    image_right = cv2.imread(filename_right)

    image_center = preprocess_image(image_center)
    image_left = preprocess_image(image_left)
    image_right = preprocess_image(image_right)

    images.append(image_center)
    images.append(image_left)
    images.append(image_right)

    correction = 0.25
    steering_center = float(sample[3])
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    steerings.append(steering_center)
    steerings.append(steering_left)
    steerings.append(steering_right)

    return images, steerings

def generator(samples, batch_size=64, augment=False):

    num_samples = len(samples)

    while 1:

        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):

            batch_samples = samples[offset:offset + batch_size]

            batch_images = []
            batch_steerings = []

            for batch_sample in batch_samples:

                sample_images, sample_steerings = load_sample(batch_sample)
                batch_images.extend(sample_images)
                batch_steerings.extend(sample_steerings)

            if augment:

                augmented_images = []
                augmented_steerings = []

                for batch_image, batch_steering in zip(batch_images, batch_steerings):

                    if random.uniform(0, 1) > 0.66:
                        augmented_image, augmented_steering = augment_sample(batch_image, batch_steering)
                        augmented_images.append(augmented_image)
                        augmented_steerings.append(augmented_steering)
                    else:
                        augmented_images.append(batch_image)
                        augmented_steerings.append(batch_steering)

                batch_images = augmented_images
                batch_steerings = augmented_steerings

            X = np.array(batch_images)
            y = np.array(batch_steerings)
            yield sklearn.utils.shuffle(X, y)

## Specify list of CSV files to read with their corresponding image paths

csv_paths = ['data/',
        'recorded_data/t1_center1/',
        'recorded_data/t1_recovery1/',
        'recorded_data/t1_c_recovery1/',
        'recorded_data/t2_smoothturn1/',
        'recorded_data/t2_center1/',
        'recorded_data/t2_centerslow1/',
        'recorded_data/t2_center2/',
        'recorded_data/t2_center3/',
        'recorded_data/t2_c_center1/',
        'recorded_data/t2_recovery1/',
        'recorded_data/t2_recoveryend1/',
        'recorded_data/t2_recoveryturn1/',
        'recorded_data/t2_c_recovery1/']
img_paths = ['data/IMG/',
        'recorded_data/t1_center1/IMG/',
        'recorded_data/t1_recovery1/IMG/',
        'recorded_data/t1_c_recovery1/IMG/',
        'recorded_data/t2_smoothturn1/IMG/',
        'recorded_data/t2_center1/IMG/',
        'recorded_data/t2_centerslow1/IMG/',
        'recorded_data/t2_center2/IMG/',
        'recorded_data/t2_center3/IMG/',
        'recorded_data/t2_c_center1/IMG/',
        'recorded_data/t2_recovery1/IMG/',
        'recorded_data/t2_recoveryend1/IMG/',
        'recorded_data/t2_recoveryturn1/IMG/',
        'recorded_data/t2_c_recovery1/IMG/']

## Generate list of samples including image paths

samples = []
steerings = []
for csv_path, img_path in zip(csv_paths, img_paths):

    print('Sampling ' + csv_path)

    lines = []
    
    with open(csv_path + 'driving_log.csv') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            lines.append(line)
        lines = lines[1:]

    print(str(len(lines)) + ' samples...')

    for line in lines:
        sample = line
        sample[0] = img_path + sample[0].split('/')[-1]
        sample[1] = img_path + sample[1].split('/')[-1]
        sample[2] = img_path + sample[2].split('/')[-1]
        samples.append(sample)
        steerings.append(float(sample[3]))

## Generate training and testing sets with generators
steerings = np.array(steerings)
num_bins = 8
avg_samples_per_bin = len(steerings)/num_bins
hist, bins = np.histogram(steerings, num_bins)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.plot((np.min(steerings), np.max(steerings)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
plt.show()

keep_probs = []
target = avg_samples_per_bin * .5
for i in range(num_bins):
    if hist[i] < target:
        keep_probs.append(1.)
    else:
        keep_probs.append(1./(hist[i]/target))

remove_list = []
for i in range(len(steerings)):
    for j in range(num_bins):
        if steerings[i] > bins[j] and steerings[i] <= bins[j+1]:
            # delete from X and y with probability 1 - keep_probs[j]
            if np.random.rand() > keep_probs[j]:
                remove_list.append(i)

#samples = np.delete(samples, remove_list, axis=0)
steerings = np.delete(steerings, remove_list, axis=0)

hist, bins = np.histogram(steerings, num_bins)
plt.bar(center, hist, align='center', width=width)
plt.plot((np.min(steerings), np.max(steerings)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
plt.show()

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('Training with ' + str(len(train_samples)) + ' samples...')
print('Validating with ' + str(len(validation_samples)) + ' samples...')

train_generator = generator(train_samples, batch_size=64, augment=True)
validation_generator = generator(validation_samples, batch_size=64, augment=False)

## Model definition

model = Sequential()
model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50, 20), (0,0))))
model.add(Convolution2D(24, 5, 5, W_regularizer=l2(0.001), subsample=(2, 2)))
#model.add(Activation('relu'))
model.add(ELU())
model.add(Convolution2D(36, 5, 5, W_regularizer=l2(0.001), subsample=(2, 2)))
#model.add(Activation('relu'))
model.add(ELU())
model.add(Convolution2D(48, 5, 5, W_regularizer=l2(0.001), subsample=(2, 2)))
#model.add(Activation('relu'))
model.add(ELU())
model.add(Convolution2D(64, 3, 3, W_regularizer=l2(0.001)))
#model.add(Activation('relu'))
model.add(ELU())
model.add(Convolution2D(64, 3, 3, W_regularizer=l2(0.001)))
#model.add(Activation('relu'))
model.add(ELU())
model.add(Flatten())
model.add(Dense(100, W_regularizer=l2(0.001)))
#model.add(Dropout(0.5))
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dense(50, W_regularizer=l2(0.001)))
#model.add(Dropout(0.5))
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dense(10, W_regularizer=l2(0.001)))
#model.add(Dropout(0.5))
#model.add(Activation('relu'))
model.add(ELU())
model.add(Dense(1))

model.compile(loss='mse', optimizer=Adam(lr=1e-3))

model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
#history_object = model.fit(X_train, y_train, validation_split=0.05, shuffle=True, nb_epoch=32, callbacks=[early_stopping])

history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
        validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=128, callbacks=[early_stopping])

print(history_object.history.keys())
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')

exit()
