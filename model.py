import csv
import cv2
import sklearn
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Activation
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Cropping2D
from keras.layers import Dropout
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2, activity_l2
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

from keras.models import Model
import matplotlib.pyplot as plt

def preprocess_image(img):

    preprocessed_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    return preprocessed_img

def load_sample(sample, img_path):

    images = []
    steerings = []

    source_path_center = sample[0]
    filename_center = source_path_center.split('/')[-1]
    source_path_left = sample[1]
    filename_left = source_path_left.split('/')[-1]
    source_path_right = sample[2]
    filename_right = source_path_right.split('/')[-1]

    image_center = cv2.imread(img_path + filename_center)
    image_left = cv2.imread(img_path + filename_left)
    image_right = cv2.imread(img_path + filename_right)

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

def generator(samples, batch_size=64):

    num_samples = len(samples)

    while 1:

        shuffle(samples)
        for offset in range(0, num_samples, batch_size):

            batch_samples = samples[offset:offset + batch_size]

            batch_images = []
            batch_steerings = []

            for batch_sample in batch_samples:

                sample_images, sample_steerings = load_sample(batch_sample[0], batch_sample[1])
                batch_images.extend(sample_images)
                batch_steerings.extend(sample_steerings)

            X = np.array(batch_images)
            y = np.array(batch_steerings)
            yield sklearn.utils.shuffle(X, y)

## Specify list of CSV files to read with their corresponding image paths
csv_paths = ['data/',
        'recorded_data/recovery1/',
        'recorded_data/c_recovery1/',
        'recorded_data/t2_rightlane1/']
img_paths = ['data/IMG/',
        'recorded_data/recovery1/IMG/',
        'recorded_data/c_recovery1/IMG/',
        'recorded_data/t2_rightlane1/IMG/']

## Generate list of samples as [line, image path]
samples = []
for csv_path, img_path in zip(csv_paths, img_paths):

    print('Sampling ' + csv_path)

    lines = []
    
    with open(csv_path + 'driving_log.csv') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            lines.append(line)
        lines = lines[1:]

    for line in lines:
        samples.append([line, img_path])

images = []
steerings = []

for csv_path, img_path in zip(csv_paths, img_paths):

    print('Processing CSV: ' + csv_path)
    lines = []

    with open(csv_path + 'driving_log.csv') as csvfile:

        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
        lines = lines[1:]

    for line in lines:

        sample_images, sample_steerings = load_sample(line, img_path)
        images.extend(sample_images)
        steerings.extend(sample_steerings)


## Data augmentation

augmented_images = []
augmented_steerings = []

for image, steering in zip(images, steerings):
    augmented_images.append(image)
    augmented_steerings.append(steering)
    augmented_images.append(cv2.flip(image, 1))
    augmented_steerings.append(steering * -1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_steerings)

## Model definition

model = Sequential()
model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer=Adam(lr=1e-4))

early_stopping = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=32, callbacks=[early_stopping])

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
