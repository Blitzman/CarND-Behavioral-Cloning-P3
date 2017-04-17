import csv
import cv2
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

csv_path = 'data/'
img_path = 'data/IMG/'

lines = []
with open(csv_path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
    lines = lines[1:]

images = []
steerings = []
for line in lines:

    source_path_center = line[0]
    filename_center = source_path_center.split('/')[-1]
    source_path_left = line[1]
    filename_left = source_path_left.split('/')[-1]
    source_path_right = line[2]
    filename_right = source_path_right.split('/')[-1]
    image_center = cv2.imread(img_path + filename_center)
    image_left = cv2.imread(img_path + filename_left)
    image_right = cv2.imread(img_path + filename_right)
    images.append(image_center)
    images.append(image_left)
    images.append(image_right)

    correction = 0.25
    steering_center = float(line[3])
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    steerings.append(steering_center)
    steerings.append(steering_left)
    steerings.append(steering_right)

augmented_images = []
augmented_steerings = []
for image, steering in zip(images, steerings):
    augmented_images.append(image)
    augmented_steerings.append(steering)
    augmented_images.append(cv2.flip(image, 1))
    augmented_steerings.append(steering * -1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_steerings)

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
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
