import csv
import cv2
import numpy as np
import keras.models import Sequential
from keras.layers import Flatten, Dropout, Dense, Lambda, Conv2D, BatchNormalization, Activation

lines = []
with open['../data/driving_log.csv'] as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = '../data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)


model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24,5,5,subsample=(2,2),activation="elu"))
model.add(Conv2D(36,5,5,subsample=(2,2),activation="elu"))
model.add(Conv2D(48,5,5,subsample=(2,2),activation="elu"))
model.add(Conv2D(64,3,3,activation="elu"))
model.add(Conv2D(64,3,3,activation="elu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100,activation="elu"))
model.add(Dropout(0.5))
model.add(Dense(50,activation="elu"))
model.add(Dropout(0.5))
model.add(Dense(10,activation="elu"))
model.add(Dense(1))
print(model.summary())