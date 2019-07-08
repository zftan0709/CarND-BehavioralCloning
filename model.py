import csv
import cv2
import math
import numpy as np
from keras.models import Sequential
from keras.layers import Cropping2D, Flatten, Dropout, Dense, Lambda, Conv2D, BatchNormalization, Activation
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


images_path = []
steering_angle = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    correction = 0.2
    first_row = next(reader)
    for line in reader:
        # Center Camera Image and Angle
        images_path.append('./data/IMG/'+line[0].split('/')[-1])
        steering_angle.append(float(line[3]))
        # Left Camera Image and Angle
        images_path.append('./data/IMG/'+line[1].split('/')[-1])
        steering_angle.append(float(line[3])+correction)
        # Right Camera Image and Angle
        images_path.append('./data/IMG/'+line[2].split('/')[-1])
        steering_angle.append(float(line[3])-correction)

X_train_path, X_valid_path, y_train_angle, y_valid_angle = train_test_split(images_path,steering_angle,test_size=0.2)

def generator(X, y, batch_size=128):
    num_samples = len(X)
    while 1: # Loop forever so the generator never terminates
        X,y = shuffle(X,y)
        for offset in range(0, num_samples, int(batch_size/2)):
            batch_X, batch_y = X[offset:offset+int(batch_size/2)],y[offset:offset+int(batch_size/2)]
            images = []
            angles = []
            for i in range(len(batch_X)):
                img = cv2.imread(batch_X[i])
                images.append(img)
                angle = batch_y[i]
                angles.append(angle)
                image_flipped = np.fliplr(img)
                images.append(image_flipped)
                angles.append(-angle)
            X_train = np.array(images)
            #X_train = np.reshape(X_train,[None,32,32,3])
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)
            
batch_size = 128
train_generator = generator(X_train_path,y_train_angle, batch_size=batch_size)
validation_generator = generator(X_valid_path,y_valid_angle, batch_size=batch_size)

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20),(0,0))))
model.add(Conv2D(filters=24,kernel_size=(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(filters=36,kernel_size=(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(filters=48,kernel_size=(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
#print(model.summary())
model.compile(loss='mse',optimizer='adam')
history = model.fit_generator(train_generator, steps_per_epoch=np.ceil(2*len(X_train_path)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=np.ceil(2*len(X_valid_path)/batch_size), 
            epochs=5, verbose=1)
model.save('./model.h5')