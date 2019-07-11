# **Udacity Behavioral Cloning Project** 
---

[//]: # (Image References)

[image1]: ./README_image/simulator.jpg "Simulator"
[image2]: ./README_image/cnn-architecture.png "NVIDIA CNN Architecture"
[image3]: ./README_image/original.jpg "Original Image"
[image4]: ./README_image/flipped.jpg "Flipped Image"
[image5]: ./README_image/left.jpg "Left Image"
[image6]: ./README_image/center.jpg "Center Image"
[image7]: ./README_image/right.jpg "Right Image"
[image8]: ./README_image/run1.jpg "Final Video"
[video1]: ./run1.avi "Final Video"

## Introduction

_**Note:** This project makes use of  [Udacity Driving Simulator](https://github.com/udacity/self-driving-car-sim) and Udacity Workspace. The simulator is not included in this repository and can be found in the link above._

The main objectives of this project are:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road

In this project, deep learning techniques are applied to teach a car to drive autonomously in a simulated driving application. 
The provided [Udacity Driving Simulator](https://github.com/udacity/self-driving-car-sim) has two tracks included and is able to switch between training mode and autonomous mode.
In the training mode, users are able to collect driving data in the form of simulated car dashboard camera images, steering angle, throttle, brake, and speed.
In each sampling point, three images are collected from the dashboard cameras which are situated at the left, center, and right side of the car.

These collected data are then fed into neural networks to output the correct steering angle of the car. 
Once the training is done, the neural network model is saved as `model.h5`. 
The command `python drive.py model.h5` is then ran to test the model in autonomous mode.

![alt text][image1]

_Udacity Driving Simulator Interface_

## Network Architecture

The neural network architecture used for this project is a known self-driving car model from [NVIDIA](https://devblogs.nvidia.com/deep-learning-self-driving-cars/).
The diagram below shows the NVIDIA CNN architecture.

![alt text][image2]

The network consists of 9 layers in total which are normalization layer, 5 convolutional layer, and 3 fully connected layers.
Since the input picture for this project is 160x320x3 instead of 66x200x3, few preprocessing layers are added at the beginning of the network.

The first layer `lambda_1` normalize the input data to value between -0.5 and 0.5. By doing so, this enables the model to have a better distributed feature 
value range and learn at a faster rate.
In order to steer the car in the right direction, the model only need useful information such as the lane line. Therefore, the upper and lower part
of the input image which consists of trees, hills, sky, and car hood should be removed. The `cropping2d_1` layer crops the images to the desired shape.
Additional dropout layer is added after the final convolutional layer to prevent the model from overfitting.
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 90, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 43, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 20, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 6, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 4, 33, 64)         36928     
_________________________________________________________________
dropout_1 (Dropout)          (None, 4, 33, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 8448)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               844900    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 981,819
Trainable params: 981,819
Non-trainable params: 0
_________________________________________________________________
```
## Data Collection and Data Processing

Udacity has provided sets of sample data which are enough to produce a working model. However, students are encouraged to collect data
using the provided simulator. There are two control methods to steer the car in the simulator which are using keyboard (W,A,S,D)
or using mouse. Using mouse to steer the car could produce better data than using the keyboard. However, it is extremely difficult to control
the car using mouse, let alone running it on the challenging track. Hence, only the provided data is used in this project.

![alt text][image3]
![alt text][image4]

_Original Image & Flipped Image_

To generate more dataset for the model, the left and right dashboard camera images are also used to train the network.
The steering angle for both left and right images are adjusted by +0.2 for the left frame and -0.15 for the right frame.
Next, for every single image read from the dataset, the image is flipped and the steering angle is mutiplied by -1 to 
mirror the original image.

![alt text][image5]
![alt text][image6]
![alt text][image7]

_Left, Center, and Right Dashboard Camera Images_

Neural network require large amount of data to produce better model. Therefore, using multiple camera and flipped images allow
the model to learn faster. It's easy for human to understand how to drive, even with a non-ideal control method such as using the keyboard
to steer the car. However, it is difficult for neural network to grasp the semantic definition of driving without optimal data.
In whole, a model can only do as best as the data provided. For instance, the model will need driving data using a steering wheel to drive like a real driver.

## Model Fine-Tuning

Through multiple attempts, adding dropout layers does not increase accuracy within 5 epochs. Several attempts of adding dropout layer between different layers have been made, but 
training the model at a larger epoch of 10 with a single dropout layer after the final convolutional layer does reduce the loss greatly and prevent overfitting.
Before training the model, the dataset is shuffled and split into 80% training data and 20% validation data. Note that testing data is not generated for this project as the final testing step 
is to let the model drive the car autonomously in the driving simulator.

The learning rate of the model was not tuned manually as the Adam optimizer is chosen for the model. Adam is a gradient descent optimization algorithm which will change the learning 
rate in between training progress to reduce loss.

## Result and Conclusion

The final model was able to drive the car without leaving the track or rolling over any surfaces. This project, again emphasize on importance of input data in deep learning. 
I have tried collecting data multiple times, but was unable to produce a satisfactory sets of data. Making changes in parameters have little to none effect on the model whereas 
feeding quality data greatly improves the car performance.

I enjoyed working on this project, from building the network to driving the car autonomously. It sure is rewarding to be able to see the car driving around the track smoothly. 
I hope to revisit this project in the future if time permits. I believe driving the car in training mode with a racing wheel will generate great data as it simulates real driving situation.
Another way to improve the model is to implement smaller or stronger neural network to reduce training time and improve performance.