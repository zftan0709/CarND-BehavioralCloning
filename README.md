# **Udacity Behavioral Cloning Project** 
---

[//]: # (Image References)

[image1]: ./README_image/simulator.jpg "Simulator"
[image2]: ./README_image/cnn-architecture.png "NVIDIA CNN Architecture"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

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

To generate more dataset for the model, the left and right dashboard camera images are also used to train the network.
The steering angle for both left and right images are adjusted by +0.2 for the left frame and -0.2 for the right frame.
Next, for every single image read from the dataset, the image is flipped and the steering angle is mutiplied by -1 to 
mirror the original image.

Neural network require large amount of data to produce better model. Therefore, using multiple camera and flipped images allow
the model to learn faster. It's easy for human to understand how to drive, even with a non-ideal control method such as using the keyboard
to steer the car. However, it is difficult for neural network to grasp the semantic definition of driving without optimal data.
In whole, a model can only do as best as the data provided. The model will need driving data using a steering wheel to drive like a real driver.

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)




#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
