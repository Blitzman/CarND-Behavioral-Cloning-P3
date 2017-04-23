# **Behavioral Cloning** 

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

The goals of this project are the following:
* Use a simulator to collect data of good driving behavior
* Build a Convolutional Neural Network in Keras that predicts steering angles from images
* Train and validate the model with the recorded data
* Test that the model successfully drives around the tracks

[//]: # (Image References)

[datasethist1]: ./img/dataset_histogram.png "Dataset Histogram"
[datasethist2]: ./img/dataset_histogram2.png "Dataset Histogram Balanced"
[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

The project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* This README.md summarizing the results and documenting the project

#### 2. Submission includes functional code

Using the Udacity provided simulator and the drive.py file, the car can be driven autonomously around the track by executing

```sh
python drive.py model.h5
```
The car is able to drive around both tracks meeting the specified requirements.

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the Convolution Neural Network into an .h5 file. The file shows the pipeline we used for loading, preprocessing, and augmenting the dataset as well as for training and validating the model. It contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model consists of a Convolutional Neural Network based on the model shown by NVIDIA in their paper. The model includes certain additions such as RELU layers to introduce nonlinearities, data cropping, data normalization using a lambda layer, and dropout.

More details about the final network architecture and preprocessing steps will be provided in the next section.

#### 2. Attempts to reduce overfitting in the model

As we have already stated, the model contains three dropout layers: one after each fully connected or dense layer. The keep probability for the dropout layers was set to 0.5 as it was found empirically to work well to reduce overfitting.

In addition, we set up early stopping using the corresponding callback in Keras. By doing that we monitor both training and validation losses and we stop training if the validation loss increases instead of decreasing.

The model was trained and validated on different data sets to ensure that the model was not overfitting, we used Keras to setup the validation partition. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track and meet the requested specifications, e.g., not driving outside the road or the shoulders.

#### 3. Model parameter tuning

The model used an ADAM optimizer, so the learning rate was not tuned manually, although we chose an starting learning rate of 1e-4 which was determined empirically to work properly after a brief experimentation.

#### 4. Appropriate training data

The training data we used consisted of the original data provided by Udacity for the project and various recordings using the simulator under different conditions on both tracks. The purpose of those recordings were to add more data for smooth lane-center driving, recoveries, and gather more data for not so common situations such as bridges or other critic zones.

In the end, the dataset consisted of the following parts:
* Original data from Udacity
* Slow and smooth lane driving (keeping the car centered)
* Lane driving in the opposite direction
* Recovery driving forcing the car to the edges of the lane and using sharp turns
* Recovery driving in the opposite direction
* Driving on specific zones (bridges, shadowed zones...)

For details about how we created the training set, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to incrementally build up over the knowledge acquired during the Nanodegree.

The first step was to use a simple CNN such as LeNet-5. We thought that, initially, this model might be appropriate and generate decent results since we used it successfully to perform traffic sign recognition in the previous project. However, it soon became obvious that the model was too simple for the problem at hand and really prone to underfitting. The car was not able to get past the first curve in the test track.


In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (defined in model.py lines 226 to 249) consisted of a variation of NVIDIA's CNN as showh in the following table:

| Layer									|     Description																								| Output shape	|
|:---------------------:|:-------------------------------------------------------------:|:--------------|
| Input									| 160x320x3 YUV image																						| 160x320x3			|
| Lambda								| Normalization (Input / 127.5 + 1.0)														|	160x320x3			|
| Cropping2D						| Crop 50 pixels from top																				|	90x320x3			|
| Convolution2D					| 5x5 kernel, 2x2 subsample, VALID padding, L2 reg (0.001)			| 43x158x24			|
| ELU										| Activation																										| 43x158x24			|
| Convolution2D					| 5x5 kernel, 2x2 subsample, VALID padding, L2 reg (0.001)			| 20x77x36			|
| ELU										| Activation																										| 20x77x36			|
| Convolution2D					| 5x5 kernel, 2x2 subsample, VALID padding, L2 reg (0.001)			| 8x37x48				|
| ELU										| Activation																										| 8x37x48				|
| Convolution2D					| 3x3 kernel, 2x2 subsample, VALID padding, L2 reg (0.001)			| 6x35x64				|
| ELU										| Activation																										| 6x35x64				|
| Convolution2D					| 3x3 kernel, 2x2 subsample, VALID padding, L2 reg (0.001)			| 4x33x64				|
| ELU										| Activation																										| 4x33x64				|
| Flatten								| -																															| 8448x1x1			|
| Dense									| 100 neurons, L2 reg (0.001)																		|	100x1x1				|
| Dropout								| Keep probability 0.5																					| 100x1x1				|
| ELU										| Activation																										|	100x1x1				|
| Dense									| 50 neurons, L2 reg (0.001)																		|	50x1x1				|
| Dropout								| Keep probability 0.5																					| 50x1x1				|
| ELU										| Activation																										|	50x1x1				|
|	Dense									|	10 neurons, L2 reg (0.001)																		|	10x1x1				|
| Dropout								| Keep probability 0.5																					|	10x1x1				|
| ELU										| Activation																										| 10x1x1				|
| Dense									| 1 neuron																											| 1x1x1					|

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

![Dataset Histogram][datasethist1]
![Dataset Histogram Balanced][datasethist2]
