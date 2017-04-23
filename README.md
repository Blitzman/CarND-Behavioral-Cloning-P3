# **Behavioral Cloning** 

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

The goals of this project are the following:
* Use a simulator to collect data of good driving behavior
* Build a Convolutional Neural Network in Keras that predicts steering angles from images
* Train and validate the model with the recorded data
* Test that the model successfully drives around the tracks

[//]: # (Image References)

[conflict1]: ./img/conflict1.jpg "Conflict 1"
[conflict2]: ./img/conflict2.jpg "Conflict 2"
[conflict3]: ./img/conflict3.jpg "Conflict 3"
[conflict4]: ./img/conflict4.jpg "Conflict 4"
[datasethist1]: ./img/dataset_histogram.png "Dataset Histogram"
[datasethist2]: ./img/dataset_histogram2.png "Dataset Histogram Balanced"
[mse]: ./img/mse.png "MSE"
[dataset1]: ./img/dataset1.jpg "Dataset 1"
[dataset2]: ./img/dataset2.jpg "Dataset 2"
[dataset3]: ./img/dataset3.jpg "Dataset 3"
[dataset4]: ./img/dataset4.jpg "Dataset 4"
[dataset5]: ./img/dataset5.jpg "Dataset 5"
[dataset6]: ./img/dataset6.jpg "Dataset 6"
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
* track1.mp4 the video of the car driving autonomously on the first track
* track2.mp4 the video of the car driving autonomously on the challenge track
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

Furthermore, L2 regularization was applied to each convolutional and dense layer using a regularizing factor of 0.001.

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

After that, we introduced a more complex network such as the one from NVIDIA proposed in the lessons. This network performed reasonably well and the car was able to get past the first curve and properly drive straight. Nevertheless, it still failed when getting to the bridge of the first track.

We switched color spaces from RGB to YUV, normalized inputs, and cropped the image to remove the hood of the car and some horizon which contained information which was not important for driving autonomously. After training the model (with the data provided by Udacity plus recovery laps recorded by ourselves), it was able to complete the whole track without getting off the road.

Providing that it was able to drive on the first track (being trained only using that track's data) we tried to run it on the challenge track without success. It was not even able to get to the first curve. Obviously, we introduced the basic training data for that track (center lane driving in both directions plus some recovery data). After training the model, it was able to drive autonomously on most of the second track but it still failed at some conflicting points:

![Conflict1][conflict1]
![Conflict2][conflict2]
![Conflict3][conflict3]
![Conflict4][conflict4]

What is more, the car stopped driving properly on the first track. We noticed that we needed to add some kind of regularization mechanisms to avoid it from overfitting certain parts of the tracks so we added dropout layers and L2 regularization for each convolutional and dense layer. We also introduced ELU activations which improved greatly the results. With all those adjustments the car was able to drive properly on the first track again but it still failed on the conflicting points of the challenge track.

We solved those problems by taking two approaches. First, we introduced more training data by driving the car slowly on those conflicting points and inducing recovery situations. Second, a visualization of the dataset made obvious that some steering angles were more dominant than others, so we needed some balancing in order not to bias the model towards driving straight. More details about the dataset generation, augmentation, and balancing process are provided in the next section.

At the end of the process, the model exhibited a proper learning behavior (as shown in the MSE plot) and the vehicle was able to drive autonomously around the track without leaving the road as shown in track1.mp4 and track2.mp4.

![MSE][mse]

In addition, we captured a couple of videos from the simulator for each track (click to watch):

[![Track1](http://img.youtube.com/vi/ePD9udGOWWA/0.jpg)](http://www.youtube.com/watch?v=ePD9udGOWWA "Self-Driving Car Nanodegree - P3: Behavioral Cloning - Track1")

[![Track2](http://img.youtube.com/vi/4WAxWsDqyrw/0.jpg)](http://www.youtube.com/watch?v=4WAxWsDqyrw "Self-Driving Car Nanodegree - P3: Behavioral Cloning - Track2")


#### 2. Final Model Architecture

The final model architecture (defined in model.py lines 226 to 249) consisted of a variation of NVIDIA's CNN as showh in the following table:

| Layer									|     Description																								| Output shape	|
|:---------------------:|:-------------------------------------------------------------:|:--------------|
| Input									| 160x320x3 YUV image																						| 160x320x3			|
| Lambda								| Normalization (Input / 127.5 - 1.0)														|	160x320x3			|
| Cropping2D						| Crop 50 pixels from top and 20 from bottom										|	90x320x3			|
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

The initial training set contained the data provided by Udacity (namely original). For the first track, we recorded an additional lap of center lane driving (t1_center1) and two recovery laps, each one of them in one direction (t1_recovery1 and t1_c_recovery1).

For the second track, we recorded a set of center lane driving laps (t2_center1, t2_center2, and t2_center3) and one of them in the opposite direction (t2_c_center1). We also captured one lap driving slowly and doing smooth turns (t2_centerslow1). Then we captured a couple of whole-lap recoveries in both directions (t2_recovery1 and t2_c_recovery1).

For the aforementioned conflicting points we recorded those specific parts either by doing recovery runs or smooth drives (t2_recoveryend1, t2_recoveryend2, t2_recoveryend3, t2_recoveryend4, t2_recoveryend5, t2_recoveryturn1, and t2_smoothturn1).

All captured data is located in the data folder. We used the three camera images for training, applying a 0.25 correction factor to the ground truth steering to the left and right images accordingly.

To augment the dataset, we implemented a generator which randomly flipped images with a 1/3 probability.

Finally, we visualized the dataset (as inspired by Jeremy Shannon) to analyze the distribution of steering angles. We used a histogram visualization with eight bins.

![Dataset Histogram][datasethist1]

It was obvious that the dataset was unbalanced towards straight driving so we decided to balance it by keeping more samples that belonged to sharp turns instead of straight driving. By doing that we came up with a better dataset which helped the model learn to perform sharp turns too and drive properly on the conflicting points which were underrepresented before.

![Dataset Histogram Balanced][datasethist2]

Finally, we used a train/test split of 80/20, and two generators to provide shuffled and randomly augmented training and validation batches respectively to the model fitting method.

![Dataset 1][dataset1]
![Dataset 1][dataset2]
![Dataset 1][dataset3]
![Dataset 1][dataset4]
![Dataset 1][dataset5]
![Dataset 1][dataset6]
