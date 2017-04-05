#**Term1-P3 Behavioral Cloning** 

###Here is my solution for Behavioral Cloning.
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[nvidia-cnn]: ./examples/nvidia-cnn.jpg "Model Visualization - Nvidia CNN"
[center-lane]: ./examples/center-lane.jpg "Center lane driving"
[flip-before]: ./examples/flip-before.jpg "Raw Image"
[flip-after]: ./examples/flip-after.jpg "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed -- I am using the Nvidia Net.

My model located in lines 79-96 in model.py.

First, the data is normalized in the model using a Keras lambda layer (code line 81) and cropped from top 70 and bottom 25 (code line 82).
PS: There is a bug in Cropping2D of Keras, which is the number of cropping pixels can not be zero. That would cause zero size in that dimension. I set (2, 2) cropped in horizontal instead of (0, 0).

Then it consists of 3 convolution neural network with 5x5 filter sizes with depths 24, 36, 48, and 2 convolution neural network with 3x3 filter sizes with depths 64. (model.py lines 79-87)

All the convolution layers are using RELU to introduce nonlinearity.

Finally the model includes a Flatten layer and 4 Dense layers with neurons 100, 50, 10, 1.

####2. Attempts to reduce overfitting in the model

I am not using dropout layer.

The model was trained and validated on augmented data sets to ensure that the model was not overfitting (code line 19-30). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 125).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, including center/left/right perspectives to augment the data, and along with fliping all the images. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first step was to use a convolution neural network model similar to the LeNet and NvidiaNet with the train data by Udacity.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

Then i tested both LeNet and NvidiaNet in the Udacity train data. I found NvidiaNet works a bit better than LeNet, but still got off track in some spots.

To improve the driving behavior in these cases, I collected more data with augment.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 79-96) .

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][nvidia-cnn]

####3. Creation of the Training Set & Training Process

First i was using Udacity train data to be the basic data.

To capture good driving behavior, I recorded one lap on track one using center lane driving. And i was using all center/left/right images together. Here is an example image of center lane driving:

![alt text][center-lane]

To augment the data sat, I also flipped images and angles thinking that this would help generalize. For example, here is an image that has then been flipped:

![alt text][flip-before]
![alt text][flip-after]

After test on these data, i found the car got off tract on two spots after the bridge, both are curves. So i took curve driving data in these two specific curves.

After the collection process, I put 20% of the data info a validation set. The numbers of train and validation are:
Train samples: 54710
Validation samples: 13678

Then i resampled appropriate number of samples from those data, to avoid keras warnings with number of samples not divided by BATCH_SIZE exactly. (code line 46-47) The Final numbers are:
Train samples: 54688
Validation samples: 13664

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by MSE almost no change after 3 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
