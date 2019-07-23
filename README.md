# **Behavioral Cloning** 

---

### Project Description

In this project I used deep neural networks to clone car driving behavior. The data consisted of images collected from a simulator through 3 cameras mounted on the front of the car (center, left, right), and the steering angles. Thus, it is a supervised learning with the steering angle as a target. 

Image preprocessing included several operations : cropping, resizing, normalization and flipping. 

The model architecture is based on [NVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) for behavioral cloning, which uses a set of convolutional layers followed by fully connected layers.


[//]: # (Image References)

[image1]: examples/crop.png "Model Visualization"
[image2]: examples/cameras.png "Model Visualization"

### Files included

- `model.py` : Building and training the model .
- `drive.py` : Driving the car in the simulator after training.
- `utils.py` : Useful function for preprocessing and data augmentation.
- `my_model.h5` : Model weights.
- `output_video.mp4` : The output video of testing the model in the simulator.

### Data preprocessing

The training images were 160x320x3 and contained useless information such as trees, the sky and the hood of the car. Therefore, I cropped 50 pixels from the top and 20 pixels from the bottom of all images and I resized them to 66x200x3 in order to fit the NVIDIA model. Finally, I normaized the data to speed up convergence. 

![alt text][image1]

### Data augmentation

- Flipping images: The simulator track contain more left turns than right ones. To avoid the bias resulting from this unbalanced data, I flipped the images with a probability of 50%. The steering angle corresponding to the flipped image is then multiplied by -1.
- Adding left and right images: For the model to recover from going off the road, I added the images coming from the left and right camera. A coefficient of 0.25 is added to the steering angle of the left image and substracted from the steering angle of the right one to compensate the distance between from the center camera.

![alt text][image2]

### Model architecture

The architecture design was inspired by the [NVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) that have been used in their End-to-End Deep Learning for Self-Driving Cars project. The architecture is as follows:

- Normalization
- Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
- Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
- Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
- Drop out: 0.5%
- Fully connected: neurons: 100, activation: ELU
- Fully connected: neurons:  50, activation: ELU
- Fully connected: neurons:  10, activation: ELU
- Fully connected: neurons:   1 (output)

Dropout was used to avoid the model from overfitting, and the ELU function introduced non-linearity and fixed the dying ReLU problem.

### Training and validation

The model was trained for 10 epochs with a batch size of 64 samples on 34197 images and validated on 8550. A Python Generator was used to yield batches when needed without storing them in memory. 
In addition, I used the Mean Squared Error loss and Adam Optimizer with a learning rate of 0.0001, tuned to avoid both overfitting and underfitting.
