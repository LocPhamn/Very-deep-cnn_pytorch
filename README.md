#  [PYTORCH] Very deep convolutional networks for Image Classification
## Introduction
Here is my pytorch implementation of the model described in the paper Very deep convolutional networks for Image Classification 
## Dataset
My dataset includes 10 classes about 10 animal species 
## Setting
 For optimizer and learning rate I use:
SGD optimizer with different learning rates (0.01 in most cases).
for model i have 2 options:
pre train model : resnet18 
custom model : MyNeuronNetwork
with loss func i use cross entropy loss
## training
this is my training process: 
![My Image](./images/img.png)
