#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]:  ./examples/example.jpg "Example"
[image22]:  ./examples/loss.png "Train loss"
[image23]:  ./examples/acc.png "Train acc"
[image24]:  ./examples/lenet.jpg "LeNet"
[image4]: ./examples/new_traffic_sign.jpg "Traffic Sign 1"
[image5]: ./examples/new_traffic_sign_gray.jpg "Traffic Sign 2"

[image61]: ./examples/softmax_1.jpg "softmax result"
[image62]: ./examples/softmax_2.jpg "softmax result"
[image63]: ./examples/softmax_3.jpg "softmax result"
[image64]: ./examples/softmax_4.jpg "softmax result"
[image65]: ./examples/softmax_5.jpg "softmax result"
[image66]: ./examples/softmax_6.jpg "softmax result"
[image67]: ./examples/softmax_7.jpg "softmax result"
[image68]: ./examples/softmax_8.jpg "softmax result"
[image69]: ./examples/softmax_9.jpg "softmax result"

[image7]: ./examples/feature_map.jpg "Feature map"



###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/gwwang16/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distribute.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I converted the images to grayscale to release the computational burden. (RGB has 3 channels and grayscale has 1 channel).
There would have a dimensional problem, I reshaped data into 4 dimensions use `np.reshape()`.
```
X_train_gray = np.dot(X_train[...,:3], [0.299, 0.587, 0.114])
X_train_gray = np.reshape(X_train_gray, X_train_gray.shape + (1,))
```

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3]
![alt text][image2]

I tried to normalized the image, however, I found it has a worse performance for my built model. Hence, I didn't use this normalization function in this project.

```
def data_normal(x):
    mid_value = np.max(x)/2.
    return (x - mid_value) /mid_value
```


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

|      Layer      |               Description                |
| :-------------: | :--------------------------------------: |
|      Input      |         32x32x1 grayscale image          |
| Convolution 3x3 | 1x1 stride, same padding, outputs 32x32x32 |
|      RELU       |                                          |
| Convolution 3x3 | 1x1 stride, same padding, outputs 32x32x64 |
|      RELU       |                                          |
|   Max pooling   |      2x2 stride,  outputs 16x16x64       |
| Convolution 3x3 | 1x1 stride, same padding, outputs 32x32x128 |
|      RELU       |                                          |
| Convolution 3x3 | 1x1 stride, same padding, outputs 32x32x128 |
|      RELU       |                                          |
|   Max pooling   |       2x2 stride,  outputs 8x8x64        |
| Fully connected |        dropout 0.5,  outputs 512         |
|      RELU       |                                          |
| Fully connected |         dropout 0.5, outputs 256         |
|      RELU       |                                          |
|     Softmax     |                outputs 43                |



####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

`AdamOptimizer` is used to train the model and the hyper parameters are
- epochs = 20
- batch_size = 128
- learning_rate = 0.0002


####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.0
* validation set accuracy of 0.98
* test set accuracy of 0.964

The following figures are training loss and training accuracy, respectively.
![alt text][image22]
![alt text][image23]

The LeNet architecture is adopted in first. The output shape has been changed into 43 to consistent with this project.
![alt text][image24]

However, its performance is limited (<0.96) with the current training dataset,  I think the reason might be conv1 layer depth is too smaller, i.e., filter number of the 1st layer is 6. The CNN model cannot extract enough features in the first layer for the following process.

I added more filters, 32 filters,  and didn't implement pool layer in the first layer. moreover, I added dropout in full connected layers to prevent overfitting problem.  The final model can be found in above.



###Test a Model on New Images

####1. Choose German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I selected nine German traffic signs that I found on the web:
![alt text][image4]
And their grayscale images are
![alt text][image5]

The 1st, 8th and 9th images might be difficult to classify.

- With the down sampling of the image, the second image has been  quite vague

- The eighth image with low resolution looks like two straight lines

- The last image is very simple with caution sign


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

|                Image                |      Prediction      |
| :---------------------------------: | :------------------: |
|        Speed limit (50km/h)         |       correct        |
|          Bicycles crossing          |       correct        |
|         Go straight or left         |       correct        |
| End of all speed and passing limits |       correct        |
|        Speed limit (100km/h)        | Speed limit (30km/h) |
|             Bumpy road              |       correct        |
|             Ahead only              |       correct        |
|      Road narrows on the right      |   Turn right ahead   |
|           Traffic signals           |   General caution    |

The model was able to correctly guess 6 of the 9 traffic signs, which gives an accuracy of 67%. This is lower than the accuracy on the test set  of 96%.

- The 100km/h speed limit sign is identified as 20km/h speed limitation, the reason might be 100km/h limitation sign, which is classes 7, is only ~1/4 of other speed limit signs  in the training dataset, as shown in the data distribution histogram. More images of 100km/h speed limit sign are preferred in training dataset.
- This  `road narrows on the right` sign is slant, the image down sampling also increased the error of the model prediction. 
- The traffic signals are similar with general caution sign, in particular with grayscale image. RGB or YUV channels would be help for this problem.


####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Visualizations of the  top five softmax probabilities

![alt text][image61]
![alt text][image62]
![alt text][image63]
![alt text][image64]
![alt text][image65]
![alt text][image66]
![alt text][image67]
![alt text][image68]
![alt text][image69]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

From these feature maps, it seems that the CNN focused more on signs' boundaries. There are also some pure black filters in these feature maps, the reason might be that 32 filters are redundant for this project, some of them have been misfired. 


![alt text][image7]



### Improvement

- The augmented data can be generated using rotation, flip, local zoom methods. I'd like to use more data in the future.
- This model is trained from scratch, I hope I can apply transfer learning in the following works with the help of modern CNN architectures, such as resnet152, VGG, etc.

