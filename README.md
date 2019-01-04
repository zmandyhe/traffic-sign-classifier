# **Traffic Sign Recognition** 
## Overview
In this project, I used what I have learned about deep neural networks and convolutional neural networks to classify 43 traffic signs using [the German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). I trained a model based on LeNet architecture so it can decode traffic signs from the datasets. After the model was trained, I then tested my model performance on the validation dataset to find tune the hyperpromaters. Once the final model is selected, I used the model to test on new images of traffic signs I found on the web.

**The steps of this project are the following:**
* Load the data set (training, validation, test)
* Explore, summarize and visualize the data set
* Pro-process, design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Dependiencies
* Jupyter
* NumPy
* SciPy
* scikit-learn
* TensorFlow
* Matplotlib
* OpenCV

## Project Steps

### Data Set Summary & Exploration

#### 1. Overview of the basic summary of the data set

The three datasets are pickled data which are providing a dictionary with 4 key/value pairs:

* 'features' is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
* 'labels' is a 1D array containing the label/class id of the traffic sign. The file signnames.csv contains id -> name mappings for each id.
* 'sizes' is a list containing tuples, (width, height) representing the original width and height the image.
* 'coords' is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. 

The sizes of the datasets are:
* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is (32, 32, 3), each is a RGB colored image.
* The number of unique classes/labels in the data set is 43.

Noticed that LeNet accepts input of 32 x 32 x C, the dataset is ready to feed in the LeNet with or without grayscale the images.

#### 2. Exploratory visualization of the dataset

The image quality are diversified with some of them are fairly good while others are low quality to recognize the type of sign visually.

![Training Data Set](https://github.com/zmandyhe/traffic-sign-classifier/blob/master/pic/X_train.png "Training Data Set")
![Validation Data Set](https://github.com/zmandyhe/traffic-sign-classifier/blob/master/pic/X_valid.png "Validation Data Set")
![Test Data Set](https://github.com/zmandyhe/traffic-sign-classifier/blob/master/pic/X_test.png "Test Data Set")

The 43 classes in each dataset are distributed fairly in the similar ratio, below shown the statistics:
![traffic signs labels in training dataset](https://github.com/zmandyhe/traffic-sign-classifier/tree/master/pic/y_train.png "traffic signs labels in training dataset") ![traffic signs labels in validation dataset](https://github.com/zmandyhe/traffic-sign-classifier/tree/master/pic/y_valid.png "traffic signs labels in validation dataset") ![traffic signs labels in test dataset](https://github.com/zmandyhe/traffic-sign-classifier/tree/master/pic/y_test.png "traffic signs labels in test dataset")

### Preprocess, Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)
As a first step, I decided to augment the training dataset by random blurring, brightening, and rotating techniques to match closely with the real-road situation. I finally chose to augment 30% images for each class from the training dataset, then merged these augmented new images with the training dataset to form an expanded new training dataset. The total number of samples become 46,892. Here is an example of an original iamge and an augmented image:
[Augmented Image](https://github.com/zmandyhe/traffic-sign-classifier/tree/master/pic/rotated.png)

As a last step, I then decided to convert the images to grayscale, and normalized them in training, validation, and test datasets.

Here is an example of a traffic sign image before and after grayscaling.
[Grayscaled and Normalized image](https://github.com/zmandyhe/traffic-sign-classifier/tree/master/pic/grayscale-normalized.png)

The difference between the original data set and the augmented data set is the augmented data set contain lower quality of images such as blurrer, brightening light, or from from different angles to a traffic sign. It feeds data with more diverse perspectives to each class labels to produce a more stable model for prediction.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.
My final model is based on LeNet architecture, with added dropout in the last fully connected layer to prevent overfitting. The model overview is as follows:
[Final Model Architecture](https://github.com/zmandyhe/traffic-sign-classifier/tree/master/pic/LeNet-architecture.png)

My final model consisted of the following layers:

| Layer                 |     Description                                | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 32x32x1 grayscaled image                               | 
| Layer 1 Convolutional  5x5         | 1x1 stride, VALID padding, outputs 28x28x6     |
| RELU                    |                                                |
| Max pooling              | 2x2 stride,  VALID padding, outputs 14x14x6                 |
|Layer 2 Convolution 5x5        | 1x1 stride, VALID padding, outputs 10x10x16     |
| RELU                    |                                                |
| Max pooling              | 2x2 stride,  VALID padding, outputs 5x5x16                 |
| Flatten              | output 400                 |
| Layer 3 Fully connected        | input 400, output 120     |
| RELU                    |                                                |
| Layer 4 Fully Connected     | input 120, output 84      |
| RELU                    |                                                |
| Dropout                        | keep_probl = 50%                |
| Layer 5 Fully Connected               | input 84, output 43           |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used tf.train.AdamOptimizer function for training the CNN model. The Adam optimizer is a robust stochastic gradient-based optimization method suited for nonconvex optimization and machine learning problems. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy: 0.757
* validation set accuracy: 0.960
* test set accuracy: 0.933

* What was the first architecture that was tried and why was it chosen?
I chose the LeNet architecture as its a multilayer Perceptron algorithm working well on the MNIST dataset. I did not use dropout method at first, which resulted a heavy overfitting with a low valadiation accuracy, so I tried to add the dropout in different layer. The current model is to have the dropout before the final fully connected layer which produced a higher validation accuracy than the other experiments.
* What were some problems with the initial architecture?
The initial architecture produce a higher training accuracy but a much lower validation accuracy (overfitting). I added dropout before the last fully connected layer, and took a longer time in pre-processing 30% of the images randomly with blurring, rotation, and brightening which are then added into the new training dataset.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
With the newly added augmented images, the training dataset is relatively large. A slower learning rate of 0.0005 learned better than learning rate of 0.001.
Although a higher epochs shall produce a better model, I found with current preprocessed data and model architecture, a epochs of 100 is about the highest level the model can learn. My current training on 200 epochs ((0.960 validation accuracy) helps very little with model improvement comparing with 100 epochs (0.951 validation accuracy). 
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
The training accuracy is much lower than the validation accuracy due to the data augmentation that reduced the data quality. I trained 200 EPOC with 128 BATCH_SIZE, so the total number of samples images used are 55% of the total samples in the training dataset. The final training accuracy is increased from 0.755 in EPOCHS 50 to 0.757 in EPOCHS 200 with a slower improvement rate, but the validation accuracy rate is increased from 0.947 in EPOCHS TO 0.960 in EPOCHS 200. 

The final learning rate is set to 0.0005, and with this relatively larger training data, the slower learning rate works better to tune a higher accuracy model.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

[8 Traffic Signs from Web](https://github.com/zmandyhe/traffic-sign-classifier/tree/master/pic/8-web-traffic-signs.png)

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set

The model was able to correctly guess 6 of the 8 traffic signs, which gives an accuracy of 75%. This has similar accuracy on newly expanded traing dataset with data augmentation. Validation accuracy and test accuracy are higher, they are respectively 0.960 and 0.933.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability

The prediction result for the eight new images are:

|                                   Label                                     |                        Prediction                                | 
|:---------------------:|:-------------------------------------------------------------------------------------:| 
| Right-of-way at the next intersection                     | Right-of-way at the next intersection             | 
| Priority road                                                            | Priority road                                                    |
| Speed limit (60k/h)                                                  | Speed limit (60k/h)                                         |
| Keep right                                                               | Keep right                                                      |
| General Caution                                                     | General Caution                                             |
| Speed limit (30k/h)                                                  | Speed limit (50k/h)  (probability of 0.64)      | 2nd guess (probability of 34) predict Speed limit (30k/h))
| Turn left ahead                                                       | Turn left ahead                                                |
| Roadwork                                                               | Bicyble crossing (probability of 0.62)           | 2nd guess (keep left, probability of 0.34), 3rd guess(Bumpy Road, probability of 0.03),
                                                                                                                                                         4th guess (Roadwork, probability of 0.02)

For the two mis-predicted images, the model is relatively unsure. The top five soft max probabilities were shown as follows. The images among the correct label and wrongly predicted labels contain pariticial simalities. Such as, the model could not see the difference between 30 and 50 to some images; pedestrian are shown on both roadwork sign and bicycles corssing signs, which confused the model. Further work can be focusing on improving the model on these relatively difficult classes.
[Class 1](https://github.com/zmandyhe/traffic-sign-classifier/tree/master/pic/input1-misprediction.png)
[Class 1](https://github.com/zmandyhe/traffic-sign-classifier/tree/master/pic/input24-misprediction.png)

### Re-produce This Notebook Pipeline:

1. Download [the German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). This is a pickled dataset in which we've already resized the images to 32x32.
2. Clone the project and start the notebook.
```
git clone https://github.com/zmandyhe/traffic-sign-classifier
jupyter notebook Traffic_Sign_Classifier.ipynb
```
Follow the instructions in the Traffic_Sign_Classifier.ipynb notebook.
