# **Traffic Sign Recognition** 


## Goal 
**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[histogram]: ./examples/hist.png "Histogram"
[orig_rgb_hist]: ./examples/original_rgb_hist.png "Original RGB histogram"
[clahe_rgb_hist]: ./examples/clahe_rgb_hist.png "CLAHE RGB histogram"
[web_img_01]: ./traffic-signs-data/internet_images/01_speed_limit_30.png "Traffic Sign 1"
[web_img_02]: ./traffic-signs-data/internet_images/07_speed_limit_100.png "Traffic Sign 2"
[web_img_03]: ./traffic-signs-data/internet_images/13_yield.png "Traffic Sign 3"
[web_img_04]: ./traffic-signs-data/internet_images/35_ahead_only.png "Traffic Sign 4"
[web_img_05]: ./traffic-signs-data/internet_images/39_keep_right.png "Traffic Sign 5"
[fm_conv1]: ./examples/feature_maps_con1.png "Feature map Conv Layer 1"


---

**Requirements**

* [Anacoda 3](https://www.anaconda.com/download/) is installed on your machine.
* Download the [data set](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip)

---
## **Getting started**

1. Clone repository:<br/>
```sh
git clone https://github.com/akrost/CarND-TrafficSignClassifier.git
cd carnd-trafficsignclassifier
```

2. Create and activate Anaconda environment:
```sh
conda create --name carnd-p2
source activate carnd-p2
```
Activating the environment may vary for your OS.

3. Install packages:
```sh
pip install -r requirements.txt
```

4. Run project
```sh
jupyter notebook Traffic_Sign_Classifier.ipynb
```

---
## Data Set Summary & Exploration

### 1. Basic summary of the data set

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is?<br/>
=> 34,799

* The size of the validation set is?<br/>
=> 4,410
* The size of test set is?<br/>
=> 12,630
* The shape of a traffic sign image is?<br/>
=> 32x32
* The number of unique classes/labels in the data set is?<br/>
=> 43

### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. The graph shows a histogram of all classes in the three datasets. The blue line shows the distirbution for the training set, orange shows the validation set and green shows the testing set. 

![Histogram of classes per dataset][histogram]


## Model Architecture

### 1. Image processing

#### 1. Contrast enhancement

The graph shows the RGB histogram of the original image. There is a big spike for all three color chanels around 30 (with a maximum of 160) hence the image seems to be very dark.

![RGB histogram of the original image][orig_rgb_hist]


To improve the input data quality the CLAHE algorithm was used to enhance the contrast of the image.
The next graph shows the RGB histogram of the same image that was used above, but this time the contrast was improved. The spike was flattened to around half its value and the overall spectrum was enhanced.

![RGB histogram of the CLAHE image][clahe_rgb_hist]

#### 2. Normalization

After improving the contrasts, the image was normalized using the function
```
pixel_value_new = (pixel_value_old - 128.) / 128. 
```

This step is necessary to make the input data zero centered ranging from -1 to 1.


### 2. Model architecture

My final model consisted of the following layers:

**Model**


| Layer              | Name          | Description                                             |
|--------------------|---------------|---------------------------------------------------------|
| Input              |               | 32x32x3 RGB image                                       |
| Convolution 5x5    | conv1         | 1x1 strides, VALID padding, outputs 28x28x6             |
| RELU               | conv1_relu    |                                                         |
| Max pooling        | conv1_maxpool | 2x2 strides, 2x2 kernel, SAME padding, outputs 14x14x6  |
| Dropout            | conv1_dropout | 0.8 keep_rate                                           |
| Convolution 5x5    | conv2         | 1x1 strides, VALID padding, outputs 10x10x16            |
| RELU               | conv2_relu    |                                                         |
| Max pooling        | conv2_maxpool | 2x2 strides, 2x2 kernelSAME padding, outputs 5x5x16     |
| Dropout            | conv2_dropout | 0.8 keep_rate                                           |
| Fully connected    | fc1           | 400 inputs, outputs 120                                 |
| RELU               | fc1_relu      |                                                         |
| Dropout            | fc1_dropout   | 0.5 keep_rate                                           |
| Fully connected    | fc2           | 120 inputs, 84 outputs                                  |
| RELU               | fc2_relu      |                                                         |
| Dropout            | fc2_dropout   | 0.5 keep_rate                                           |
| Fully connected    | fc3           | 84 inputs, outputs 43 (#classes)                        |
 


### 3. Training
To train the model, I used the following parameters:

* Optimizer: Adam omptimizer
* Batch size: 128
* Epochs: 20
* Learning rate: 0.001
* Keep rate convolutional layers: 0.8
* Keep rate fully connected layers: 0.5


### 4. Description of the approach 

My final model results were:
* training set accuracy of 98.3 %
* validation set accuracy of 96.0 %
* test set accuracy of 94.7 %

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  * I started with the LeNet Architecture since it is a proven architecture for similar problems. 
* What were some problems with the initial architecture?
  * The LeNet architecture had to be adapted to the give image size
  * The original architecture tended to overfit
* How was the architecture adjusted and why was it adjusted?
  * The input depth was adjusted to handle RDG images
  * Size of the fully connected layers had to be adjusted to the new depth
  * Dropout was introduced to both convolutional and fully connected layers to prevent the model from overfitting
* Which parameters were tuned? How were they adjusted and why?
  * The keep rate was tuned for both convolutional and fully connected layers
* What are some of the important design choices and why were they chosen?
  * Probably the most important design choice was to use dropout, since the diverging between the training set accuracy and the validation set accuracy was a strong indicator for overfitting.

If a well known architecture was chosen:
* What architecture was chosen?
  * LeNet as a basis
* Why did you believe it would be relevant to the traffic sign application?
  * Since it proved itself for similar tasks
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
  * All three accuracies are quite high. This should be an indicator that the general architacture is ok and the model is not underfitting too much
  * Training and validation accuracy are close to each other so the model does not seem to overfit
  * Test accuracy is quite high which underlines the two points above
 

## Test a Model on New Images

### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Traffic Sign 1][web_img_01] ![Traffic Sign 2][web_img_02] ![Traffic Sign 3][web_img_03] 
![Traffic Sign 4][web_img_04] ![Traffic Sign 5][web_img_05]

The first two images might be difficult to classify since speed limit signs look very similar to each other, especially with a 32x32 resolution. The third image should not be too hard. The fourth and the fifth image might be harder again, for the same reason as abov.

### 2. Model's predictions

Here are the results of the prediction:

| Image			    | Prediction		| 
|:-----------------:|:-----------------:| 
| 30 km/h      		| 30 km/h			| 
| 100 km/h     		| 30 km/h			|
| Yield				| Yield				|
| Ahead only   		| Ahead only		|
| Keep right		| Keep right        |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80 %. This compares badly to the accuracy on the test set of 94.7 %. This might be due to the very small set tested here.

### 3. Certainty of the model

The code for making predictions on my final model is located in the 25th cell of the Ipython notebook.

For the first image, the model classifies the immage correctly but it is not very sure since other speed limit signs seem to be quite similar. The top five soft max probabilities were

| Probability   |     Prediction	       	| 
|:-------------:|:-------------------------:| 
| **.4913**    	| **30 km/h** 				| 
| .3349			| 20 km/h 					|
| .0915			| Vehicles over 3.5 metric tons prohibited	|
| .0506			| 50 km/h	 				|
| .0087		    | 70 km/h 					|


For the second image, the classification is actually wrong. This time the probabilities are not as close as they were for the first picture, but unfortunately the best probability was wrong.

| Probability   |     Prediction	       	| 
|:-------------:|:-------------------------:| 
| .4653     	| 30 km/h			    	| 
| **.1476**		| **100 km/h**				|
| .1134			| Roundabout mandatory  	|
| .1133			| 80 km/h	 				|
| .0730		    | 50 km/h 					|


For the third image, the model is completely sure that this is a yield sign, and the image does contain a yield sign.

| Probability   |     Prediction	       	| 
|:-------------:|:-------------------------:| 
| **1.**     	| **Yield**    		    	| 
| 0     		| Priority road				|
| 0		    	| No passing              	|
| 0		    	| No vehicles 				|
| 0 		    | No entry					|

For the fourth image, the model is almost as confident as it is for the third image. The model predicts a *ahead only* sign and in the image there is indeed a *ahead only* sign.

| Probability   |     Prediction	       	| 
|:-------------:|:-------------------------:| 
| **.9999*     	| **Ahead only**	    	| 
| .00001   		| Turn left ahead	    	|
| 0		    	| Turn right ahead         	|
| 0		    	| Go straight or right		|
| 0 		    | Go straight or left		|

For the fifth image, the model again is completely ure that this image contains a *keep right* sign, and it does contain this sign.

| Probability   |     Prediction	       	| 
|:-------------:|:-------------------------:| 
| **1.**     	| **Keep right**	    	| 
| 0     		| Turn left ahead			|
| 0		    	| Go straight or right     	|
| 0		    	| Roundabout mandatory  	|
| 0 		    | Dangerous curve to the right|

## Visualization of the Neural Network
### 1. Characteristics the neural network used to make classifications

The image shows the feature map of the first convolutional layer:

![Feature map of first convolutional layer][fm_conv1]

According to th efeature maps the characteristics of this layer are the read edge of the *30 km/h* sign and the number 30 in the middle of the sign. This layer already shows a comprehensive list of features, that identify the sign.
