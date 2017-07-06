# **Traffic Sign Classification** 


[//]: # (Image References)

[image1]: ./networkvisualizationtensorboard.PNG "Visualization"
[image2]: ./signvisualize.PNG "Signs"
[image3]: ./softmaxprobabilities.PNG "softmax_probabilities"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

Here is a link to my [project code](https://github.com/FreedomChal/traffic-sign-classification/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### Summary of the data set

To get the size of the dataset, I simply used vanilla python to get the length of X_train, X_valid, and X_test, a slightly more complicated length finding technique for image_shape, and a set for n_classes.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### Visualization of the dataset

Here is an exploratory visualization of the data set. It simply shows 5 labeled images with transformations:

![alt text][image2]

### Model Architecture, training, and data preprocessing

#### Preprocessing of the the image data

To generate new training data, I run the data through a generator that rotates, warps, shears, and zooms in and out on the image to prevent overfitting, and to improve the model's results on imperfect images.

![alt text][image2]

As a last step, I normalized the image data because otherwise the gradients would become extreme.

Here is an example of some unaugmented images and some augmented images:

unagumented:
![alt text][image3]

augmented:
![alt text][image2]

The difference between the original data set and the augmented data set is that the colors are blurred, and the image is warped. 

#### Model architecture

My final model consisted of the following:
![alt text][image1]
 
As you can see, at first, there are three different convolutional paths, which are then concatenated in an inception module. Then, the data goes through three convolutional layers, then through three fully connected layers, which then is turned into the output logits with a fully connected layer with 43 outputs and a softmax activation applied.


#### Model training

To train the model, I used a relatively low learning rate, but a high batch size and number of epochs, so though it takes a while for the model to train, it is fairly stable, and will get good results with sufficent training time. Also, I designed the model to only save when the validation accuracy is greater than 0.93, and is greater than the previous highest validation accuracy the model has seen. This allows the model to be stopped easily when the accuracy peaks, and never saves the model if it gets worse.

#### Training results

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 96.4%
* test set accuracy of 93.9%

At first, I basically used the [Lenet Archetecture](https://github.com/udacity/CarND-LeNet-Lab), only with more layers. Over time, I found several fundemental errors in the code that made the model not train properly, and fixed them. Later, I added dropout, an inception module, and more fully connected and convolutional layers. I discovered that my model tends to do better with a larger batch size and fairly low learning rate, which may be due to it having an ability to train faster, and therefore learn more in a shorter period of time, and being stable due to the low learning rate.

### Testing the Model on New Images

#### Signs from the internet

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

I discovered that my model did extremely good on the last three images, but horribly on the first two. This suggests that my model is only good at classifing signs that are in the center of the image, as the first two images are not very well centered.

#### Predictions and certainty of my model

For the last three images, the model is almost completely right; the highest probability of an incorrect answer being predicted is ~0.00000000001437%. Yet, with the first two, the prediction is completely wrong. The correct prediction is not in the top five probabilities for either of them, In fact, the model is fairly certain on a single wrong answer being correct, which suggests, as said before that the model is only good at classigfing well-centered images.

![alt text][image3]
