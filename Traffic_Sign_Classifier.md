# **Traffic Sign Recognition**

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualisation.png "Visualisation"
[image2]: ./examples/random.png "Unprocessed image"
[image3]: ./examples/grayscale.png "Grayscaling & normalisation"
[image4]: ./examples/new_signs.png "Traffic Signs"
[image5]: ./examples/new_signs2.png "Traffic Signs"

---
## Project Source

Link to my [project code](https://github.com/codesmyth/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)


##Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 51839

##2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram showing how the data is comprised of the different signs and their respective frequencies. It should be noted that there are significant differences in the frequencies across the 43 distinct signs. However the validation and testing datasets have similar distributions to the training dataset.

![alt text][image1]

##Design and Test a Model Architecture


As a first step, I decided to convert the images to grayscale because I found that during the additional colour channels didn't make much difference to the accuracy. Because the image was grayscaled I reduced the number for channels from 3 to 1.

Here is an example of a traffic sign image before and after grayscaling and normalisation

![alt text][image2]
![alt text][image3]

As a last step, I normalized the image data because to improve the conditioning of the data as a well conditioned dataset makes it an easier task for the optimiser.

I decided to not to generate additional data and to leverage the architecture of the covnet such that data I had was sufficient to achieve 93% accuracy on the validation set.


##Final Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscaled image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x8 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x8 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16
| Flatten in: 5x5x16 -> 400
| Fully connected		| Input = 400. Output = 120.
| RELU					|												|
| Dropout           | Keep Probability 80%
| Fully connected		| Input = 120. Output = 84.
| RELU					|
| Dropout           | Keep Probability 90%
| Fully connected   | Input = 84. Output = 43.
|						|												|


To train the model, I used an adamoptimiser, with bathes of size 128. I settled on a learning rate of 0.001 and trained the network over 70 epochs. I'd initially used less when I was experimenting but 70 was suffice to hit the 93% target. i'd also used batch sizes 64 & 256 in addition to 128.


My final model results were:
* training set accuracy of 0.996
* validation set accuracy of 0.943
* test set accuracy of 0.933


Initially I had used the colour images, which I'd normalised, so the input to trainging was 32x32x3 for each image. the results were not close enough to 93% so I then applied a grayscale conversion, I'd replicated the grayscale normalised values over the RGB channels. But again didn't see much improvement.

I felt that the duplicated grayscale values on each of the three channels wasn't necessary, so I reduced it to one grayscale channel only. Input 32x32X1.

Next I modified the depths of each convolution and settled on 2 convolutions with output depths 8 and 16 respectfully.

the results where still not good enough until I introduced dropout on the fully connected layers. i tried a number of different values for the keep probability but found a keep probability of 0.8 and 0.9 to be most effective.

The architecture was very much modelled on the lenet architecture as it had been covered in the course material and has been demonstrated to effectively classify images.



##Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image4]

The images where all different sizes so I transformed each one to be 32x32x3, then applied normalisation and grayscaling.
the network predicted each image correctly.


Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Priority Road (12)     		| Priority Road 									|
| Yield	(13)				| Yield											|
| No Entry (17)      | No Entry
| 30 km/h	 (1)     		| 30 km/h	   					 				|
| Turn left (34)         | Turn left                  |

Initially the model was able to correctly guess 5 of the 5 traffic signs, i had selected images that were similar to those in the data set. It gave an accuracy of 100%. This compares favorably to the accuracy on the test set of 0.934.

I ran it again. this time I included and image that wasn't in the data set lables then the accuracy was only ever going to be 80%. however it made a resonable prediction.

![alt text][image5]

For the second image, two eldery people crossing, which wasn't in the set of labels the model is relatively sure that this is a "Vehicles over 3.5 metric tons prohibited" (probability of 0.98) and more interestingly that it is a "Dangerous curve to the right" (probabiliy of 0.15) the other probabilities are negligible.

for the second image,

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .98         			| Vehicles over 3.5 metric tons prohibited 									|
| .15     				| Dangerous curve to the right 										|
| ~.00					| Roundabout mandatory											|
| ~.00	      			| Traffic signals				 				|
| ~.00				    | Speed limit (20km/h)      							|




