# Model-for--Classifying--Handwritten-Digits

# Title 
Model for Classifying Handwritten Digits


# Description
The purpose of this project is to build a model that classifies handwritten digits, given the handwritten images using the MNIST dataset. The dataset is a set of 70.000 small images of digits handwritten by high school students and employees of the US Census Bureau. Each image is labeled with the digit it represents. 

The Modeling phase was split into two parts. In the first part, the problem was simplified, and a model was trained to predict one digit – for example the number 5. This “5-detector” is an example of a binary classifier, capable of distinguishing between two classes, 5 and not-5. In the second part, explore multiclass classifiers was explored to predicts all classes. The following figure shows a few images from the MNIST dataset to give a feel for the complexity of the classification task. 
 
![image](https://user-images.githubusercontent.com/25030435/168308265-e57cfc98-2b45-454e-9233-ae4cad8ae7f8.png)


The CRISP-DM framework was used, and the project was split according to the 6 phases of the framework:  
1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling 
5. Evaluation
6. Deployment

The problem is a supervised learning problem since each image is labeled with the digit it represents. It is also a classification problem since the task is to classify digits. In case of predicting the classes “5” and “not 5”, the problem is binary classification problem. In case of predicting all classes, the problem is a multiclass classification problem. 
In case of a binary classification problem, the main goal is to train a model that identifies the digit 5. This class is referred to as the positive class, while the not-5 class is referred to as the negative class.

The four fundamental performance measures for classifiers were used to evaluate the model, these are Accuracy, Confusion Matrix, Precision and Recall, and F1 Score. The F1 Score is the Harmonic Mean between Precision and Recall.

# Project Implementation
The Project was implemented on KNIME, the Konstanz Information Miner. KNIME is free and open-source data analytics, reporting and integration platform. It is easy to learn and offers a platform for drag-and-drop analytics, machine learning and statistics; no code required! 

# Installation
To install KNIME, please go to: 
https://www.knime.com/downloads/download-knime 

# Modeling
# Training and Evaluating on the Training Set
The problem was framed, and the data was explored. The data is already split into training and testing, and it does not contain any missing values or categorical features. The data was prepared for modelling and the preparation output showed that 9% of the labels correspond to the digit 5, while the remaining 91% are not 5. 
## Training Logistic Regression Model for Binary Classifier
In the first part of the modeling phase, Binary Logistic Regression classifier was trained to predict if an image corresponds to the digit 5 or not and evaluated using Cross-Validation.

Logistic regression is a classification algorithm used to assign observations to a discrete set of classes. Unlike linear regression which outputs continuous number values, logistic regression transforms its output using the logistic sigmoid function to return a probability value which can then be mapped to two or more discrete classes. The prediction function returns a probability (p) score between 0 and 1. In order to map this to a discrete class, we select a threshold value or tipping point above which we will classify values into class 1 and below which we classify values into class 2. 
P>=0.5, class = 1
P<0.5, class = 0
The threshold was set to 0.5 .

Previously, we saw that approximately 91% of the binary labels belong to the class not 5. Thus, if the accuracy score is 91%, then it’s known that the model did nothing, i.e, it always predicted the class not 5. 
From the Binary Classifier output, the accuracy is approximately 97%, thus the model actually did something. Precision is 89.7%, recall is 76% and F1 score 82.3%. Thus, we have a higher precision than recall.
The threshold was increased to 0.7 and make manual predictions based on the probabilities of the positive class and this new threshold.
Previously we obtained a recall and precision of 76% and 89.7%, respectively. That is, precision was higher than recall. Now the precision and recall are 93.7% and 67.5%, respectively. That is, precision is still higher than recall. This confirms that raising the threshold decreases the recall further.
Now how do we decide which threshold to use? It would be nice to have a way of knowing how recall and precision change with different thresholds. One way to know is using Classification Threshold Analysis node in Knime.

# Training Logistic Regression Model for Multiple Classes
Unlike binary classification, there are no positive or negative classes. The overall accuracy obtained was 91.54%, and the precision, recall and F1 scores for all classes were also obtained.

Macro F1: 
Macro averaging is perhaps the most straightforward amongst the numerous averaging methods. The macro-averaged F1 score (or macro F1 score) is computed by taking the mean of all the per-class F1 scores. This method treats all classes equally. The macro F1 score is 0.914 or 91.4%.

Micro F1: 
Micro averaging computes a global average F1 score by counting the sums of the True Positives (TP), False Negatives (FN), and False Positives (FP). The macro F1 score is 0.915 or 91.5%.

Weighted Average F1: 
The last one is weighted-average F1 score. Unlike Macro F1, it takes the mean of all per-class F1 scores while considering each class’s support. Support refers to the number of actual occurrences of the class in the dataset. The macro F1 score is 9.156

Which average should you choose?
In general, if you are working with an imbalanced dataset where all classes are equally important, using the macro average would be a good choice as it treats all classes equally.
It means that for our example involving the classification of the digits 0-9, we would use the macro-F1 score.
If you have an imbalanced dataset but want to assign greater contribution to classes with more examples in the dataset, then the weighted average is preferred. This is because, in weighted averaging, the contribution of each class to the F1 average is weighted by its size.
Suppose you have a balanced dataset and want an easily understandable metric for overall performance regardless of the class. In that case, you can go with accuracy, which is essentially our micro F1 score.

# Model Evaluation
At the evaluation phase, I evaluated the best model on the test set. Macro F1 Score is 85.2%
 
# Deployment 
Below is the Project Workflow

![Workflow](https://user-images.githubusercontent.com/25030435/168309032-a99fc630-47cb-4a27-9da2-235f965a3b25.jpg)
