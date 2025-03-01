Prerequisites
-------------
Python 3.x
NumPy
scikit-learn
matplotlib


Explanation
-----------
Load the MNIST Dataset:

The MNIST dataset is fetched using fetch_openml from scikit-learn. It contains 70,000 images of handwritten digits, each 28x28 pixels.
Data Splitting:
--------------
The dataset is split into training (80%) and testing (20%) sets using train_test_split.
Data Standardization:

Features are scaled to have zero mean and unit variance using StandardScaler.
KNN Classifier:
--------------
A KNN classifier is initialized with n_neighbors=3 and trained on the training set.
Prediction:

The classifier is used to predict the labels for the test set.

Evaluation:
-----------
The model's performance is evaluated using accuracy score and classification report.
Visualization:

A function display_digit is defined to display an image of a digit from the dataset. A few test images along with their predicted and actual labels are displayed.
