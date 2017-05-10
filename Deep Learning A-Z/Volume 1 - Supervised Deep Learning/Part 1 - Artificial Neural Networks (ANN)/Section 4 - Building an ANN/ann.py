# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 21:21:21 2017

@author: Brandon
"""

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
# Consider pd.get_dummies()
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values
person = pd.DataFrame([[600, "France", "Male", 40, 3, 60000, 2, 1, 1, 50000]]).values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

categorical_columns = [1,2]
for i in categorical_columns:
    labelencoder_X = LabelEncoder()
    X[:, i] = labelencoder_X.fit_transform(X[:, i])
    person[:, i] = labelencoder_X.transform(person[:, i])
    
onehotencoder = OneHotEncoder(categorical_features = [1]) ## specify columns with categorical features
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] ## get rid of redundant dummy variable

person = onehotencoder.transform(person).toarray()
person = person[:, 1:] ## get rid of redundant dummy variable

"""    
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = 1)
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
"""

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
person = sc.transform(person)
# Part 2 - Making the ANN
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
# Fixed according to documentation
classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'uniform', input_shape = (11,)))

# Fixed according to ipython output
#classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
""" Tutorial
classifier.add(Dense(units = 6, init = 'uniform', activation = 'relu', input_dim = 11))
"""

# Adding the second hidden layer
classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'uniform'))

# Adding the output layer
classifier.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'uniform'))
# Use activation = 'softmax' function for more then 2 output choices?

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# categorical_crossentropy for non binary dependent variable
 
#Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

person_pred = classifier.predict(person)
person_pred = (person_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

correct = 1506 + 215
accuracy = correct / 2000

"""
Predict Subset of Dataset
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000
"""

"""
0, 0 = France
0, 1 = Spain
1, 0 = Germany
0 = Female
1 = Male
"""
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (person_pred > 0.5)