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

"""
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer with dropout
# Add dropout if there is overfitting, signs of overfitting include variance in training in accuracy, noticable in kfold, 
# overfitting is also noticable when the accuracy of the training set doesnt match the test (some tests pass some dont)
# to much dropout can cause underfitting (model isnt learning enough)

# Fixed according to documentation
classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'uniform', input_shape = (11,)))
#classifier.add(Dropout(p = 0.1))
# Fixed according to ipython output
#classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

#Tutorial
#classifier.add(Dense(units = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'uniform'))
#classifier.add(Dropout(p = 0.1)) 

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

new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (person_pred > 0.5)
"""

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



0, 0 = France
0, 1 = Spain
1, 0 = Germany
0 = Female
1 = Male
"""


# Part 4 - Evaulating, Improving and Tuning the ANN
"""
# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 8, activation = 'relu', kernel_initializer = 'uniform', input_shape = (11,)))
    #classifier.add(Dropout(rate = 0.1)) 
    classifier.add(Dense(units = 8, activation = 'relu', kernel_initializer = 'uniform'))
    #classifier.add(Dropout(rate = 0.1)) 
    classifier.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'uniform'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
mean = accuracies.mean()
variance = accuracies.std()
# Improving the ANN
# Dropout regularization to reduce overfitting if needed

# Tuning the ANN
# Two type of parameters: the ones that are learned (weights)and fixed hyper parameters (number of epochs, 
# batch size, optimizer, number of neurons)
# Idea is to find parameters that lead to best accuracy
"""

# how to run multiple models in a row
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
def build_classifier(optimizer, neuron_count):
    classifier = Sequential() 
    classifier.add(Dense(units = neuron_count, activation = 'relu', kernel_initializer = 'uniform', input_shape = (11,)))
    classifier.add(Dense(units = neuron_count, activation = 'relu', kernel_initializer = 'uniform'))
    classifier.add(Dense(units = neuron_count, activation = 'relu', kernel_initializer = 'uniform'))
    classifier.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'uniform'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
#Add grid search to find best tuning
#create dictionary of parameters
parameters = {'batch_size': [32],
              'epochs': [200],
              'optimizer': ['rmsprop'],
              'neuron_count': [15]}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
