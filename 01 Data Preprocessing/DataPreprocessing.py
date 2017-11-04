#Created By Niketan Rane : Dated Nov 3, 2017

#Data Preprocessing 

#Import Libraries 
import numpy as np
import matplotlib as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#read data and initialise input and output data
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values    #need to convert the series to ndarray in order to use imputer. 
y = dataset.iloc[:, -1].values

#Handle Missing Data
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) #Imputer takes input of ndarray type
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

'''Note: 'fit' calculates the standard deviation and mean on the given data and stores internally. 
transform then uses these values depending on required 'strategy' we need '''

#To Encode Categorical data 
''' LabelEncoder gives labels to all different types. But generally need to avoid this because it gives
numerical values to observations and one observation(2) > other(0) which should not be the case. Instead use OneHotEncoder 
But for y(output) you can use label encoder  '''
''' Note that input to onehotencoder is matrix of integers. So you need to first convert the data using labelencoder. '''
labelencoder_X = LabelEncoder()
labelencoder_X = labelencoder_X.fit(X[:, 0])
X[:, 0] = labelencoder_X.transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
onehotencoder = onehotencoder.fit(X)
X = onehotencoder.transform(X).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Feature Scaling
''' Since most of the mahcine learning models are based on Euclidean distance between two points, we need make sure
that all columns are scaled/normalized so that any one column cannot dominate the results. '''
standardscaler_X = StandardScaler()
X = standardscaler_X.fit_transform(X)
''' Note: You don't need to scale the dummy variables[An input field is OneHotEncoded into many columns]. Depends on if
you want to keep your model intuitive as possible. '''

#Split the dataset into training and testing data. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
