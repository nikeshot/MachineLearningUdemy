#Created By Niketan Rane : Dated Nov 3, 2017

#Data Preprocessing 

#Import Libraries 
import numpy as np
import matplotlib as plt
import pandas as pd
from sklearn.preprocessing import Imputer

#read data and initialise input and output data
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]


