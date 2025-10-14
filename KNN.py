import numpy as np
import pandas as pd



train_data=pd.read_csv("diabetes_train.csv")
y_train_data=train_data.iloc[:,-1] #assign the last column as the y label
x_train_data=train_data.iloc[:, :-1] #assign the rest as the features
test_data=pd.read_csv("diabetes_test.csv")
y_test_data=test_data.iloc[:,-1]
x_test_data=test_data.iloc[:, :-1]
