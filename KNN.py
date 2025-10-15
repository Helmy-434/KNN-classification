import numpy as np
import pandas as pd

from Classification import classification

if __name__ == '__main__':

    train_data=pd.read_csv("diabetes_train.csv")
    test_data = pd.read_csv("diabetes_test.csv")

    y_train_data=train_data.iloc[:,-1] #assign the last column as the y label
    x_train_data=train_data.iloc[:, :-1].apply(pd.to_numeric, errors='coerce') #convert all values to numeric

    y_test_data=test_data.iloc[:,-1]
    x_test_data=test_data.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')

    x_train = x_train_data.to_numpy(dtype=float) # change to numpy array
    y_train = y_train_data.to_numpy()

    x_test = x_test_data.to_numpy(dtype=float)
    y_test = y_test_data.to_numpy()

    k = int(input("Enter K: "))
    classification(x_train,y_train,x_test,y_test,k)