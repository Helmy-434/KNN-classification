import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.matrixlib.defmatrix import matrix
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix




def main():
    test = pd.read_csv("test.csv")
    train = pd.read_csv("train.csv")
    x_train = train.drop(columns=["class",train.columns[0]])
    y_train = train["class"]
    x_test = test.drop(columns=["class",train.columns[0]])
    y_test = test["class"]
    model=model_training(x_train,y_train)
    matrix,accuracy = model_testing(model,x_test,y_test)
    print(f"Model Accuracy: {accuracy:.4f}")  
    print("\nConfusion Matrix:")
    print(matrix)


def model_training (x_train,y_train):
    knn = KNeighborsClassifier()
    model = GridSearchCV(
        knn,
        param_grid={
        'n_neighbors':range(1,30),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    ,cv=5 # splits el data into 5 parts
    ,n_jobs=-1 # Use all CPU cores
    ,verbose=1   # prints info about el progress
    )
    model.fit(x_train,y_train)
    return model.best_estimator_

def model_testing(model,x_test,y_test):
    predictions=model.predict(x_test)
    accuracy=accuracy_score(y_test,predictions)
    matrix=confusion_matrix(y_test,predictions)
    return matrix,accuracy


main()