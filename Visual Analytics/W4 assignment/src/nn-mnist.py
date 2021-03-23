#!/usr/bin/env python
# coding: utf-8

#importing libraries
import sys,os
sys.path.append(os.path.join(".."))
import argparse
import numpy as np

from utils.neuralnetwork import NeuralNetwork
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets


#defining main() function
def main():
    
    #fetching data, where X is the dataset and y is labels of the data
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)


    #converting data into numpy arrays and data type float, bc otherwise the model won't converge later
    X = np.array(X, dtype=np.float128)
    y = np.array(y, dtype=np.float128)


    #predefining classes
    classes = sorted(set(y))
    nclasses = len(classes)


    # MinMax regularization
    X = (X - X.min())/(X.max() - X.min())


    #splitting data into training and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, #our data
                                                        y, #labels
                                                        random_state=9, #makes it reproducible
                                                        train_size=0.8, #splitting by 80%-20%
                                                        test_size=0.2)


    #converting labels from integers to vectors
    y_train = LabelBinarizer().fit_transform(y_train)
    y_test = LabelBinarizer().fit_transform(y_test)


    #training network (from 784 nodes to 10)
    print("[INFO] training network...")
    nn = NeuralNetwork([X_train.shape[1], 400, 120, 10])
    print("[INFO] {}".format(nn))
    nn.fit(X_train, y_train, epochs=500)


    #evaluating network
    print(["[INFO] evaluating network..."])
    predictions = nn.predict(X_test)
    predictions = predictions.argmax(axis=1)
    print(classification_report(y_test.argmax(axis=1), predictions))
    

#defining behaviour when called from command line
if __name__=="__main__":
    main()



