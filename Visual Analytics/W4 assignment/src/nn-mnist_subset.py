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
    
    #loading in the 8x8 version of the dataset, as the whole dataset took too much time to run
    digits = datasets.load_digits()

    #Converting to floats
    data = digits.data.astype("float")

    #MinMax regularization
    data = (data - data.min())/(data.max() - data.min())

    #splitting data
    X_train, X_test, y_train, y_test = train_test_split(data, 
                                                      digits.target, 
                                                      test_size=0.2)

    #converting labels from integers to vectors
    y_train = LabelBinarizer().fit_transform(y_train)
    y_test = LabelBinarizer().fit_transform(y_test)


    #training network
    print("[INFO] training network...")
    nn = NeuralNetwork([X_train.shape[1], 32, 16, 10])
    #here instead of putting in hte number of input, we just put the data in

    print("[INFO] {}".format(nn))
    nn.fit(X_train, y_train, epochs=1000)


    #evaluating network
    print(["[INFO] evaluating network..."])
    predictions = nn.predict(X_test)
    predictions = predictions.argmax(axis=1)
    print(classification_report(y_test.argmax(axis=1), predictions))
    

#defining behaviour when called from command line
if __name__=="__main__":
    main()





