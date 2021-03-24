#!/usr/bin/env python
# coding: utf-8

#importing libraries
import os
import sys
sys.path.append(os.path.join(".."))

import numpy as np
import utils.classifier_utils as clf_util

from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import argparse

#defining main() function
def main():
    #argument parser
    ap = argparse.ArgumentParser()
    
    #command line arguments to define
    ap.add_argument("-o", "--outfile", required=True, help="Output filename")
    
    #parse arguments
    args = vars(ap.parse_args())
    
    #creating variables that can be defined in command line
    out_file_name = args["outfile"]
                    
    #creating a folder called out, if it doesn't exist (to save the output files here)
    out = os.path.join("..", "out")
    if not os.path.exists(out):
        os.mkdir(out)
    
    #fetching data, where X is the dataset and y is labels of the data
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)


    #converting data into numpy arrays
    X = np.array(X)
    y = np.array(y)


    #predefining classes
    classes = sorted(set(y))
    nclasses = len(classes)


    #splitting data into training and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, #our data
                                                        y, #labels
                                                        random_state=9, #makes it reproducible
                                                        test_size= 0.2)

    #scaling the features by deviding by 255, so the scale features are between 0-1
    X_train_scaled = X_train/255.0
    X_test_scaled = X_test/255.0

    #defining logistic regression model
    log_reg_model = LogisticRegression(penalty='none',
                             tol=0.1,
                             solver='saga',
                             multi_class='multinomial').fit(X_train_scaled, y_train) #fitting model to training data


    #calculating predicitins with the trained model on the scaled test data
    y_pred = log_reg_model.predict(X_test_scaled)


    #showing confusion matrix
    confusion_matrix = metrics.classification_report(y_test, y_pred)
    print(confusion_matrix)

    #saving the output as a txt file
    outpath = os.path.join("..", "out", out_file_name)
    file = open(outpath, "w")
    file.write(confusion_matrix)
    file.close()

#defining behaviour when called from command line
if __name__=="__main__":
    main()




