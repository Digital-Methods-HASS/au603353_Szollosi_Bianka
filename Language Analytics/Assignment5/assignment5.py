#!/usr/bin/env python
# coding: utf-8

#importing packeges
import os
import sys
sys.path.append(os.path.join(".."))

import pandas as pd
import utils.classifier_utils_a5 as clf

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics
from sklearn import preprocessing

import matplotlib.pyplot as plt
import seaborn as sns

#defining main() function
def main():
    #reading in data set (which I previously merged)
    filename = os.path.join("data", "full_corona.csv")
    data = pd.read_csv(filename, index_col=0)

    #data["Sentiment"].value_counts()

    #creating more balanced data with a random samples
    balanced_data = clf.balance(data, 5000) 

    #creating variables
    texts = balanced_data["OriginalTweet"]
    labels = balanced_data["Sentiment"]

    #splitting the data
    X_train, X_test, y_train, y_test = train_test_split(texts, # texts for the model
                                                        labels, # classification labels
                                                        test_size=0.2,   # create an 80/20 split
                                                        random_state=42) # random state for reproducibility


    #creating a vectorizer
    vectorizer = TfidfVectorizer(ngram_range = (1,2),     #unigrams and bigrams (1 word and 2 word units), tokens that appear in the doc
                                 lowercase =  True,       #making everything lowercase?
                                 max_df = 0.95,           #remove very common words, 
                                 min_df = 0.05,           #remove very rare words
                                 max_features = 1000)     # keep only top 1000 features


    #applying it to training data
    X_train_feats = vectorizer.fit_transform(X_train)
    #applying it to test data
    X_test_feats = vectorizer.transform(X_test)

    #fitting the classifier to the data
    classifier = LogisticRegression(random_state=42, max_iter = 500).fit(X_train_feats, y_train)

    #making predictions with the classifier 
    y_pred = classifier.predict(X_test_feats)

    #evaluating model
    classifier_metrics = metrics.classification_report(y_test, y_pred)
    print(classifier_metrics)

#defining behaviour when called from command line
if __name__=="__main__":
    main()



