#!/usr/bin/env python
# coding: utf-8

# In[2]:


# system tools
import os
import sys
sys.path.append(os.path.join(".."))

# pandas, numpy, gensim
import pandas as pd
import numpy as np

# import Ross's classifier utility functions
import utils.classifier_utils as clf

# Machine learning stuff
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics

# tools from tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Embedding, 
                                     Flatten, GlobalMaxPool1D, Conv1D)
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# matplotlib
import matplotlib.pyplot as plt


# In[9]:


#load in data
data_path = os.path.join("..", "data", "Game_of_Thrones_Script.csv")


# In[10]:


got_data = pd.read_csv(data_path)


# In[35]:


sentences = got_data['Sentence'].values
labels = got_data['Season'].values


# In[36]:


# train and test split using sklearn
X_train, X_test, y_train, y_test = train_test_split(sentences, 
                                                    labels, 
                                                    test_size=0.2, 
                                                    random_state=42)


# In[37]:


#vectorizing
vectorizer = CountVectorizer()


# In[38]:


# First we do it for our training data...
X_train_feats = vectorizer.fit_transform(X_train)
#... then we do it for our test data
X_test_feats = vectorizer.transform(X_test)
# We can also create a list of the feature names. 
feature_names = vectorizer.get_feature_names()


# In[39]:


#logistic regression classifier
classifier = LogisticRegression(random_state=42, max_iter =600).fit(X_train_feats, y_train)


# In[40]:


y_pred = classifier.predict(X_test_feats)


# In[41]:


#evaluating performance
classifier_metrics = metrics.classification_report(y_test, y_pred)
print(classifier_metrics)


# In[ ]:




