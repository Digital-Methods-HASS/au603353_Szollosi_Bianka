## Assignment 5

__Motivation and dataset__

For this assignment I chose the Coronavirus Tweets NLP- text classification dataset from Kaggle (https://www.kaggle.com/datatattle/covid-19-nlp-text-classification) to see if it was possible to predict sentiment (Extremely Positive, Extremely Negative, Positive, Negative, Neutral) with supervised machine learning from corona virus tagged tweets from across the world.

__Practical info:__

This repository contains:

-a .py script that can be run from the command line

-a requirements.txt file listing the required Python libraries for being able to run the script

-a bash script for setting up a virtual enviroment for running the script (recommended)

In order to run this script, open your terminal and

1. Clone this repo with 'git clone'
2. navigate to the apropriate folder (Assignment5) and activate a virtual environment (recommended) by: 
      
      -python3 -m venv ass5
      
     -source ass5/bin/activate
     
4. Install the necessary libraries if needed (a possibility: pip install *library name*)
5. run the script by python assignment5.py
6. deactivate virtual environment by deactivate

-preprocessing of the data:
The original dataset was already split into train and test sets, which came in two csv files. I merged these back together in order to be able to split the data differently if I wanted to. I included the merged dataset in the data folder.

__Conclusion__

I wanted to study if it was possible to predict sentiment from corona virus tagged tweets. However, after playing around with some parameters, the most successful model I trained only had a weighted accuracy of 0.34, which implies that either the model was not trained according to the appropriate parameters or that it is not possible to predict sentiment from the content of the tweets alone. It is important to mention that upon inspecting the top 20 most important features - according to which the model classified the sentiment of the tweets - these were rather uninformative (eg. containing filling word such as 'of', 'the' or words with no actual meaning 'https'). I tried to correct for this problem by removing very common words, however it did not seem to help much.

PS: Any constructional feedback and tips on how I could improve the accuracy of the model are most welcome :)
