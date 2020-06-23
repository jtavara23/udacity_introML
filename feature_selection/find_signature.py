#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "rb"))
authors = pickle.load( open(authors_file, "rb") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import model_selection
features_train, features_test, labels_train, labels_test = model_selection.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')

features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
#features_train = features_train[:150].toarray()
#labels_train   = labels_train[:150]

import streamlit as st
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
clf.n_features_

# Accuracy of Your Overfit Decision Tree

score = clf.score(features_test, labels_test)
score



### Top Features
### Use TfIdf to Get the Most Important Word
"""
    Only printing out the feature importance if it’s above some threshold (say, 0.2--remember,
    if all words were equally important, each one would give an importance of far less than 0.01)
"""
features = zip( range(len(clf.feature_importances_))  , clf.feature_importances_)
for number, feature in features:
   if (feature > 0.001):
       st.write(number, feature, vectorizer.get_feature_names()[number])

#top_features = [(number, feature, vectorizer.get_feature_names()[number]) for number, feature in
#                zip(range(len(clf.feature_importances_)), clf.feature_importances_) if feature > 0.2]

