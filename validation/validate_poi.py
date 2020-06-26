#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset_unix.pkl", "rb"))

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



# (Overfit) POI Identifier
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(features,labels)

pred = clf.score(features,labels)
print(pred)




from sklearn.model_selection import train_test_split
features_train,features_test,labels_train, labels_test  = train_test_split(features,labels,test_size=0.3,random_state=42)

# random_state controls which points go into the training set
# and which are used for testing;
# setting it to 42 means we know exactly which events are in which set

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)

pred = clf.score(features_test,labels_test)
print(labels_train)
print(pred)
