#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC


#features_train = features_train[:int(len(features_train)/100)]
#labels_train = labels_train[:int(len(labels_train)/100)]

classif = SVC(C=10000,kernel='rbf')
t0 = time()
classif.fit(features_train, labels_train)
print("training time: ", round(time()-t0, 3), "s")

#pred = classif.score(features_test,labels_test)
#print(f"accuracy: {pred}")
#########################################################

#res = classif.predict([features_test[10],features_test[26], features_test[50]])
res = classif.predict(features_test).tolist().count(1)
print(f"resulots: {res}, total{len(features_test)}")
