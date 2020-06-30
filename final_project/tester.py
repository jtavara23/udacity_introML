#!/usr/bin/pickle

""" a basic script for importing student's POI identifier,
    and checking the results that they get from it 
 
    requires that the algorithm, dataset, and features list
    be written to my_classifier.pkl, my_dataset.pkl, and
    my_feature_list.pkl, respectively

    that process should happen at the end of poi_id.py
"""

import pickle
import sys
import streamlit as st
from sklearn.model_selection import StratifiedShuffleSplit
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from classifiers import select_best_features, evaluate_model

PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\nPrecision: {:>0.{display_precision}f}\n\
Recall: {:>0.{display_precision}f}\nF1: {:>0.{display_precision}f}\nF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\nTrue positives: {:4d}\nFalse positives: {:4d}\
\nFalse negatives: {:4d}\nTrue negatives: {:4d}"

def test_classifier(clf, dataset, feature_list, folds=1000):
    st.write("Getting test metrics from udacity tester function")
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(n_splits=folds, random_state=42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv.split(features, labels):
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)

        from sklearn.metrics import classification_report
        target_names = ['Not PoI', 'PoI']
        print(clf)
        print(classification_report(labels_test, predictions, target_names=target_names))

        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print("Warning: Found a predicted label not == 0 or 1.")
                print("All predictions should take value 0 or 1.")
                print( "Evaluating performance for processed predictions:")
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        st.write(clf)
        print (PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5))
        print (RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives))
        print ("___________________________________________________________________")
        return tuple((accuracy, precision, recall, f1, round(f2,2)))
    except:
        print( "Got a divide by zero when trying out:", clf)
        print( "Precision or recall may be undefined due to a lack of true positive predicitons.")


def testing_on_stratified_data(clf_stratified, name, evaluation_metrics, x_features, y_labels,
                               my_dataset, features_list, sss, dump=False):

    evaluation_metrics.append((name,) + evaluate_model(clf_stratified, x_features, y_labels, cv=sss))
    evaluation_metrics.append((name,) + test_classifier(clf_stratified.best_estimator_, my_dataset, features_list))
    feautures = select_best_features(clf_stratified, features_list)
    if dump:
        dump_classifier_and_data(clf_stratified, my_dataset, feautures, name)

def testing_on_non_stratified_data(clf_non_stratified, name, evaluation_metrics, my_dataset, features_list, dump=False):
    from poi_id import features_train, labels_train, features_test, labels_test
    evaluation_metrics.append((name,) + evaluate_model(clf_non_stratified, features_train, labels_train,
                                                       x_test=features_test, y_test=labels_test))
    evaluation_metrics.append((name,) + test_classifier(clf_non_stratified.best_estimator_, my_dataset, features_list))
    feautures = select_best_features(clf_non_stratified, features_list)
    if dump:
        dump_classifier_and_data(clf_non_stratified, my_dataset, feautures, name)


CLF_PICKLE_FILENAME = "my_classifier_"
DATASET_PICKLE_FILENAME = "my_dataset_"
FEATURE_LIST_FILENAME = "my_feature_list_"

def dump_classifier_and_data(clf, dataset, feature_list, name):
    with open(CLF_PICKLE_FILENAME+name+'.pkl', "wb") as clf_outfile:
        pickle.dump(clf, clf_outfile)
    with open(DATASET_PICKLE_FILENAME+name+'.pkl', "wb") as dataset_outfile:
        pickle.dump(dataset, dataset_outfile)
    with open(FEATURE_LIST_FILENAME+name+'.pkl', "wb") as featurelist_outfile:
        pickle.dump(feature_list, featurelist_outfile)

def load_classifier_and_data():
    with open(CLF_PICKLE_FILENAME, "r") as clf_infile:
        clf = pickle.load(clf_infile)
    with open(DATASET_PICKLE_FILENAME, "r") as dataset_infile:
        dataset = pickle.load(dataset_infile)
    with open(FEATURE_LIST_FILENAME, "r") as featurelist_infile:
        feature_list = pickle.load(featurelist_infile)
    return clf, dataset, feature_list

def main():
    ### load up student's classifier, dataset, and feature_list
    clf, dataset, feature_list = load_classifier_and_data()
    ### Run testing script
    test_classifier(clf, dataset, feature_list)

if __name__ == '__main__':
    main()
