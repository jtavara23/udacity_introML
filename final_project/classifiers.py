from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


def select_best_features(clf_grid, features_list):
    k_best = clf_grid.best_estimator_.named_steps['selector']

    features_array = np.array(features_list)
    features_array = np.delete(features_array, 0)
    indices = np.argsort(k_best.scores_)[::-1]
    k_features = k_best.get_support().sum()

    features = []
    for i in range(k_features):
        features.append(features_array[indices[i]])

    features = features[::-1]
    scores = k_best.scores_[indices[range(k_features)]][::-1]

    plt.figure(figsize=(15, 9))
    plt.barh(range(k_features), scores)
    plt.yticks(np.arange(0.4, k_features), features)
    plt.title('SelectKBest Feature Importance\'s')
    st.pyplot()


def evaluate_model(grid, X, y, cv=None, x_test=None, y_test=None):
    print("Own evaluation function: ")
    st.write(" Getting test metrics from own model evaluation")
    cv_accuracy = []
    cv_precision = []
    cv_recall = []
    cv_f1 = []

    nested_score = cross_val_score(grid, X=X, y=y, cv=cv, n_jobs=-1)
    st.write("Nested f1 score: {}".format(nested_score.mean()))

    grid.fit(X, y)
    st.write("Best parameters: ", grid.best_params_)

    x_train = X
    y_train = y

    if cv is not None:
        for train_index, test_index in cv.split(X, y):
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

    grid.best_estimator_.fit(x_train, y_train)
    prediction = grid.best_estimator_.predict(x_test)

    cv_accuracy.append(accuracy_score(y_test, prediction))
    cv_precision.append(precision_score(y_test, prediction))
    cv_recall.append(recall_score(y_test, prediction))
    cv_f1.append(f1_score(y_test, prediction))

    acc = np.mean(cv_accuracy)
    precision = np.mean(cv_precision)
    recall = np.mean(cv_recall)
    f1 = np.mean(cv_f1)

    print("Mean Accuracy: {}".format(acc))
    print("Mean Precision: {}".format(precision))
    print("Mean Recall: {}".format(recall))
    print("Mean f1: {}".format(f1))
    print("------------------------------------------------")
    return tuple((acc, precision, recall, f1, "-"))


def classifier_gaussian_nb(feature_scaler, feature_selector, feature_transformer,
                           scaler, selector_k, reducer_n_components, sss=None):
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()

    pipe = Pipeline([
        ('scaler', feature_scaler),
        ('selector', feature_selector),
        ('reducer', feature_transformer),
        ('clf', clf)
    ])

    param_grid = {
        'scaler': scaler,
        'selector__k': selector_k,
        'reducer__n_components': reducer_n_components
    }
    if sss is not None:
        gnb_grid = GridSearchCV(pipe, param_grid, scoring='f1', cv=sss)
    else:
        gnb_grid = GridSearchCV(pipe, param_grid, scoring='f1')

    return gnb_grid




