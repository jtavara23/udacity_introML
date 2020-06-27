from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def evaluate_model(grid, X, y, cv):
    nested_score = cross_val_score(grid, X=X, y=y, cv=cv, n_jobs=-1)
    print("Nested f1 score: {}".format(nested_score.mean()))

    grid.fit(X, y)
    print("Best parameters: {}".format(grid.best_params_))

    cv_accuracy = []
    cv_precision = []
    cv_recall = []
    cv_f1 = []
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        grid.best_estimator_.fit(X_train, y_train)
        prediction = grid.best_estimator_.predict(X_test)

        cv_accuracy.append(accuracy_score(y_test, prediction))
        cv_precision.append(precision_score(y_test, prediction))
        cv_recall.append(recall_score(y_test, prediction))
        cv_f1.append(f1_score(y_test, prediction))

    print("Mean Accuracy: {}".format(np.mean(cv_accuracy)))
    print("Mean Precision: {}".format(np.mean(cv_precision)))
    print("Mean Recall: {}".format(np.mean(cv_recall)))
    print("Mean f1: {}".format(np.mean(cv_f1)))





