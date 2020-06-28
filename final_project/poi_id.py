#!/usr/bin/python

import sys
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

st.markdown("# Data Wrangling")
"""
 ## Task 1: Start Analysis on Data
"""
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset_unix.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

df = pd.DataFrame(data_dict)
df_poi = df.transpose()

number_people = number_poi = number_features = 0

def checkingDistribution():
    global number_people , number_poi , number_features
    number_people = len(df_poi.index)
    number_poi = len(df_poi[df_poi['poi']])
    number_features = len(df_poi.columns)
    st.write("Number of people: ", number_people)
    st.write("Number of POI: ", number_poi)
    st.write("Number of features: ", number_features)

checkingDistribution()

financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock',
'director_fees']
#financial_features

email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages',
'from_this_person_to_poi', 'shared_receipt_with_poi']
#email_features

df_poi

#--------------------------------------------------------------------------------------------
"""
 ## Task 2: Remove missing data and outliers
"""
#--------------------------------------------------------------------------------------------

"""### 2.1 Identify and handle missing values")"""

import numpy as np
df_poi.replace("NaN",np.nan, inplace = True) # to replace everything
df_isnull = df_poi.isnull()
missing_data = []

for column in df_isnull.columns.values.tolist():
    missing_val = df_isnull[column].value_counts()
    if(missing_val[0] == 146):
        percentage = 0
        missing_data.append((column, 0, percentage))
    else:
        number_nan = int(missing_val[1])
        percentage =round((number_nan/number_people)*100,2)
        missing_data.append((column, number_nan, percentage))
    #st.write(column, " | Missing features: ",number_nan, " =>",percentage,"%")
    #st.write(missing_val)

#missing_data
#--------------------------------------------------------------------------------------------

### Plot missing data
df_missing_data = pd.DataFrame.from_records(missing_data, columns=["feature", "nan", "percentage"])
df_missing_data.set_index("feature", inplace=True)

missing_value_chart = df_missing_data["nan"].plot(kind='barh', figsize=(15,12))
plt.xlabel("Number of missing values")

n_values = df_missing_data['nan']
"""we have 6 features which more than half of its values are missing."""

possible_non_important_features = []
for ind, perc in enumerate(df_missing_data['percentage']):
    if perc > 50:
        possible_non_important_features.append((missing_data[ind][0] , perc))
        color = 'r'
    else:
        color = 'b'
    plt.annotate(str(perc)+"%",                # text to display
                 xy=(n_values[ind]+2,ind),     # start the text at at point (x,y)
                 rotation=0,
                 va='bottom',                  # want the text to be vertically 'bottom' aligned
                 ha='left',                    # want the text to be horizontally 'left' algned.
                 color=color
                )
st.pyplot()

possible_non_important_features.sort(key= lambda x: x[1], reverse=True)
possible_non_important_features #['deferral_payments','deferred_income','director_fees','loan_advances', 'long_term_incentive', 'restricted_stock_deferred']

for outlier, perc in possible_non_important_features:
    uniq = len(df_poi[outlier].unique())
    if perc > 75:
        uniq = str(uniq)+" ✔"
    st.write("-", outlier, "unique values: ", uniq)

"""> 'load_advances', 'director_fees' and 'restricted_stock_deferred' are removed."""
"""> Those feature have more than 75% of their data missing and only have less than 20 unique values."""
"""> We also remove the email_address feature as it's a string value"""

#One other data point can be eliminated (THE TRAVEL AGENCY IN THE PARK)
#because it does not represent a person, and therefore can’t be a POI, leaving 144 data points.

df_poi.drop(possible_non_important_features[0][0], inplace=True, axis=1)
df_poi.drop(possible_non_important_features[1][0], inplace=True, axis=1)
df_poi.drop(possible_non_important_features[2][0], inplace=True, axis=1)
df_poi.drop("email_address", inplace=True, axis=1)


""" ### Checking distribution"""
checkingDistribution()
financial_features = ['salary', 'deferral_payments', 'total_payments', 'bonus',
                      'deferred_income', 'total_stock_value', 'expenses',
                      'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock']
#financial_features

email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages',
                  'from_this_person_to_poi', 'poi', 'shared_receipt_with_poi']
#email_features

#----------------------------------------------------------------------------------------------------------

"""
### 2.2 Identifying Outliers
"""
st.write("taking into account the first two of the features")
fig = plt.figure(figsize=(10,7)) # create figure
ax0 = fig.add_subplot(1, 2, 1) # add subplot 1 (1 row, 2 columns, first plot)
ax1 = fig.add_subplot(1, 2, 2) # add subplot 2 (1 row, 2 columns, second plot). See tip below

df_poi.plot(kind="scatter",x = 'bonus', y = 'deferral_payments', alpha=0.5, ax=ax0)
ax0.set_title("Before the TOTAL feature removal")
st.write("Finding to whom the huge \"total_payments\" belogns to: ")
with st.echo():
    df_poi['total_payments'].idxmax()
st.write("->",df_poi['total_payments'].idxmax())
st.write("that is not a proper person's name, we will remove it.")

df_poi.drop('TOTAL', inplace = True)
df_poi.plot(kind="scatter", x = 'bonus', y = 'deferral_payments',alpha=0.5, ax=ax1)
ax1.set_title("After the TOTAL feature removal")
st.pyplot(fig)



#--------------------------------------------------------------------------------------------
"""# Exploratory Data Analysis"""
"""
## Task 3: Feature Handling
"""
"""
### 3.1 Verifying correlation among features
Considering feature extraction, high correlated variables usually are useless for machine learning classification. In this case, it's better to use uncorrelated variables as features,
 in the way they are orthogonal to each other and so brings on different information aspects from data.

"""
from scipy import stats
def plot_correlation(type):
    if type == "pearson":
        correlation_df = df_poi[financial_features].corr()
    else:
        correlation_df = df_poi[financial_features].corr(method= lambda x,y : stats.pearsonr(x,y)[1]) - np.eye(len(df_poi[financial_features].columns))
    #correlation_df
    # Drawing a heatmap with the numeric values in each cell
    fig1, ax = plt.subplots(figsize=(15,12))
    fig1.subplots_adjust(top=.945)
    plt.suptitle('Features '+ type +' correlation from the Enron POI dataset', fontsize=14, fontweight='bold')
    #email-addresses is not count
    cbar_kws = {'orientation':"vertical", 'pad':0.025, 'aspect':70}
    import seaborn as sns
    sns.heatmap(correlation_df, annot=True, fmt='.3f', linewidths=.3, ax=ax, cbar_kws=cbar_kws, cmap="YlGnBu");

    st.pyplot()

""" 3.1.1 Verifying Pearson correlation"""

plot_correlation("pearson")
"""
'Deferral_payments' and 'expenses' are not highly correlated among the other financial features.
"""
""" 3.1.2 Verifying P-value correlation """
plot_correlation("p-value")
"""
'Deferral_payments' and 'expenses' have very low confident that the correlation between the financial variables is significant.
"""

df_poi.drop("deferral_payments", inplace=True, axis=1)
df_poi.drop("expenses", inplace=True, axis=1)

""" ### Checking new distribution"""
checkingDistribution()

financial_features = ['salary', 'total_payments','deferred_income', 'bonus','total_stock_value', 'exercised_stock_options',
                      'other', 'long_term_incentive', 'restricted_stock']
email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages',
                  'from_this_person_to_poi', 'shared_receipt_with_poi']
features_list = ['poi'] + financial_features + email_features




#--------------------------------------------------------------------------------------------
"""
### 3.2 Feature Creation
Let's check if new features originally taken from the existing ones can make a difference to the results:

- total_messages = to_messages +  from_messages

- total_messages_with_poi_ratio: 
(from_this_person_to_poi + from_poi_to_this_person + shared_receipt_with_poi)/ total_messages
 
- emails_to_poi_ratio: from_this_person_to_poi / (total_messages)
 
*ratio of total emails to a POI to total emails *
 
- emails_from_poi_ratio: from_poi_to_this_person / (total_messages)

*ratio of total emails from a POI to total emails*
"""

def createNewFeatures():
    df_poi["total_messages"] = df_poi["to_messages"] + df_poi["from_messages"]
    df_poi["total_messages_with_poi_ratio"] = (df_poi['from_this_person_to_poi'] + df_poi['from_poi_to_this_person'] +
                                               df_poi['shared_receipt_with_poi']) / df_poi["total_messages"]
    df_poi['emails_to_poi_ratio'] = df_poi['from_this_person_to_poi'] / df_poi['total_messages']
    df_poi['emails_from_poi_ratio'] = df_poi['from_poi_to_this_person'] / df_poi['total_messages']
    x_feature = "total_messages"
    y_feature = "total_messages_with_poi_ratio"
    ax = df_poi[df_poi['poi'] == False].plot.scatter(x=x_feature, y=y_feature, color='blue', label='non-poi')
    df_poi[df_poi['poi'] == True].plot.scatter(x=x_feature, y=y_feature, color='red', label='poi', ax=ax)
    st.pyplot()

createNewFeatures()





#----------------------------------------------------------------------------------------------------------
"""
### 3.3 Feature Scaling
Some feature values have a large range of values. 
 - The MinMaxScaler adjusts the feature values and scales them so that any patterns can be identified easier.
 - StandardScaler adjust by removing the mean and scaling to unit variance.
"""
from sklearn.preprocessing import MinMaxScaler, StandardScaler
SCALER = [None, StandardScaler(),MinMaxScaler()]

# #(df - df.min()) / (df.max() - df.min())
# scaler = MinMaxScaler()
# df_poi[df_poi.columns] = scaler.fit_transform(df_poi)
# df_poi
#
# """Replace Nan by zeros"""
# df_poi.replace(np.nan, 0., inplace = True) # to replace everything
# df_poi


#----------------------------------------------------------------------------------------------------------
"""
### 3.4 Feature Selection
Decide which features we importarnt to train the ML algorithm

How many features?
 - SelectPercentile
 - SelectKBest
"""
from sklearn.feature_selection import SelectKBest, SelectPercentile
SELECTOR_K = [8, 9, 10, 11, 12, 'all']
#----------------------------------------------------------------------------------------------------------
"""
### 3.5 Feature Transformer
PCA - a linear dimensionality reduction technique that can be utilized for extracting information 
from a high-dimensional space by projecting it into a lower-dimensional sub-space
"""
from sklearn.decomposition import PCA
REDUCER_N_COMPONENTS = [2, 4, 6, 8]



#----------------------------------------------------------------------------------------------------------
"""# 4. Model Training"""
### Load the dictionary containing the dataset
filled_df = df_poi.fillna(value='NaN') # featureFormat expects 'NaN' strings
data_dict = filled_df.to_dict(orient='index')

### Store to my_dataset for easy export below.
my_dataset = data_dict
#my_dataset
#features_list

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
x_features = np.array(features)
y_labels = np.array(labels)
#----------------------------------------------------------------------------------------------------------
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info: http://scikit-learn.org/stable/modules/pipeline.html

#Stratified
from sklearn.model_selection import StratifiedShuffleSplit
# Cross-validation
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
#Non-Stratified
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.25, random_state=42)

evaluation_metrics = []

#Try a variety of classifiers.
#----------------------------------------------------------------------------------------------------------
from classifiers import select_best_features, evaluate_model, classifier_gaussian_nb

""" ## 4.1 Gaussian Naive Bayes"""
name = "gaussian_stratified"
""" ### 4.1.1 Stratified Shuffle Split"""
clf_stratified = classifier_gaussian_nb(StandardScaler(), SelectKBest(), PCA(random_state=42),
                       SCALER, SELECTOR_K, REDUCER_N_COMPONENTS, sss)

evaluation_metrics.append((name,) + evaluate_model(clf_stratified, x_features, y_labels, cv=sss))
evaluation_metrics.append((name,) + test_classifier(clf_stratified.best_estimator_, my_dataset, features_list))
select_best_features(clf_stratified,features_list)
dump_classifier_and_data(clf_stratified, my_dataset, features_list, name)

st.markdown("_________________")

""" ### 4.1.2 Non Stratified Split"""
name = "gaussian_non_Stratified"
clf_non_stratified = classifier_gaussian_nb(StandardScaler(), SelectKBest(), PCA(random_state=42),
                       SCALER, SELECTOR_K, REDUCER_N_COMPONENTS)

evaluation_metrics.append((name,) + evaluate_model(clf_non_stratified, features_train, labels_train, x_test=features_test, y_test=labels_test ))
evaluation_metrics.append((name,) + test_classifier(clf_non_stratified.best_estimator_, my_dataset, features_list))
select_best_features(clf_non_stratified,features_list)
dump_classifier_and_data(clf_non_stratified, my_dataset, features_list, name)

#----------------------------------------------------------------------------------------------------------
""" ## 4.2 SVC """




df_evaluation_metrics = pd.DataFrame.from_records(evaluation_metrics, columns=["clf_name", "accuracy", "precision", "recall", "f1", "f2"])
df_evaluation_metrics.set_index("clf_name", inplace=True)
df_evaluation_metrics

#from sklearn.svm import SVC
#from sklearn.tree import DecisionTreeClassifier
