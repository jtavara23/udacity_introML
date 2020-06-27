#!/usr/bin/python

import sys
import pickle
import streamlit as st
import pandas as pd

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

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

number_people = len(df_poi.index)
number_poi = len(df_poi[df_poi['poi']])
st.write("Number of people: ",number_people)
st.write("Number of POI: ", number_poi)
st.write("Number of features: ",len(df_poi.columns))

financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock',
'director_fees']
#financial_features

email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages',
'from_this_person_to_poi', 'shared_receipt_with_poi']
#email_features


st.write(df_poi)

#--------------------------------------------------------------------------------------------
"""
 ## Task 2: Remove outliers
"""
st.markdown("### Identify and handle missing values")

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
import matplotlib.pyplot as plt

df_missing_data = pd.DataFrame.from_records(missing_data, columns=["feature", "nan", "percentage"])
df_missing_data.set_index("feature", inplace=True)

missing_value_chart = df_missing_data["nan"].plot(kind='barh', figsize=(15,12))
plt.xlabel("Number of missing values")

n_values = df_missing_data['nan']
"""we have 6 features which more than half of its values are missing."""

possible_outliers = []
for ind, perc in enumerate(df_missing_data['percentage']):
    if perc > 50:
        possible_outliers.append((missing_data[ind][0] , perc))
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

#--------------------------------------------------------------------------------------------
"""
### Identifying Outliers
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

possible_outliers.sort(key= lambda x: x[1], reverse=True)
possible_outliers #['deferral_payments','deferred_income','director_fees','loan_advances', 'long_term_incentive', 'restricted_stock_deferred']

for outlier, perc in possible_outliers:
    uniq = len(df_poi[outlier].unique())
    if perc > 75:
        uniq = str(uniq)+" ✔"
    st.write("-", outlier, "unique values: ", uniq)

st.markdown("> load_advances, director_fees and restricted_stock_deferred are removed. Those feature have more than 75% of their data missing and only have less than 20 unique values.")

#One other data point can be eliminated (THE TRAVEL AGENCY IN THE PARK)
#because it does not represent a person, and therefore can’t be a POI, leaving 144 data points.

df_poi.drop(possible_outliers[0][0], inplace=True, axis=1)
df_poi.drop(possible_outliers[1][0], inplace=True, axis=1)
df_poi.drop(possible_outliers[2][0], inplace=True, axis=1)
""" ### Checking distribution"""
number_people = len(df_poi.index)
number_poi = len(df_poi[df_poi['poi']])
number_features = len(df_poi.columns)
st.write("Number of people: ",number_people)
st.write("Number of POI: ", number_poi)
st.write("Number of features: ",number_features)

financial_features = ['salary', 'deferral_payments', 'total_payments', 'bonus',
                      'deferred_income', 'total_stock_value', 'expenses',
                      'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock']
#financial_features

email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages',
                  'from_this_person_to_poi', 'poi', 'shared_receipt_with_poi']
#email_features





#--------------------------------------------------------------------------------------------
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
Deferral_payments and expenses are not highly correlated among the other financial features.
"""
""" 3.1.2 Verifying P-value correlation """
plot_correlation("p-value")
"""
Deferral_payments and expenses have very low confident that the correlation between the financial variables is significant.
"""

df_poi.drop("deferral_payments", inplace=True, axis=1)
df_poi.drop("expenses", inplace=True, axis=1)

""" ### Checking new distribution"""
number_people = len(df_poi.index)
number_poi = len(df_poi[df_poi['poi']])
number_features = len(df_poi.columns)
st.write("Number of people: ",number_people)
st.write("Number of POI: ", number_poi)
st.write("Number of features: ",number_features)

financial_features = ['salary', 'total_payments','deferred_income', 'bonus','total_stock_value', 'exercised_stock_options',
                      'other', 'long_term_incentive', 'restricted_stock']

email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages',
                  'from_this_person_to_poi', 'shared_receipt_with_poi'] # we remove ("email_address") as it's a string

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

"""



#----------------------------------------------------------------------------------------------------------
"""
### 3.4 Feature Selection
Decide which features we importarnt to train the ML algorithm

How many features?
 - SelectPercentile
 - SelectKBest
"""



#----------------------------------------------------------------------------------------------------------

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

#dump_classifier_and_data(clf, my_dataset, features_list)
#st.write("Finishing dumping files.")