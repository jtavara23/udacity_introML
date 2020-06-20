#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot as plt
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
import streamlit as st

### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset_unix.pkl", "rb") )
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

data
### your code below

for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
#plt.show()

st.pyplot()

import pandas as pd

df = pd.DataFrame(data_dict)

df = df.transpose()
#df.loc['salary', : ] = pd.to_numeric(df.loc['salary',:] , errors = 'coerce') #n invalid parsing will be set as NaN.
#df.loc['bonus', : ] = pd.to_numeric(df.loc['bonus',:] , errors = 'coerce')

df.loc[:, 'salary'] = pd.to_numeric(df.loc[:, 'salary'] , errors = 'coerce') #n invalid parsing will be set as NaN.
df.loc[: ,'bonus' ] = pd.to_numeric(df.loc[: ,'bonus'] , errors = 'coerce')

sorted_Val = df.sort_values("salary", ascending=False)
x = sorted_Val.head(5)["salary"]
x


st.title("Removing TOTAl index")
data_dict.pop('TOTAL', 0)

data = featureFormat(data_dict, features)

plt.scatter(data[:,0], data[:,1])
plt.xlabel("salary")
plt.ylabel("bonus")
st.pyplot()

