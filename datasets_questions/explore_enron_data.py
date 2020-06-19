#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""
import streamlit as st
import pickle
import pandas as pd

enron_data = pickle.load(open("../final_project/final_project_dataset_unix.pkl", "rb"))

st.title("Enron Data")
df = pd.DataFrame(enron_data)
st.write("Original size:", df.shape)
st.title("Transposing")
dft = df.transpose()
st.dataframe(dft)

dft.index
dft.columns

st.write("number of POI", dft[dft['poi'] == 1].shape)

dft[( dft["total_payments"] == "NaN") & (dft["poi"] == 1) ].shape
dft[dft["poi"] == 1].shape
