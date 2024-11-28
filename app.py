# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

# Caching to improve performance
@st.cache_resource
def load_model():
    try:
        return joblib.load('mlp_model.pkl')
    except FileNotFoundError:
        st.error("Model file 'mlp_model.pkl' not found.")
        st.stop()

@st.cache_resource
def load_scaler():
    try:
        return joblib.load('scaler.pkl')
    except FileNotFoundError:
        st.error("Scaler file 'scaler.pkl' not found.")
        st.stop()

@st.cache_resource
def load_selected_features():
    try:
        return joblib.load('selected_features.pkl')
    except FileNotFoundError:
        st.error("Selected features file 'selected_features.pkl' not found.")
        st.stop()

@st.cache_data
def load_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    return X

# Load necessary components
model = load_model()
scaler = load_scaler()
selected_features = load_selected_features()
X = load_data()

# App Title and Description
st.title("Breast Cancer Prediction App")
st.write("""
This app predicts whether a breast tumor is **Malignant** or **Benign** based on various features.
Adjust the parameters in the sidebar to make predictions.
""")

# Function to get user input from sliders
def get_user_input():
    st.sidebar.header('User Input Parameters')

    user_data = {}
    for feature in selected_features:
        min_val = float(X[feature].min())
        max_val = float(X[feature].max())
        mean_val = float(X[feature].mean())
        user_data[feature] = st.sidebar.slider(
            feature, min_val, max_val, mean_val
        )

    features = pd.DataFrame(user_data, index=[0])
    return features

# Get user input
user_input = get_user_input()

# Preprocess user input
try:
    user_input_scaled = scaler.transform(user_input)
except Exception as e:
    st.error(f"Error during scaling: {e}")
    st.stop()

# Make prediction
try:
    prediction = model.predict(user_input_scaled)
    prediction_proba = model.predict_proba(user_input_scaled)
except Exception as e:
    st.error(f"Error during prediction: {e}")
    st.stop()

# Display results
st.subheader('Prediction')
cancer_types = np.array(['Malignant', 'Benign'])
st.write(cancer_types[prediction][0])

st.subheader('Prediction Probability')
fig, ax = plt.subplots()
ax.bar(cancer_types, prediction_proba[0])
ax.set_ylim([0, 1])
st.pyplot(fig)
