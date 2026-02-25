import streamlit as st
import pandas as pd
import pickle
import numpy as np

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Tourism Experience Analytics", layout="wide")

st.title("🌍 Tourism Experience Analytics System")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("../data/processed/final_cleaned_dataset.csv")

df = load_data()

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():
    reg_model = pickle.load(open("../models/regression_model.pkl", "rb"))
    clf_model = pickle.load(open("../models/classification_model.pkl", "rb"))
    return reg_model, clf_model

reg_model, clf_model = load_models()

# -----------------------------
# Sidebar Navigation
# -----------------------------
menu = st.sidebar.selectbox("Select Feature",
                            ["Visit Mode Prediction",
                             "Rating Prediction",
                             "Attraction Recommendation"])

# =====================================================
# 1️⃣ VISIT MODE PREDICTION
# =====================================================
if menu == "Visit Mode Prediction":

    st.header("🔍 Predict Visit Mode")

    col1, col2 = st.columns(2)

    with col1:
        continent = st.selectbox("Continent", df['Continent'].unique())
        region = st.selectbox("Region", df['Region'].unique())
        country = st.selectbox("Country", df['Country'].unique())

    with col2:
        city = st.selectbox("City", df['CityName'].unique())
        month = st.selectbox("Visit Month", list(range(1, 13)))
        attraction_type = st.selectbox("Attraction Type", df['AttractionType'].unique())

    if st.button("Predict Visit Mode"):

        input_data = pd.DataFrame({
            'Continent': [continent],
            'Region': [region],
            'Country': [country],
            'CityName': [city],
            'VisitMonth': [month],
            'AttractionType': [attraction_type]
        })

        input_data = pd.get_dummies(input_data).reindex(columns=clf_model.feature_names_in_, fill_value=0)

        prediction = clf_model.predict(input_data)[0]

        st.success(f"Predicted Visit Mode: {prediction}")

# =====================================================
# 2️⃣ RATING PREDICTION
# =====================================================
elif menu == "Rating Prediction":

    st.header("⭐ Predict Attraction Rating")

    col1, col2 = st.columns(2)

    with col1:
        continent = st.selectbox("Continent ", df['Continent'].unique())
        region = st.selectbox("Region ", df['Region'].unique())
        country = st.selectbox("Country ", df['Country'].unique())
        visit_mode = st.selectbox("Visit Mode", df['VisitMode'].unique())

    with col2:
        city = st.selectbox("City ", df['CityName'].unique())
        month = st.selectbox("Visit Month ", list(range(1, 13)))
        attraction_type = st.selectbox("Attraction Type ", df['AttractionType'].unique())

    if st.button("Predict Rating"):

        input_data = pd.DataFrame({
            'Continent': [continent],
            'Region': [region],
            'Country': [country],
            'CityName': [city],
            'VisitMonth': [month],
            'VisitMode': [visit_mode],
            'AttractionType': [attraction_type]
        })

        input_data = pd.get_dummies(input_data).reindex(columns=reg_model.feature_names_in_, fill_value=0)

        prediction = reg_model.predict(input_data)[0]

        st.success(f"Predicted Rating: ⭐ {round(prediction,2)}")

# =====================================================
# 3️⃣ RECOMMENDATION SYSTEM
# =====================================================
elif menu == "Attraction Recommendation":

    st.header("🎯 Get Personalized Attraction Recommendations")

    attraction_type = st.selectbox("Select Attraction Type", df['AttractionType'].unique())
    country = st.selectbox("Select Country", df['Country'].unique())

    if st.button("Recommend"):

        filtered = df[(df['AttractionType'] == attraction_type) &
                      (df['Country'] == country)]

        top_places = filtered.groupby('Attraction')['Rating'].mean().sort_values(ascending=False).head(5)

        st.subheader("Top Recommended Attractions:")
        for place in top_places.index:
            st.write(f"• {place}")