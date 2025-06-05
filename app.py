# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load saved files
similarity = joblib.load("similarity_matrix.pkl")
disease_list = joblib.load("disease_list.pkl")
workout_list = joblib.load("workout_list.pkl")

# Title
st.title("🏋️ Workout Recommendation System")
st.write("Get personalized workout suggestions based on your health condition.")

# Dropdown for selecting disease
selected_disease = st.selectbox("Select your disease:", disease_list)

# On click recommend
if st.button("Recommend Workouts"):
    idx = disease_list.index(selected_disease)
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Collect top 3 similar diseases and their workouts
    recommended_workouts = set()
    for i, _ in sim_scores[1:4]:
        similar_disease = disease_list[i]
        st.write(f"Similar to: {similar_disease}")
        df = pd.read_csv("workout_df.csv")
        workouts = df[df["disease"] == similar_disease]["workout"].unique()
        recommended_workouts.update(workouts)

    st.subheader("Top Workout Recommendations:")
    for i, w in enumerate(recommended_workouts):
        st.write(f"{i+1}. {w}")
