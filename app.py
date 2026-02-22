import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# LOAD SAVED MODEL & DATA
# -----------------------------
model = joblib.load("model.pkl")
data = joblib.load("processed_data.pkl")

# Features used during training
features = [
    "Batting_Average",
    "Batting_Strike_Rate",
    "Total_Runs",
    "Strike_Rate",
    "IPL_Matches"
]

st.title("🏏 IPL Player Performance Prediction (2026)")

# Player selection dropdown
player_list = sorted(data["Player_Name"].unique())
selected_player = st.selectbox("Select Player", player_list)

if st.button("Generate Player Report"):

    player = data[data["Player_Name"] == selected_player]

    if player.empty:
        st.error("Player not found!")
    else:
        # Get exact training features
        input_df = player[features]

        # Predict
        predicted_runs = model.predict(input_df)[0]
        current_runs = player["Runs_Scored"].iloc[0]

        # Performance level logic
        if predicted_runs >= 500:
            level = "High"
        elif predicted_runs >= 200:
            level = "Medium"
        else:
            level = "Low"

        form = "Good Form" if predicted_runs > current_runs else "Poor Form"

        # Weather impact (rule-based logic)
        weather = ["Sunny", "Cloudy", "Humid"]
        weather_scores = [
            predicted_runs + 15,
            predicted_runs + 10,
            predicted_runs + 5
        ]
        best_weather = weather[np.argmax(weather_scores)]

        # Pitch impact (rule-based logic)
        pitch = ["Flat", "Green", "Dusty"]
        pitch_scores = [
            predicted_runs + 20,
            predicted_runs + 10,
            predicted_runs + 15
        ]
        best_pitch = pitch[np.argmax(pitch_scores)]

        yearly = data[data["Player_Name"] == selected_player][
            ["Year", "Runs_Scored"]
        ]

        # -----------------------------
        # DISPLAY REPORT
        # -----------------------------
        st.subheader("🏏 Complete Player Report")
        st.write("**Player:**", selected_player)
        st.write("**Current Runs:**", int(current_runs))
        st.write("**Predicted Runs (2026):**", int(predicted_runs))
        st.write("**Performance Level:**", level)
        st.write("**Player Form:**", form)
        st.write("**Best Weather:**", best_weather)
        st.write("**Best Pitch:**", best_pitch)

        st.subheader("📊 Year Wise Runs")
        st.dataframe(yearly)

        # -----------------------------
        # GRAPHS
        # -----------------------------
        # Runs Comparison
        fig1, ax1 = plt.subplots()
        ax1.bar(["Current", "Predicted"],
                [current_runs, predicted_runs])
        ax1.set_title("Runs Comparison")
        st.pyplot(fig1)

        # Weather Impact
        fig2, ax2 = plt.subplots()
        ax2.bar(weather, weather_scores)
        ax2.set_title("Weather Impact")
        st.pyplot(fig2)

        # Pitch Impact
        fig3, ax3 = plt.subplots()
        ax3.bar(pitch, pitch_scores)
        ax3.set_title("Pitch Impact")
        st.pyplot(fig3)