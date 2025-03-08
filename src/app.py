# app.py
import streamlit as st
from rec_engine import get_recommendations
from response_gen import generate_response

def main():
    st.title("AI-Driven Supplement Recommender")
    
    st.header("Enter Your Health Profile")
    diet = st.selectbox("Diet Type", options=["Omnivore", "Vegetarian", "Vegan", "Pescatarian"])
    health_goals = st.multiselect("Health Goals", options=["Weight Loss", "Muscle Gain", "General Wellness", "Improved Digestion"])
    lifestyle = st.selectbox("Lifestyle", options=["Sedentary", "Active", "Very Active"])
    conditions = st.text_input("Existing Health Conditions (comma separated)")
    location = st.text_input("Your Location (City, Country)")
    
    if st.button("Get Recommendations"):
        user_profile = {
            "diet": diet,
            "health_goals": health_goals,
            "lifestyle": lifestyle,
            "conditions": [cond.strip() for cond in conditions.split(",")] if conditions else [],
            "location": location
        }
        st.write("User Profile:", user_profile)
        
        # Call the recommendation engine
        recommendations = get_recommendations(user_profile)
        st.write("Raw Recommendations:", recommendations)
        
        # Generate a full response with scientific context and purchasing info
        final_response = generate_response(recommendations, user_profile)
        st.write(final_response)

if __name__ == "__main__":
    main()
