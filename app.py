import streamlit as st
import pandas as pd
import joblib
from label_encoder import label_encoding

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.set_page_config(page_title="Instagram Fake Account Detector", page_icon="📷", layout="centered")
st.title("📷 Instagram Fake Account Detector")
st.write("Fill in the details below to check if the account is likely FAKE or REAL.")

# User inputs instead of API call
profile_pic = st.selectbox("Profile Picture Present?", ["Yes", "No"])
extern_url = st.selectbox("External URL Present?", ["Yes", "No"])
private = st.selectbox("Is Account Private?", ["Yes", "No"])
ratio_numlen_username = st.number_input("Ratio of Digits in Username", min_value=0.0, max_value=1.0, step=0.01)
len_fullname = st.number_input("Full Name Length", min_value=0)
ratio_numlen_fullname = st.number_input("Ratio of Digits in Full Name", min_value=0.0, max_value=1.0, step=0.01)
len_desc = st.number_input("Biography Length", min_value=0)
num_posts = st.number_input("Number of Posts", min_value=0)
num_followers = st.number_input("Number of Followers", min_value=0)
num_following = st.number_input("Number Following", min_value=0)
sim_name_username = st.selectbox("Full Name Similar to Username?", ["Yes", "No"])

if st.button("Check Account"):
    try:
        # Create DataFrame with inputs
        features = pd.DataFrame([{
            "profile_pic": profile_pic,
            "extern_url": extern_url,
            "private": private,
            "ratio_numlen_username": ratio_numlen_username,
            "len_fullname": len_fullname,
            "ratio_numlen_fullname": ratio_numlen_fullname,
            "len_desc": len_desc,
            "num_posts": num_posts,
            "num_followers": num_followers,
            "num_following": num_following,
            "sim_name_username": sim_name_username
        }])

        # Encode and scale
        df_encoded = label_encoding(features)
        features_scaled = scaler.transform(df_encoded)

        # Predict
        pred = model.predict(features_scaled)[0]
        prob = model.predict_proba(features_scaled)[0, 1]

        # Show result
        if pred == 1:
            st.error(f"🚨 FAKE Account Detected! (Probability: {prob:.2f})")
        else:
            st.success(f"✅ Real Account Detected! (Probability: {prob:.2f})")

    except Exception as e:
        st.error(f"Error: {e}")
