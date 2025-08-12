import streamlit as st
import pandas as pd
import joblib
import requests
from label_encoder import label_encoding

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Fetch Instagram profile data using RapidAPI
def fetch_instagram_features(username):
    url = "https://instagram-scraper-api2.p.rapidapi.com/v1/info"
    querystring = {"username_or_id_or_url": username}
    headers = {
        "X-RapidAPI-Key": st.secrets["RAPIDAPI_KEY"],
        "X-RapidAPI-Host": "instagram-scraper-api2.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    if response.status_code != 200:
        raise Exception(f"API request failed: {response.status_code}")
    data = response.json()

    # Extract profile details
    profile_data = data.get("data", {})
    if not profile_data:
        raise Exception("No profile data found")

    # Convert to features expected by model
    features = {
        "profile_pic": "Yes" if profile_data.get("profile_pic_url") else "No",
        "extern_url": "Yes" if profile_data.get("external_url") else "No",
        "private": "Yes" if profile_data.get("is_private") else "No",
        "ratio_numlen_username": sum(c.isdigit() for c in username) / len(username),
        "len_fullname": len(profile_data.get("full_name", "")),
        "ratio_numlen_fullname": sum(c.isdigit() for c in profile_data.get("full_name", "")) / max(len(profile_data.get("full_name", "")), 1),
        "len_desc": len(profile_data.get("biography", "")),
        "num_posts": profile_data.get("media_count", 0),
        "num_followers": profile_data.get("follower_count", 0),
        "num_following": profile_data.get("following_count", 0),
        "sim_name_username": "Yes" if profile_data.get("full_name", "").lower() in username.lower() else "No"
    }
    return pd.DataFrame([features])

# Prediction function
def predict_user(username):
    df_features = fetch_instagram_features(username)
    df_encoded = label_encoding(df_features)
    features_scaled = scaler.transform(df_encoded)
    pred = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0, 1]
    return pred, prob

# Streamlit UI
st.set_page_config(page_title="Instagram Fake Account Detector", page_icon="📷", layout="centered")
st.title("📷 Instagram Fake Account Detector")
st.write("Enter an Instagram username to check if the account is likely FAKE or REAL.")

username = st.text_input("Instagram Username (without @):")

if st.button("Check Account"):
    if not username.strip():
        st.warning("Please enter a username.")
    else:
        try:
            with st.spinner("Fetching profile data..."):
                pred, prob = predict_user(username)

            if pred == 1:
                st.error(f"🚨 FAKE Account Detected! (Probability: {prob:.2f})")
            else:
                st.success(f"✅ Real Account Detected! (Probability: {prob:.2f})")

        except Exception as e:
            st.error(f"Error: {e}")
