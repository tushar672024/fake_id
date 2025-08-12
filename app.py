import streamlit as st
import pandas as pd
import joblib
import instaloader
from label_encoder import label_encoding

# ---------------------------
# Load trained model and scaler
# ---------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------------------
# Fetch Instagram profile data
# ---------------------------
def fetch_instagram_features(username):
    L = instaloader.Instaloader()

    # Optional: Login to avoid rate limits
    # L.login("your_username", "your_password")

    profile = instaloader.Profile.from_username(L.context, username)

    data = {
        "profile_pic": "Yes" if profile.has_profile_pic else "No",
        "extern_url": "Yes" if profile.external_url else "No",
        "private": "Yes" if profile.is_private else "No",
        "ratio_numlen_username": sum(c.isdigit() for c in username) / len(username),
        "len_fullname": len(profile.full_name),
        "ratio_numlen_fullname": sum(c.isdigit() for c in profile.full_name) / max(len(profile.full_name), 1),
        "len_desc": len(profile.biography),
        "num_posts": profile.mediacount,
        "num_followers": profile.followers,
        "num_following": profile.followees,
        "sim_name_username": "Yes" if profile.full_name.lower() in username.lower() else "No"
    }
    return pd.DataFrame([data])

# ---------------------------
# Prediction function
# ---------------------------
def predict_user(username):
    df_features = fetch_instagram_features(username)
    df_encoded = label_encoding(df_features)
    features_scaled = scaler.transform(df_encoded)
    pred = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0, 1]
    return pred, prob

# ---------------------------
# Streamlit UI
# ---------------------------
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
            st.error(f"Error fetching data: {e}")
