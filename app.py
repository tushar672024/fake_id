import pandas as pd
import streamlit as st
import joblib

# ------------------------
# Label Encoding Function
# ------------------------
def label_encoding(df):
    """
    Encode categorical features using one-hot encoding.
    Columns: 'profile_pic', 'extern_url', 'private', 'sim_name_username'
    """
    categorical_cols = ["profile_pic", "extern_url", "private", "sim_name_username"]
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    return df_encoded

# -----------------------
# Streamlit page config
# ------------------------
st.set_page_config(
    page_title="Instagram Fake Account Detector — Manual Input",
    page_icon="📝",
    layout="centered"
)
st.title("📝 Instagram Fake Account Detector — Manual Input")
st.caption("Fill in the profile details, and we'll predict Fake/Real.")

# ------------------------
# Load artifacts (model + scaler)
# ------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error("❌ Model or scaler file not found. Please upload model.pkl and scaler.pkl.")
    st.stop()

# ------------------------
# User Inputs
# ------------------------
username = st.text_input("Username")
fullname = st.text_input("Full Name")
num_posts = st.number_input("Number of Posts", min_value=0, step=1)
num_followers = st.number_input("Number of Followers", min_value=0, step=1)
num_following = st.number_input("Number of Following", min_value=0, step=1)
profile_pic = st.selectbox("Profile Picture Present?", ["Yes", "No"])
private = st.selectbox("Is Private Account?", ["Yes", "No"])
extern_url = st.selectbox("Has External URL?", ["Yes", "No"])

# Bio input
bio = st.text_area("Paste Bio Here")
len_desc = len(bio)  # Auto-compute bio length

# Ratios & Derived Features
ratio_numlen_username = sum(c.isdigit() for c in username) / max(len(username), 1)
ratio_numlen_fullname = sum(c.isdigit() for c in fullname) / max(len(fullname), 1)
sim_name_username = "Yes" if username and fullname and username.lower() in fullname.lower() else "No"

# ------------------------
# Predict Button
# ------------------------
if st.button("Predict"):
    # Step 1: Build feature row
    feat_row = {
        "profile_pic": profile_pic,
        "extern_url": extern_url,
        "private": private,
        "ratio_numlen_username": float(ratio_numlen_username),
        "len_fullname": int(len(fullname)),
        "ratio_numlen_fullname": float(ratio_numlen_fullname),
        "len_desc": int(len_desc),
        "num_posts": int(num_posts),
        "num_followers": int(num_followers),
        "num_following": int(num_following),
        "sim_name_username": sim_name_username
    }

    # Step 2: Create DataFrame
    features_df = pd.DataFrame([feat_row])

    try:
        # Step 3: Encode categorical variables
        df_encoded = label_encoding(features_df)

        # Step 4: Ensure column order matches training
        train_columns = scaler.feature_names_in_
        for col in train_columns:
            if col not in df_encoded.columns:
                # Add missing columns with 0
                df_encoded[col] = 0
        df_encoded = df_encoded[train_columns]

        # Step 5: Scale and Predict
        X = scaler.transform(df_encoded)
        pred = model.predict(X)[0]
        proba = float(model.predict_proba(X)[0, 1])

        # Step 6: Display results
        if pred == 1:
            st.error(f"🚨 FAKE Account Detected — probability {proba:.2f}")
        else:
            st.success(f"✅ Real Account Detected — probability {proba:.2f}")

    except Exception as e:
        st.error("❌ Model/encoding error — check that your artifacts match the training pipeline.")
        st.exception(e)
