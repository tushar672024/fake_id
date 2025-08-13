import io
import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import streamlit as st
import joblib
from PIL import Image

# OCR: EasyOCR (downloads model weights once on first run)
import easyocr

# Your encoder
from label_encoder import label_encoding


# ---------------------------
# Utilities
# ---------------------------
def similar(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def parse_compact_number(s: str) -> int:
    """
    Parse Instagram-style compact numbers:
    '12.3k', '1.2K', '1,234', '2.1m', '987', '1.2 M' -> int
    """
    if not s:
        return 0
    x = s.strip().lower().replace(",", "")
    m = re.match(r"^([0-9]*\.?[0-9]+)\s*([kmb])?$", x)
    if m:
        val = float(m.group(1))
        suf = m.group(2)
        if suf == "k":
            return int(val * 1_000)
        elif suf == "m":
            return int(val * 1_000_000)
        elif suf == "b":
            return int(val * 1_000_000_000)
        else:
            return int(val)
    # fallback: extract digits
    digits = re.sub(r"[^\d]", "", x)
    return int(digits) if digits else 0


def extract_counts(text: str):
    """
    Try to extract posts, followers, following counts from the OCR text blob.
    Works with patterns like "1,234 posts", "12.3k followers", "56 following".
    """
    t = text.lower()

    # Try “X posts”
    posts = 0
    m = re.search(r"([0-9][0-9,\.]*\s*[kmbKMB]?)\s*posts?", t)
    if m:
        posts = parse_compact_number(m.group(1))

    # Try “X followers”
    followers = 0
    m = re.search(r"([0-9][0-9,\.]*\s*[kmbKMB]?)\s*followers?", t)
    if m:
        followers = parse_compact_number(m.group(1))

    # Try “X following”
    following = 0
    m = re.search(r"([0-9][0-9,\.]*\s*[kmbKMB]?)\s*following", t)
    if m:
        following = parse_compact_number(m.group(1))

    return posts, followers, following


def extract_username(lines: list[str]) -> str:
    """
    Heuristics to guess the username from OCR lines.
    Prefer lines with @something, otherwise a short alnum handle on top lines.
    """
    # Prefer lines containing @
    for line in lines[:8]:  # look near the top
        m = re.search(r"@([A-Za-z0-9._]+)", line)
        if m:
            return m.group(1)

    # Otherwise, pick first short-ish alnum-ish token that could be a handle
    for line in lines[:8]:
        m = re.search(r"\b([A-Za-z0-9._]{3,30})\b", line)
        if m:
            return m.group(1)

    return ""


def extract_fullname(lines: list[str]) -> str:
    """
    Heuristics to guess full name: typically a line near top that:
    - doesn't start with '@'
    - has spaces or titlecase words
    """
    for line in lines[:10]:
        if line.strip() and not line.strip().startswith("@"):
            # Avoid lines that are clearly counts/CTA
            if not re.search(r"(posts?|followers?|following|message|follow|edit profile)", line.lower()):
                return line.strip()
    return ""


def detect_private(text: str) -> bool:
    t = text.lower()
    return ("this account is private" in t) or ("private account" in t) or ("requested" in t and "follow" in t)


def detect_external_url(text: str) -> bool:
    return bool(re.search(r"(https?://|www\.)", text.lower()))


def ocr_image_to_text(reader: easyocr.Reader, image: Image.Image) -> tuple[list[str], str]:
    """
    Run OCR and return (lines, joined_text)
    """
    # EasyOCR accepts numpy arrays (RGB)
    arr = np.array(image.convert("RGB"))
    lines = reader.readtext(arr, detail=0)
    joined = "\n".join(lines)
    return lines, joined


# ---------------------------
# App
# ---------------------------
st.set_page_config(page_title="Instagram Fake Account Detector (Image-Only)", page_icon="📷", layout="centered")
st.title("📷 Instagram Fake Account Detector — Image Only")
st.caption("Upload an Instagram profile **screenshot**. We’ll OCR it, engineer features, and predict Fake/Real — no external APIs.")

# Model + scaler
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

# OCR reader (cache to avoid reloading)
@st.cache_resource
def get_reader():
    # First run downloads model weights; needs one-time internet
    return easyocr.Reader(["en"], gpu=False)

model, scaler = load_artifacts()
reader = get_reader()

uploaded = st.file_uploader("Upload Instagram profile screenshot", type=["png", "jpg", "jpeg"])

with st.expander("Optional settings"):
    assume_profile_pic = st.selectbox("Profile picture present?", ["Auto (assume Yes)", "Assume Yes", "Assume No"], index=0)
    show_debug = st.checkbox("Show OCR & feature debug", value=False)

if uploaded:
    try:
        img_bytes = uploaded.read()
        image = Image.open(io.BytesIO(img_bytes))

        with st.spinner("Reading text from image…"):
            lines, text_blob = ocr_image_to_text(reader, image)

        # Extract structured data
        username = extract_username(lines)
        fullname = extract_fullname(lines)
        posts, followers, following = extract_counts(text_blob)

        # Booleans
        private = detect_private(text_blob)
        extern_url = detect_external_url(text_blob)

        # Profile pic heuristic
        if assume_profile_pic == "Assume Yes":
            has_pp = True
        elif assume_profile_pic == "Assume No":
            has_pp = False
        else:
            # Auto: Instagram profile screenshots usually show a picture area; full CV detection is complex.
            # We default to Yes for better recall; you can later add a circle/face detector.
            has_pp = True

        # Feature engineering
        ratio_numlen_username = (sum(c.isdigit() for c in username) / max(len(username), 1)) if username else 0.0
        len_fullname = len(fullname)
        ratio_numlen_fullname = (sum(c.isdigit() for c in fullname) / max(len(fullname), 1)) if fullname else 0.0
        # Bio length proxy: remove count lines and obvious UI words to approximate biography
        cleaned_blob = re.sub(r"(posts?|followers?|following|message|follow|edit profile|share profile|contacts?)", "", text_blob, flags=re.I)
        len_desc = len(cleaned_blob.strip())

        sim_name_username = "Yes" if similar(username, fullname) >= 0.7 and username and fullname else "No"

        # Assemble features for your model
        feat_row = {
            "profile_pic": "Yes" if has_pp else "No",
            "extern_url": "Yes" if extern_url else "No",
            "private": "Yes" if private else "No",
            "ratio_numlen_username": float(ratio_numlen_username),
            "len_fullname": int(len_fullname),
            "ratio_numlen_fullname": float(ratio_numlen_fullname),
            "len_desc": int(len_desc),
            "num_posts": int(posts),
            "num_followers": int(followers),
            "num_following": int(following),
            "sim_name_username": sim_name_username
        }

        features_df = pd.DataFrame([feat_row])

        if show_debug:
            st.subheader("🧪 OCR Debug")
            st.image(image, caption="Uploaded screenshot", use_column_width=True)
            st.code(text_blob, language="text")

            st.subheader("🔎 Parsed Fields")
            st.json({
                "username": username,
                "fullname": fullname,
                "posts": posts,
                "followers": followers,
                "following": following,
                "private_detected": private,
                "external_url_detected": extern_url,
                "similarity(name vs username)": similar(username, fullname)
            })

            st.subheader("🧱 Model Features")
            st.dataframe(features_df)

        # Predict
        try:
            df_encoded = label_encoding(features_df)
            X = scaler.transform(df_encoded)
            pred = model.predict(X)[0]
            proba = float(model.predict_proba(X)[0, 1])
        except Exception as model_err:
            st.error("Model/encoding error. Double-check that `label_encoder.py`, `model.pkl`, and `scaler.pkl` match the training pipeline.")
            st.exception(model_err)
            st.stop()

        # Output
        if pred == 1:
            st.error(f"🚨 FAKE Account Detected — probability {proba:.2f}")
        else:
            st.success(f"✅ Real Account Detected — probability {proba:.2f}")

        # Quick summary card
        st.markdown("---")
        st.markdown("#### Summary")
        cols = st.columns(3)
        cols[0].metric("Posts", f"{posts:,}")
        cols[1].metric("Followers", f"{followers:,}")
        cols[2].metric("Following", f"{following:,}")
        st.caption(f"Username: `{username or 'N/A'}` | Name: `{fullname or 'N/A'}` | Private: **{'Yes' if private else 'No'}** | External URL: **{'Yes' if extern_url else 'No'}**")

    except Exception as e:
        st.error("Failed to process the image.")
        st.exception(e)
else:
    st.info("Upload a clear screenshot of the profile page (top area with username, counts, and bio visible).")
