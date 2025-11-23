import streamlit as st
import numpy as np
import cv2
import pytesseract
import re
import os
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# -------------------------------
# Configuration
# -------------------------------
st.set_page_config(page_title="AI Document Verification", layout="centered")
st.title("🧠 AI-Powered Identity Verification & Fraud Detection (UID Aadhaar)")
st.write("Upload an Aadhaar or KYC document below for automated verification and fraud detection.")

# -------------------------------
# Tesseract OCR Setup
# -------------------------------
# Disable Windows path for Render
# Tesseract is pre-installed in Render Linux image
pytesseract.pytesseract.tesseract_cmd = "tesseract"

# -------------------------------
# Load Deep Learning Model (Render-Friendly)
# -------------------------------
@st.cache_resource
def load_cnn_model():
    # Load model from same folder as app.py
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "document_authentication_model.h5")
    model = load_model(MODEL_PATH)
    return model

with st.spinner("🔄 Loading AI model... Please wait."):
    model = load_cnn_model()
st.success("✅ Model loaded successfully!")

# -------------------------------
# Helper: Image Preprocessing for OCR
# -------------------------------
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Denoise
    gray = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)

    # Sharpen
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(gray, -1, kernel)

    # Normalize
    norm_img = cv2.normalize(sharp, None, 0, 255, cv2.NORM_MINMAX)

    # Threshold
    bw = cv2.adaptiveThreshold(norm_img, 255,
                               cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 29, 10)

    # Upscale
    bw = cv2.resize(bw, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)

    return bw

# -------------------------------
# Extract OCR Text
# -------------------------------
def extract_text(img):
    config = r'--oem 3 --psm 3'
    return pytesseract.image_to_string(img, config=config)

# -------------------------------
# Extract Aadhaar Fields
# -------------------------------
def extract_fields(text):

    text = text.replace("\n", " ")
    text = re.sub(r'[^A-Za-z0-9\s:/-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    st.text_area("🧾 Cleaned OCR Text (debug)", text, height=120)

    aadhaar_pattern = r"(\d{4}\s?\d{4}\s?\d{4}|\d{12})"
    dob_pattern = r"(?:DOB|Date of Birth|YOB)[^\d]*([0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{2,4})"
    name_pattern = r"(?:Name)\s*[:\-]?\s*([A-Za-z]{3,}(?:\s[A-Za-z]{2,}){0,3})"

    aadhaar = re.search(aadhaar_pattern, text)
    dob = re.search(dob_pattern, text, re.IGNORECASE)
    name = re.search(name_pattern, text, re.IGNORECASE)

    return {
        "Name": name.group(1).strip() if name else None,
        "DOB": dob.group(1).strip() if dob else None,
        "Aadhaar": aadhaar.group(1).strip() if aadhaar else None
    }

# -------------------------------
# Prediction: Genuine vs Tampered
# -------------------------------
def predict_authenticity(img_array):
    img_array = cv2.resize(img_array, (224, 224))
    img_array = image.img_to_array(img_array)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    confidence = float(prediction[0][0])

    label = "Tampered" if confidence > 0.5 else "Genuine"
    confidence = confidence if label == "Tampered" else 1 - confidence

    return label, round(confidence, 3)

# -------------------------------
# File Upload UI
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload Document Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_pil = Image.open(uploaded_file)
    img = np.array(image_pil)

    img = cv2.resize(img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    st.image(img, caption="Upscaled Document", use_container_width=True)

    with st.spinner("⚙️ Preprocessing for OCR..."):
        processed = preprocess_image(img)

    st.image(processed, caption="OCR Preprocessed Image", use_container_width=True)

    with st.spinner("🤖 Running AI Prediction..."):
        label, confidence = predict_authenticity(img)

    st.subheader("🎯 Prediction Result")
    st.write(f"**Prediction:** {label}")
    st.write(f"**Confidence:** {confidence}")

    with st.spinner("🔍 Extracting text..."):
        extracted_text = extract_text(processed)

    st.text_area("📝 OCR Output", extracted_text, height=150)

    fields = extract_fields(extracted_text)

    st.subheader("📋 Extracted Details")
    st.write(f"**Name:** {fields['Name'] or '❌ Not Found'}")
    st.write(f"**DOB/YOB:** {fields['DOB'] or '❌ Not Found'}")
    st.write(f"**Aadhaar:** {fields['Aadhaar'] or '❌ Not Found'}")

    st.write("---")
    st.subheader("✅ Final Verification Summary")

    validation_status = "🟩 Passed" if label == "Genuine" and all(fields.values()) else "🟥 Failed"

    st.markdown(f"""
    **Authenticity:** {label}  
    **Confidence:** {confidence}  
    **Name:** {fields['Name'] or 'Missing'}  
    **DOB/YOB:** {fields['DOB'] or 'Missing'}  
    **Aadhaar:** {fields['Aadhaar'] or 'Missing'}  
    **Validation:** {validation_status}
    """)

    if validation_status == "🟩 Passed":
        st.success("✅ Document Verified Successfully!")
    else:
        st.error("❌ Verification Failed!")
else:
    st.info("👆 Upload a document image to begin verification.")
