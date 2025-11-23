import streamlit as st
import numpy as np
import cv2
import pytesseract
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# -------------------------------
# Configuration
# -------------------------------
st.set_page_config(page_title="AI Document Verification", layout="centered")
st.title("🧠 AI-Powered Identity Verification & Fraud Detection (UID Aadhaar)")
st.write("Upload an Aadhaar or KYC document below for automated verification and fraud detection.")

# -------------------------------
# Tesseract OCR Setup (LOCAL WINDOWS)
# -------------------------------
# If Tesseract is installed here, this is fine for your PC.
# On Render, you must REMOVE this line or change it.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -------------------------------
# Load Deep Learning Model (cached)
# -------------------------------
@st.cache_resource
def load_cnn_model():
    # 🔴 VERY IMPORTANT:
    # This assumes document_authentication_model.h5
    # is in the SAME FOLDER as app.py
    MODEL_PATH = "document_authentication_model.h5"
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

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    gray = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(gray, -1, kernel)

    norm_img = cv2.normalize(sharp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    bw = cv2.adaptiveThreshold(
        norm_img, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 29, 10
    )

    bw = cv2.resize(bw, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)

    cv2.imwrite("debug_preprocessed_for_ocr.png", bw)

    return bw

# -------------------------------
# Helper: OCR Text Extraction
# -------------------------------
def extract_text(img):
    custom_config = r'--oem 3 --psm 3'
    text = pytesseract.image_to_string(img, config=custom_config, lang='eng')
    return text

# -------------------------------
# Helper: Data Field Extraction
# -------------------------------
def extract_fields(text):
    text = text.replace("\n", " ")
    text = re.sub(r'[^A-Za-z0-9\s:/-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    st.text_area("🧾 Cleaned OCR Text (for debugging)", text, height=120)

    aadhaar_pattern = r"(\d{4}\s?\d{4}\s?\d{4}|\d{12})"
    aadhaar = re.search(aadhaar_pattern, text)

    dob_pattern = r"(?:DOB|DoB|D0B|Date of Birth|YOB|जन्म)[^\d]*([0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{2,4})"
    dob = re.search(dob_pattern, text, re.IGNORECASE)

    name_pattern = r"(?:Name|Narne|Nam|Nane|HaNa|Hame|Namee|नाम)\s*[:\-]?\s*([A-Za-z]{3,}(?:\s[A-Za-z]{2,}){0,3})"
    name = re.search(name_pattern, text, re.IGNORECASE)

    return {
        "Name": name.group(1).strip() if name else None,
        "DOB": dob.group(1).strip() if dob else None,
        "Aadhaar": aadhaar.group(1).strip() if aadhaar else None
    }

# -------------------------------
# Helper: Prediction
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
# File Upload Section
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload a document image (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)
    img = np.array(image_pil)

    img = cv2.resize(img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

    st.image(img, caption="📄 Upscaled Document (for better OCR)", use_container_width=True)

    with st.spinner("⚙️ Preprocessing image for OCR..."):
        preprocessed = preprocess_image(img)
    st.image(preprocessed, caption="🧾 Preprocessed (Used for OCR)", use_container_width=True)

    with st.spinner("🤖 Running AI model for authenticity check..."):
        label, confidence = predict_authenticity(img)
    st.subheader("🎯 Prediction Result")
    st.write(f"**Prediction:** {label}")
    st.write(f"**Confidence:** {confidence}")

    with st.spinner("🔍 Extracting text from image..."):
        extracted_text = extract_text(preprocessed)
    st.text_area("📝 Extracted OCR Text", extracted_text, height=150)

    fields = extract_fields(extracted_text)
    st.subheader("📋 Extracted Details")
    st.write(f"**Name:** {fields['Name'] if fields['Name'] else '❌ Not Found'}")
    st.write(f"**Date of Birth / YOB:** {fields['DOB'] if fields['DOB'] else '❌ Not Found'}")
    st.write(f"**Aadhaar Number:** {fields['Aadhaar'] if fields['Aadhaar'] else '❌ Not Found'}")

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
        st.error("❌ Verification Failed! Some details missing or document tampered.")
else:
    st.info("👆 Upload a document image to begin verification.")
