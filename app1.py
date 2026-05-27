# app.py - Handwritten Text Recognition from Mobile Photos (with preprocessing)
import streamlit as st
import numpy as np
from PIL import Image
import cv2
import easyocr
from datetime import datetime

# -------------------------------
# Fix for Pillow >= 10.0 (ANTIALIAS removed)
# -------------------------------
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS

st.set_page_config(page_title="Handwriting OCR - Mobile Ready", layout="wide", initial_sidebar_state="expanded")

# -------------------------------
# Custom CSS for minimal animations (same as before)
# -------------------------------
st.markdown("""
<style>
    .stButton > button {
        transition: all 0.3s ease;
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: bold;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Session state for history
# -------------------------------
if "ocr_history" not in st.session_state:
    st.session_state.ocr_history = []

# -------------------------------
# EasyOCR reader (cached)
# -------------------------------
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'])

# -------------------------------
# Advanced preprocessing for mobile photos
# -------------------------------
def preprocess_mobile_image(img):
    """
    Preprocess a mobile‑captured handwritten image:
    - Convert to grayscale
    - Resize large images (speed up)
    - Deskew (rotate to straighten text)
    - Increase contrast
    - Remove noise
    - Adaptive thresholding
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    # Resize if too large (max 1200px width)
    h, w = gray.shape
    if w > 1200:
        scale = 1200 / w
        new_w = 1200
        new_h = int(h * scale)
        gray = cv2.resize(gray, (new_w, new_h))

    # --- Deskew (rotate to correct slight tilt) ---
    # Use Hough line detection to find the dominant angle
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            if -45 < angle < 45:
                angles.append(angle)
        if angles:
            median_angle = np.median(angles)
            if abs(median_angle) > 0.5:
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_REPLICATE)

    # --- Contrast enhancement (CLAHE) ---
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # --- Noise removal (bilateral filter preserves edges) ---
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

    # --- Adaptive thresholding (better for variable lighting) ---
    binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Optional: small morphological closing to connect broken strokes
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return cleaned

# -------------------------------
# Sidebar history
# -------------------------------
with st.sidebar:
    st.markdown("## 📜 History")
    if st.button("🗑️ Clear History"):
        st.session_state.ocr_history = []
        st.rerun()
    if st.session_state.ocr_history:
        for entry in reversed(st.session_state.ocr_history[-10:]):
            st.image(entry["preview"], width=60)
            st.caption(f"{entry['timestamp']}\n{entry['text'][:60]}...")
            st.markdown("---")
    else:
        st.info("No history yet.")

# -------------------------------
# Main UI
# -------------------------------
st.title("✍️ Handwritten Text Recognition")
st.markdown("Take a photo of your handwritten note with your **mobile phone**, upload it, and the text will be extracted.")

uploaded_file = st.file_uploader("Choose an image (PNG, JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)
    
    if st.button("🔍 Extract Text", type="primary"):
        with st.spinner("Preprocessing image and recognizing text..."):
            try:
                # Convert PIL to numpy (RGB)
                img_np = np.array(image)
                # Preprocess for mobile
                processed_img = preprocess_mobile_image(img_np)
                # Show preprocessed image in a small expander (optional)
                with st.expander("See preprocessed image (after deskew, contrast, threshold)"):
                    st.image(processed_img, caption="Preprocessed", use_column_width=True, clamp=True)
                
                # Run EasyOCR
                reader = load_reader()
                results = reader.readtext(processed_img, detail=0, paragraph=True)
                extracted_text = ' '.join(results).strip()
                
                # If still empty, try without aggressive preprocessing (fallback)
                if not extracted_text:
                    results = reader.readtext(img_np, detail=0, paragraph=True)
                    extracted_text = ' '.join(results).strip()
                
                with col2:
                    if extracted_text:
                        st.success("✅ Extracted Text:")
                        st.markdown(f'<div class="fade-in" style="background-color:#f0f2f6; padding:15px; border-radius:10px; font-size:16px;">{extracted_text}</div>', unsafe_allow_html=True)
                        # Save history
                        preview = image.copy()
                        preview.thumbnail((100, 100))
                        st.session_state.ocr_history.append({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "text": extracted_text,
                            "preview": preview
                        })
                    else:
                        st.warning("No text recognized. Try taking a photo with better lighting and straight angle.")
            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("---")
st.caption("Optimized for mobile‑captured handwriting: deskew, contrast enhancement, and adaptive thresholding.")