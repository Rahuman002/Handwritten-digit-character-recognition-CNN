# app.py - Handwritten Text Recognition with Animations & History
import streamlit as st
import numpy as np
from PIL import Image
import cv2
import time
from datetime import datetime

# -------------------------------
# Fix for Pillow >= 10.0 (ANTIALIAS removed)
# -------------------------------
import PIL.Image
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS

# -------------------------------
# Check EasyOCR availability
# -------------------------------
try:
    import easyocr
    EASYOCR_OK = True
except ImportError:
    EASYOCR_OK = False

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Handwriting OCR", layout="wide", initial_sidebar_state="expanded")

# -------------------------------
# Custom CSS for minimal animations
# -------------------------------
st.markdown("""
<style>
    /* Button hover animation */
    .stButton > button {
        transition: all 0.3s ease;
        border: none;
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: bold;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        background: linear-gradient(45deg, #45a049, #4CAF50);
        cursor: pointer;
    }
    .stButton > button:active {
        transform: scale(0.98);
    }
    
    /* Fade-in animation for extracted text */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Pulse animation for the extract button while processing (via spinner) */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); opacity: 0.9; }
        100% { transform: scale(1); }
    }
    
    /* Sidebar history item hover */
    .history-item {
        padding: 8px;
        margin: 5px 0;
        border-radius: 6px;
        transition: background 0.2s;
        cursor: pointer;
    }
    .history-item:hover {
        background: rgba(76, 175, 80, 0.1);
    }
    
    /* Image preview subtle zoom on hover */
    .stImage > img {
        transition: transform 0.2s ease;
    }
    .stImage > img:hover {
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Initialize session state for history
# -------------------------------
if "ocr_history" not in st.session_state:
    st.session_state.ocr_history = []  # list of dicts: {timestamp, image, text, preview}

# -------------------------------
# Load EasyOCR (cached)
# -------------------------------
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'])

# -------------------------------
# Preprocess image
# -------------------------------
def preprocess_image(img):
    if isinstance(img, Image.Image):
        img_np = np.array(img)
    else:
        img_np = img
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    denoised = cv2.medianBlur(thresh, 3)
    return denoised

# -------------------------------
# Sidebar - History
# -------------------------------
with st.sidebar:
    st.markdown("## 📜 History")
    if st.button("🗑️ Clear History"):
        st.session_state.ocr_history = []
        st.rerun()
    
    if st.session_state.ocr_history:
        for idx, entry in enumerate(reversed(st.session_state.ocr_history[-10:])):  # show last 10
            with st.container():
                col1, col2 = st.columns([1, 3])
                with col1:
                    # tiny thumbnail
                    if entry["preview"]:
                        st.image(entry["preview"], width=50)
                with col2:
                    st.markdown(f"**{entry['timestamp']}**")
                    st.caption(entry["text"][:50] + "..." if len(entry["text"]) > 50 else entry["text"])
                st.markdown("---")
    else:
        st.info("No history yet. Extract text from an image and it will appear here.")

# -------------------------------
# Main UI
# -------------------------------
st.title("✍️ Handwritten Text Recognition")
st.markdown("Upload an image of handwritten text – the system will extract the text with **minimal animations** and save your history.")

if not EASYOCR_OK:
    st.error("❌ EasyOCR not installed. Run: `pip install easyocr`")
    st.stop()

uploaded_file = st.file_uploader("Choose an image (PNG, JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("🔍 Extract Text", type="primary", use_container_width=True):
        with st.spinner("Recognizing text... (pulse animation)"):
            try:
                reader = load_reader()
                processed = preprocess_image(image)
                results = reader.readtext(processed, detail=0, paragraph=True)
                text = ' '.join(results).strip()
                if not text:
                    results = reader.readtext(np.array(image), detail=0, paragraph=True)
                    text = ' '.join(results).strip()
                
                # Simulate a small delay to show the animation (optional)
                time.sleep(0.2)
                
                with col2:
                    if text:
                        st.success("✅ Extracted Text:")
                        st.markdown(f'<div class="fade-in" style="background-color:#f0f2f6; padding:15px; border-radius:10px; font-size:16px;">{text}</div>', unsafe_allow_html=True)
                        
                        # Save to history
                        # Create a tiny preview (resized thumbnail)
                        preview = image.copy()
                        preview.thumbnail((100, 100))
                        st.session_state.ocr_history.append({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "image": image,  # full image not stored to save memory, only preview
                            "text": text,
                            "preview": preview
                        })
                        # Keep only last 20 entries
                        if len(st.session_state.ocr_history) > 20:
                            st.session_state.ocr_history.pop(0)
                    else:
                        st.warning("⚠️ No text recognized. Try a clearer image.")
            except Exception as e:
                st.error(f"Error: {e}")
else:
    # Placeholder info when no image uploaded
    st.info("👈 Upload an image to get started. Your extracted text will appear here, and history will be saved on the sidebar.")

# Footer
st.markdown("---")
