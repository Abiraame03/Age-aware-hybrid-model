import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import os
import gdown

# --- CONFIGURATION ---
IMG_H, IMG_W = 160, 160
# Replace the ID below with your actual Google Drive File ID
DRIVE_FILE_ID = 'YOUR_ACTUAL_FILE_ID_HERE' 

# --- DOWNLOAD & LOAD MODEL ---
@st.cache_resource
def load_dyslexia_model():
    model_path = "Improved_Hybrid_AgeModel.keras"
    if not os.path.exists(model_path):
        url = f'https://drive.google.com/uc?id={DRIVE_FILE_ID}'
        try:
            with st.spinner("Downloading trained model from Google Drive..."):
                gdown.download(url, model_path, quiet=False)
        except Exception as e:
            st.error(f"Download failed: {e}")
            return None
            
    return tf.keras.models.load_model(model_path)

model = load_dyslexia_model()

# --- VIDEO PROCESSING CLASS ---
class HandwritingAnalyzer(VideoTransformerBase):
    def __init__(self, age):
        self.age = age

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Image Preprocessing
        img_resized = cv2.resize(img, (IMG_W, IMG_H))
        img_norm = img_resized / 255.0
        
        # 2. Reshape for model 
        # Since the model uses MobileNetV2 + BiLSTM, it expects: (Batch, Sequence, H, W, C)
        # We wrap the current frame as a single-step sequence
        img_input = np.expand_dims(img_norm, axis=0) # Shape: (1, 160, 160, 3)
        age_input = np.array([[self.age]])

        # 3. Prediction
        if model:
            prob = model.predict([img_input, age_input], verbose=0)[0][0]
            
            # Severity & Condition Logic
            if prob < 0.25: 
                res, sev, color = "Normal", "No Risk", (0, 255, 0)
            elif prob < 0.50: 
                res, sev, color = "Normal", "Low Risk", (255, 165, 0)
            elif prob < 0.75: 
                res, sev, color = "Dyslexic", "Moderate Risk", (0, 165, 255)
            else: 
                res, sev, color = "Dyslexic", "High Risk", (0, 0, 255)

            # Draw Overlay
            label = f"{res} ({sev})"
            cv2.putText(img, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            cv2.putText(img, f"Conf: {round(prob, 3)}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
        return img

# --- STREAMLIT UI ---
st.title("🧠 Live Handwriting & Dyslexia Analysis")
st.write("The model analyzes handwriting patterns based on age-specific development.")

with st.sidebar:
    st.header("Parameters")
    age = st.slider("Child's Age", 5, 15, 8)
    st.write("---")
    st.info("Ensure the camera is stable and handwriting is clearly visible.")

if model:
    webrtc_streamer(
        key="dyslexia-video",
        video_transformer_factory=lambda: HandwritingAnalyzer(age),
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
else:
    st.error("Model could not be loaded. Check your Drive File ID and permissions.")
