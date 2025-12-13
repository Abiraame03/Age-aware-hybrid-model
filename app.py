import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import os
import gdown

# --- CONFIGURATION ---
IMG_H, IMG_W = 160, 160
# This is extracted from your provided link
DRIVE_FILE_ID = '12aPOVWZZzG3KzdjGxcS7I5KkTFc_fh4q'

# --- MODEL LOADING LOGIC ---
@st.cache_resource
def load_dyslexia_model():
    model_path = "Improved_Hybrid_AgeModel.keras"
    if not os.path.exists(model_path):
        url = f'https://drive.google.com/uc?id={DRIVE_FILE_ID}'
        try:
            with st.spinner("Downloading AI Model from Google Drive..."):
                gdown.download(url, model_path, quiet=False)
        except Exception as e:
            st.error(f"Download failed: {e}")
            return None
    
    # Load without compiling to avoid optimizer version conflicts
    return tf.keras.models.load_model(model_path, compile=False)

model = load_dyslexia_model()

# --- LIVE VIDEO ANALYZER ---
class HandwritingAnalyzer(VideoTransformerBase):
    def __init__(self, age):
        self.age = age

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Image Preprocessing (Matches your training)
        img_resized = cv2.resize(img, (IMG_W, IMG_H))
        img_norm = img_resized / 255.0
        
        # 2. Reshape for MobileNetV2 + BiLSTM
        # Input shape: (Batch_size, H, W, C)
        img_input = np.expand_dims(img_norm, axis=0).astype(np.float32)
        age_input = np.array([[self.age]], dtype=np.float32)

        if model:
            try:
                # 3. Prediction
                prob = model.predict([img_input, age_input], verbose=0)[0][0]
                
                # 4. Severity & Logic
                if prob < 0.25: 
                    res, sev, color = "Normal", "No Risk", (0, 255, 0) # Green
                elif prob < 0.50: 
                    res, sev, color = "Normal", "Low Risk", (255, 165, 0) # Orange
                elif prob < 0.75: 
                    res, sev, color = "Dyslexic", "Moderate Risk", (0, 165, 255) # Light Blue
                else: 
                    res, sev, color = "Dyslexic", "High Risk", (0, 0, 255) # Red

                # 5. Visual Overlays
                label = f"{res}: {sev}"
                cv2.putText(img, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.putText(img, f"Conf: {round(float(prob), 3)}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.rectangle(img, (10, 10), (img.shape[1]-10, img.shape[0]-10), color, 4)
                
            except Exception as e:
                # Silently fail for individual frames to keep video smooth
                pass
            
        return img

# --- STREAMLIT USER INTERFACE ---
st.set_page_config(page_title="Dyslexia AI Screener", layout="centered")
st.title("📝 Handwriting Pattern Analysis")
st.write("Live analysis for dyslexia detection considering handwriting & child's age.")

with st.sidebar:
    st.header("Analysis Settings")
    age_val = st.slider("Select Child's Age:", 5, 15, 8)
    st.divider()
    st.info("Ensure the paper is well-lit and the camera is steady.")

if model:
    webrtc_streamer(
        key="dyslexia-video-stream",
        video_transformer_factory=lambda: HandwritingAnalyzer(age_val),
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
    )
else:
    st.error("Model could not be initialized. Please check the logs.")
