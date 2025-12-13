import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Dyslexia Detection and severity prediction", layout="centered")

# --- LOAD TFLITE MODEL ---
@st.cache_resource
def load_tflite_model():
    model_path = "Improved_Hybrid_AgeModel.tflite"
    if not os.path.exists(model_path):
        st.error(f"'{model_path}' not found in GitHub repository!")
        return None
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

# --- PREDICTION LOGIC ---
class DyslexiaAnalyzer(VideoTransformerBase):
    def __init__(self, age):
        self.age = age
        if interpreter:
            self.input_details = interpreter.get_input_details()
            self.output_details = interpreter.get_output_details()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Preprocess Image
        # Resize to match your training (160x160)
        img_resized = cv2.resize(img, (160, 160))
        img_norm = (img_resized / 255.0).astype(np.float32)
        img_input = np.expand_dims(img_norm, axis=0) # Shape (1, 160, 160, 3)
        
        # 2. Preprocess Age
        age_input = np.array([[self.age]], dtype=np.float32)

        if interpreter:
            try:
                # TFLite needs inputs set by index
                # Usually, index 0 is Image and index 1 is Age (or vice versa)
                # We identify them by shape to be safe
                for detail in self.input_details:
                    if len(detail['shape']) == 4: # Image input (1, 160, 160, 3)
                        interpreter.set_tensor(detail['index'], img_input)
                    else: # Age input (1, 1)
                        interpreter.set_tensor(detail['index'], age_input)
                
                interpreter.invoke()
                
                # Get Result
                prob = interpreter.get_tensor(self.output_details[0]['index'])[0][0]

                # 3. Visualization Logic
                if prob < 0.25: res, sev, color = "Normal", "No Risk", (0, 255, 0)
                elif prob < 0.50: res, sev, color = "Normal", "Low Risk", (255, 165, 0)
                elif prob < 0.75: res, sev, color = "Dyslexic", "Moderate", (0, 165, 255)
                else: res, sev, color = "Dyslexic", "High Risk", (0, 0, 255)

                # Draw to screen
                cv2.putText(img, f"{res}: {sev}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.rectangle(img, (5, 5), (img.shape[1]-5, img.shape[0]-5), color, 4)
            except Exception as e:
                pass

        return img

# --- UI ---
st.title("🧠 Live Dyslexia Handwriting Analysis")
st.write("Using optimized TFLite for real-time edge detection.")

age = st.sidebar.slider("Student Age", 5, 15, 8)

if interpreter:
    webrtc_streamer(
        key="dyslexia-tflite",
        video_transformer_factory=lambda: DyslexiaAnalyzer(age),
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False}
    )
