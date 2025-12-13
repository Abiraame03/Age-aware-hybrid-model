import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Dyslexia AI - TFLite", layout="centered")

# --- LOAD TFLITE MODEL ---
@st.cache_resource
def load_tflite_model():
    model_path = "Improved_Hybrid_AgeModel.tflite"
    if not os.path.exists(model_path):
        st.error(f"'{model_path}' not found! Please check the filename in your GitHub repo.")
        return None
    
    # TFLite Interpreter setup
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

# --- VIDEO PROCESSING CLASS ---
class DyslexiaAnalyzer(VideoProcessorBase):
    def __init__(self):
        self.age = 8  # Default age
        if interpreter:
            self.input_details = interpreter.get_input_details()
            self.output_details = interpreter.get_output_details()

    def recv(self, frame):
        # Convert incoming WebRTC frame to OpenCV BGR format
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Preprocess Image (160x160)
        img_resized = cv2.resize(img, (160, 160))
        img_norm = (img_resized / 255.0).astype(np.float32)
        img_input = np.expand_dims(img_norm, axis=0)
        
        # 2. Preprocess Age
        age_input = np.array([[self.age]], dtype=np.float32)

        if interpreter:
            try:
                # Map inputs to correct indices based on shape
                for detail in self.input_details:
                    if len(detail['shape']) == 4: # Image input (Batch, H, W, C)
                        interpreter.set_tensor(detail['index'], img_input)
                    else: # Age input (Batch, 1)
                        interpreter.set_tensor(detail['index'], age_input)
                
                interpreter.invoke()
                prob = interpreter.get_tensor(self.output_details[0]['index'])[0][0]

                # 3. Severity Logic
                if prob < 0.25: res, sev, color = "Normal", "No Risk", (0, 255, 0)
                elif prob < 0.50: res, sev, color = "Normal", "Low Risk", (0, 255, 255)
                elif prob < 0.75: res, sev, color = "Dyslexic", "Moderate", (0, 165, 255)
                else: res, sev, color = "Dyslexic", "High Risk", (0, 0, 255)

                # 4. Draw Overlay on Frame
                cv2.putText(img, f"{res}: {sev}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.putText(img, f"Conf: {round(float(prob), 3)}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.rectangle(img, (10, 10), (img.shape[1]-10, img.shape[0]-10), color, 4)
            except Exception as e:
                pass

        # Return processed frame back to the browser
        return frame.from_ndarray(img, format="bgr24")

# --- UI ---
st.title("🧠 Dyslexia Handwriting Analysis")
st.write("Live AI analysis using a Hybrid MobileNet-BiLSTM TFLite model.")

age_val = st.sidebar.slider("Student Age", 5, 15, 8)

if interpreter:
    # Initialize Streamer
    ctx = webrtc_streamer(
        key="dyslexia-tflite",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=DyslexiaAnalyzer,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Pass the sidebar age value to the video processor
    if ctx.video_processor:
        ctx.video_processor.age = age_val
else:
    st.warning("Awaiting model initialization...")
