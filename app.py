import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Dyslexia AI - TFLite", layout="wide")

# --- MODEL LOADING ---
@st.cache_resource
def load_tflite_model():
    model_path = "Improved_Hybrid_AgeModel.tflite"
    if not os.path.exists(model_path):
        st.error(f"Error: {model_path} not found in the repository!")
        return None
    
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

# --- VIDEO PROCESSING LOGIC ---
class DyslexiaAnalyzer(VideoProcessorBase):
    def __init__(self):
        self.age = 8  # Default age value
        if interpreter:
            self.input_details = interpreter.get_input_details()
            self.output_details = interpreter.get_output_details()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Image Preprocessing (160x160)
        img_resized = cv2.resize(img, (160, 160))
        img_norm = (img_resized / 255.0).astype(np.float32)
        img_input = np.expand_dims(img_norm, axis=0) # Shape: (1, 160, 160, 3)
        
        # 2. Age Input Preprocessing
        age_input = np.array([[self.age]], dtype=np.float32)

        if interpreter:
            try:
                # 3. Dynamic Input Mapping
                # Identify which input index is for Image and which is for Age
                for detail in self.input_details:
                    if len(detail['shape']) == 4: # Typically the Image input
                        interpreter.set_tensor(detail['index'], img_input)
                    else: # Typically the Age input (scalar/1D)
                        interpreter.set_tensor(detail['index'], age_input)
                
                # 4. Run Inference
                interpreter.invoke()
                prob = interpreter.get_tensor(self.output_details[0]['index'])[0][0]

                # 5. Logic & Visualization
                if prob < 0.25: res, sev, color = "Normal", "No Risk", (0, 255, 0)
                elif prob < 0.50: res, sev, color = "Normal", "Low Risk", (0, 255, 255)
                elif prob < 0.75: res, sev, color = "Dyslexic", "Moderate", (0, 165, 255)
                else: res, sev, color = "Dyslexic", "High Risk", (0, 0, 255)

                # 6. UI Overlays
                cv2.putText(img, f"{res}: {sev}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(img, f"Probability: {round(float(prob), 3)}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.rectangle(img, (10, 10), (img.shape[1]-10, img.shape[0]-10), color, 4)
            except Exception as e:
                pass

        return frame.from_ndarray(img, format="bgr24")

# --- USER INTERFACE ---
st.title("🧠 Live Handwriting Pattern Analyzer")
st.markdown("This system analyzes handwriting stroke patterns in real-time to detect dyslexic indicators.")

# Sidebar age selection
age_val = st.sidebar.slider("Student's Age", 5, 15, 8)
st.sidebar.info("Position the camera directly over the paper for best results.")

if interpreter:
    # Initialize the video streamer
    ctx = webrtc_streamer(
        key="dyslexia-scan",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=DyslexiaAnalyzer,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

    # Update the age parameter in the background processor
    if ctx.video_processor:
        ctx.video_processor.age = age_val
else:
    st.error("AI Model failed to load. Please check your GitHub file list.")
