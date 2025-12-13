import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import tempfile
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Dyslexia AI Analyzer", layout="wide")

# --- MODEL LOADING ---
@st.cache_resource
def load_tflite_model():
    model_path = "Improved_Hybrid_AgeModel.tflite"
    if not os.path.exists(model_path):
        st.error("Model file not found!")
        return None
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

def predict(img_array, age, interpreter):
    # Preprocess Image
    img_resized = cv2.resize(img_array, (160, 160))
    img_norm = (img_resized / 255.0).astype(np.float32)
    img_input = np.expand_dims(img_norm, axis=0)
    
    # Preprocess Age
    age_input = np.array([[age]], dtype=np.float32)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Map Inputs
    for detail in input_details:
        if len(detail['shape']) == 4:
            interpreter.set_tensor(detail['index'], img_input)
        else:
            interpreter.set_tensor(detail['index'], age_input)
    
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0][0]

# --- UI ---
st.title("🧠 Dyslexia Handwriting Pattern Analyzer")
analysis_mode = st.sidebar.selectbox("Choose Input Mode", ["Upload Photo", "Upload Video"])
age = st.sidebar.slider("Student Age", 5, 15, 8)

if analysis_mode == "Upload Photo":
    uploaded_file = st.file_uploader("Upload handwriting image...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        img_array = np.array(image.convert("RGB"))
        st.image(image, caption="Uploaded Image", width=400)
        
        if st.button("Analyze Image"):
            with st.spinner("Analyzing..."):
                prob = predict(img_array, age, interpreter)
                st.subheader(f"Risk Score: {round(float(prob)*100, 1)}%")
                if prob > 0.5:
                    st.error("Result: High Indicator of Dyslexic Patterns")
                else:
                    st.success("Result: Patterns appear Normal")

elif analysis_mode == "Upload Video":
    uploaded_video = st.file_uploader("Upload a video of writing...", type=["mp4", "mov", "avi"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st.video(uploaded_video)
        
        if st.button("Analyze Video Pattern"):
            # Get the middle frame of the video for analysis
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            ret, frame = cap.read()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                prob = predict(frame_rgb, age, interpreter)
                st.write(f"Analysis complete on sample frame...")
                st.progress(float(prob))
                st.write(f"Probability: {round(float(prob), 4)}")
            cap.release()
