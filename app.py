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
        return None
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except:
        return None

interpreter = load_tflite_model()

def run_inference(img_array, age):
    if interpreter is None:
        return None
    
    # 1. Image Preprocessing
    img_resized = cv2.resize(img_array, (160, 160))
    img_norm = (img_resized / 255.0).astype(np.float32)
    img_input = np.expand_dims(img_norm, axis=0)
    
    # 2. Age Preprocessing
    age_input = np.array([[age]], dtype=np.float32)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 3. Dynamic Tensor Assignment
    for detail in input_details:
        if len(detail['shape']) == 4:
            interpreter.set_tensor(detail['index'], img_input)
        else:
            interpreter.set_tensor(detail['index'], age_input)
    
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0][0]

# --- UI ---
st.title("🧠 Dyslexia Handwriting Pattern Analyzer")
st.info("Upload a photo or video of handwriting to analyze the pattern.")

with st.sidebar:
    mode = st.radio("Input Method", ["Image Upload", "Video Upload"])
    age = st.slider("Child's Age", 5, 15, 8)
    st.divider()
    st.write("Model: Hybrid CNN-BiLSTM TFLite")

if interpreter is None:
    st.error("Model Error: The system could not allocate memory for the AI model. Please check the 'runtime.txt' is set to python-3.11.")
else:
    if mode == "Image Upload":
        file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
        if file:
            img = Image.open(file)
            st.image(img, width=400)
            if st.button("Analyze Pattern"):
                prob = run_inference(np.array(img.convert("RGB")), age)
                if prob is not None:
                    res = "Dyslexic Pattern Detected" if prob > 0.5 else "Normal Pattern"
                    st.metric("Probability", f"{round(float(prob)*100, 1)}%", delta=res)

    else:
        file = st.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'])
        if file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(file.read())
            st.video(file)
            
            if st.button("Extract & Analyze"):
                cap = cv2.VideoCapture(tfile.name)
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 2))
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    prob = run_inference(frame_rgb, age)
                    st.success(f"Video Analysis Complete. Confidence: {round(float(prob), 3)}")
                cap.release()
