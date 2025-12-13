import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="Dyslexia Risk Analyzer", layout="centered")

@st.cache_resource
def load_tflite_model():
    model_path = "Improved_Hybrid_AgeModel2.tflite"
    if not os.path.exists(model_path):
        return None
    try:
        # Load the interpreter
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        # This will show the specific 'FULLY_CONNECTED' error in the UI if it persists
        st.error(f"Detailed Model Error: {e}")
        return None

interpreter = load_tflite_model()

def get_severity_info(prob):
    """Maps probability to categorical severity."""
    if prob < 0.25:
        return "Normal", "No Risk", "green", "Handwriting follows standard patterns."
    elif prob < 0.50:
        return "Normal", "Low Risk", "blue", "Minor variations detected; within normal range."
    elif prob < 0.75:
        return "Dyslexic", "Moderate Risk", "orange", "Pattern indicates moderate dyslexic markers."
    else:
        return "Dyslexic", "High Risk", "red", "Strong dyslexic indicators found. Consultation recommended."

def run_inference(img_array, age_val):
    if interpreter is None: return None
    
    # Preprocess Image (160x160)
    img_resized = cv2.resize(img_array, (160, 160))
    img_norm = (img_resized / 255.0).astype(np.float32)
    img_input = np.expand_dims(img_norm, axis=0)
    
    # Preprocess Age
    age_input = np.array([[age_val]], dtype=np.float32)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Dynamic Assignment
    for detail in input_details:
        if len(detail['shape']) == 4:
            interpreter.set_tensor(detail['index'], img_input)
        else:
            interpreter.set_tensor(detail['index'], age_input)
    
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0][0]

st.title("🧠 Dyslexia Handwriting Pattern Analyzer")

if interpreter:
    mode = st.sidebar.radio("Input Method", ["Photo Upload", "Video Upload"])
    age = st.sidebar.slider("Student Age", 5, 15, 8)

    if mode == "Photo Upload":
        file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
        if file and st.button("Analyze"):
            img = Image.open(file)
            st.image(img, use_column_width=True)
            prob = run_inference(np.array(img.convert("RGB")), age)
            if prob is not None:
                res, sev, color, desc = get_severity_info(prob)
                st.markdown(f"### Status: :{color}[{res} - {sev}]")
                st.info(desc)
                st.progress(float(prob))
                st.write(f"Probability: {round(float(prob)*100, 2)}%")
    else:
        file = st.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'])
        if file and st.button("Analyze Video"):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(file.read())
            cap = cv2.VideoCapture(tfile.name)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 2))
            ret, frame = cap.read()
            if ret:
                prob = run_inference(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), age)
                res, sev, color, _ = get_severity_info(prob)
                st.success(f"Result: {res} ({sev})")
            cap.release()
            os.unlink(tfile.name)
else:
    st.warning("Please upload the re-converted TFLite model to your GitHub repository.")
