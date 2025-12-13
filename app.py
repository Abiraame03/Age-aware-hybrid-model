import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import tempfile
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Dyslexia Risk Analyzer", layout="centered")

# --- MODEL LOADING ---
@st.cache_resource
def load_tflite_model():
    model_path = "Improved_Hybrid_AgeModel.tflite"
    if not os.path.exists(model_path):
        return None
    try:
        # Load interpreter with Flex delegate support for BiLSTM
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.sidebar.error(f"Allocation Error: {e}")
        return None

interpreter = load_tflite_model()

def run_inference(img_array, age_val):
    if interpreter is None:
        return None
    
    # 1. Preprocess Image (160x160)
    img_resized = cv2.resize(img_array, (160, 160))
    img_norm = (img_resized / 255.0).astype(np.float32)
    img_input = np.expand_dims(img_norm, axis=0)
    
    # 2. Preprocess Age
    age_input = np.array([[age_val]], dtype=np.float32)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 3. Assign Tensors
    for detail in input_details:
        if len(detail['shape']) == 4:
            interpreter.set_tensor(detail['index'], img_input)
        else:
            interpreter.set_tensor(detail['index'], age_input)
    
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0][0]

def get_severity(prob):
    """Maps probability to severity levels."""
    if prob < 0.25:
        return "Normal", "No Risk", "green"
    elif prob < 0.50:
        return "Normal", "Low Risk", "blue"
    elif prob < 0.75:
        return "Dyslexic", "Moderate Risk", "orange"
    else:
        return "Dyslexic", "High Risk", "red"

# --- UI ---
st.title("🧠 Dyslexia Handwriting Analysis")
st.markdown("### Risk Assessment & Severity Level")

if interpreter is None:
    st.error("Model Error: Failed to allocate memory. Ensure the model was converted with 'SELECT_TF_OPS' in Colab.")
else:
    with st.sidebar:
        st.header("Settings")
        mode = st.radio("Input Type", ["Photo", "Video"])
        age = st.slider("Student Age", 5, 15, 8)

    if mode == "Photo":
        file = st.file_uploader("Upload handwriting photo", type=['jpg', 'png', 'jpeg'])
        if file:
            img = Image.open(file)
            st.image(img, use_column_width=True)
            
            if st.button("Analyze Photo"):
                prob = run_inference(np.array(img.convert("RGB")), age)
                if prob is not None:
                    res, sev, color = get_severity(prob)
                    st.divider()
                    st.subheader(f"Condition: :{color}[{res}]")
                    st.markdown(f"**Severity Level:** {sev}")
                    st.progress(float(prob))
                    st.write(f"Confidence Score: {round(float(prob)*100, 2)}%")

    else:
        file = st.file_uploader("Upload writing video", type=['mp4', 'mov', 'avi'])
        if file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(file.read())
            st.video(file)
            
            if st.button("Analyze Video"):
                cap = cv2.VideoCapture(tfile.name)
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 2))
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    prob = run_inference(frame_rgb, age)
                    res, sev, color = get_severity(prob)
                    st.success(f"Analysis Complete: {res} ({sev})")
                    st.write(f"Probability: {round(float(prob), 4)}")
                cap.release()
                os.unlink(tfile.name)
