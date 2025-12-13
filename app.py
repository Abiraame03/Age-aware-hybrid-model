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
        # Load TFLite interpreter
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.sidebar.error(f"Model Error: {e}")
        return None

interpreter = load_tflite_model()

def run_inference(img_array, age_val):
    if interpreter is None: return None
    
    # 1. Image Preprocessing (160x160)
    img_resized = cv2.resize(img_array, (160, 160))
    img_norm = (img_resized / 255.0).astype(np.float32)
    img_input = np.expand_dims(img_norm, axis=0)
    
    # 2. Age Preprocessing
    age_input = np.array([[age_val]], dtype=np.float32)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 3. Assign Tensors by Shape
    for detail in input_details:
        if len(detail['shape']) == 4:
            interpreter.set_tensor(detail['index'], img_input)
        else:
            interpreter.set_tensor(detail['index'], age_input)
    
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0][0]

def get_severity_info(prob):
    """Categorizes the probability into severity levels."""
    if prob < 0.25:
        return "Normal", "No Risk", "green", "No dyslexic indicators detected."
    elif prob < 0.50:
        return "Normal", "Low Risk", "blue", "Minor variations; patterns appear normal for this age."
    elif prob < 0.75:
        return "Dyslexic", "Moderate Risk", "orange", "Some indicators found. Suggest further screening."
    else:
        return "Dyslexic", "High Risk", "red", "Strong indicators detected. Professional evaluation advised."

# --- USER INTERFACE ---
st.title("🧠 Dyslexia Handwriting Pattern Analyzer")

if interpreter is None:
    st.error("Model Error: Failed to allocate tensors. Please ensure you have pushed 'Improved_Hybrid_AgeModel.tflite' to your repo.")
else:
    with st.sidebar:
        st.header("Settings")
        mode = st.radio("Input Method", ["Photo Upload", "Video Upload"])
        age = st.slider("Student Age", 5, 15, 8)
        st.divider()
        st.caption("Engine: Hybrid CNN-BiLSTM TFLite")

    if mode == "Photo Upload":
        file = st.file_uploader("Upload handwriting photo", type=['jpg', 'png', 'jpeg'])
        if file:
            img = Image.open(file)
            st.image(img, use_column_width=True)
            if st.button("Run Analysis"):
                with st.spinner("Analyzing handwriting patterns..."):
                    prob = run_inference(np.array(img.convert("RGB")), age)
                    if prob is not None:
                        res, sev, color, desc = get_severity_info(prob)
                        st.divider()
                        st.markdown(f"### Status: :{color}[{res}]")
                        st.markdown(f"**Severity Level:** {sev}")
                        st.info(desc)
                        st.progress(float(prob))
                        st.write(f"Confidence Score: {round(float(prob)*100, 2)}%")

    else:
        file = st.file_uploader("Upload writing video", type=['mp4', 'mov', 'avi'])
        if file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(file.read())
            st.video(file)
            if st.button("Analyze Video"):
                with st.spinner("Processing video frames..."):
                    cap = cv2.VideoCapture(tfile.name)
                    # Sample middle frame for stability
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 2))
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        prob = run_inference(frame_rgb, age)
                        res, sev, color, desc = get_severity_info(prob)
                        st.divider()
                        st.markdown(f"### Status: :{color}[{res}]")
                        st.markdown(f"**Severity Level:** {sev}")
                        st.write(f"Probability: {round(float(prob), 4)}")
                    cap.release()
                    os.unlink(tfile.name)
