import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import time
import os

st.set_page_config(page_title="Dyslexia Risk Analyzer", layout="wide")

# --- Age-Based Reference Benchmarks ---
# Benchmarks represent typical handwriting speed (seconds) and developmental milestones
AGE_REFS = {
    5: {"time": 60, "goal": "Basic stroke orientation and shape recognition."},
    6: {"time": 52, "goal": "Letter sizing and basic baseline alignment."},
    7: {"time": 45, "goal": "Word spacing and consistent letter formation."},
    8: {"time": 38, "goal": "Fluidity and connection between letters."},
    9: {"time": 32, "goal": "Automaticity in letter production."},
    10: {"time": 28, "goal": "Legibility at increased writing speeds."},
    11: {"time": 24, "goal": "Personalized style and efficient speed."},
    12: {"time": 20, "goal": "Adult-level motor control and speed."},
}

@st.cache_resource
def load_tflite_model():
    model_path = "Improved_Hybrid_AgeModel2.tflite"
    if not os.path.exists(model_path):
        st.error("Model file not found!")
        return None
    try:
        # To fix the Flex Op error, we use the standard tf.lite.Interpreter. 
        # When 'tensorflow' is imported, it automatically registers Flex delegates.
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Flex Delegate Error: {e}")
        return None

interpreter = load_tflite_model()

def get_custom_feedback(prob, age, elapsed_time):
    ref = AGE_REFS.get(age, AGE_REFS[12])
    speed_status = "appropriate" if elapsed_time <= ref["time"] else "delayed"
    
    if prob < 0.3:
        summary = "Strong Performance"
        details = f"For age {age}, the writing speed ({elapsed_time}s) is {speed_status}. No significant dyslexic markers detected."
    elif prob < 0.7:
        summary = "Moderate Observation"
        details = f"At age {age}, we look for {ref['goal']}. The current speed is {speed_status}, and some pattern inconsistencies suggest a moderate risk."
    else:
        summary = "High Priority"
        details = f"The writing patterns and {speed_status} speed indicate high-risk markers. Further educational screening is recommended."
    
    return summary, details

def run_inference(img, age_val):
    if interpreter is None: return None
    
    # Image Preprocessing (160x160)
    # Removing alpha channel if it exists from the canvas
    img_rgb = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGBA2RGB)
    img_resized = cv2.resize(img_rgb, (160, 160))
    img_final = (img_resized / 255.0).astype(np.float32)
    img_input = np.expand_dims(img_final, axis=0)
    
    # Age Preprocessing
    age_input = np.array([[age_val]], dtype=np.float32)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for detail in input_details:
        if len(detail['shape']) == 4:
            interpreter.set_tensor(detail['index'], img_input)
        else:
            interpreter.set_tensor(detail['index'], age_input)
    
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0][0]

# --- UI Interface ---
st.title("🧠 Neuro-Writing Analysis Board")
st.markdown("### Stylus/Touch Input Mode")

with st.sidebar:
    st.header("Student Parameters")
    student_age = st.slider("Select Age", 5, 12, 8)
    pen_size = st.slider("Pen Thickness", 1, 15, 3)
    st.info(f"Target goal for age {student_age}: {AGE_REFS[student_age]['goal']}")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Writing Surface")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=pen_size,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=400,
        width=600,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        if "start_time" not in st.session_state:
            st.session_state.start_time = time.time()

with col_right:
    st.subheader("Diagnostic Report")
    if st.button("Generate Prediction"):
        if canvas_result.image_data is not None and "start_time" in st.session_state:
            total_time = round(time.time() - st.session_state.start_time, 2)
            prediction = run_inference(canvas_result.image_data, student_age)
            
            if prediction is not None:
                title, feedback = get_custom_feedback(prediction, student_age, total_time)
                st.metric("Writing Speed", f"{total_time}s")
                st.metric("Risk Probability", f"{round(float(prediction)*100, 1)}%")
                st.markdown(f"**Result:** {title}")
                st.write(feedback)
                
                # Reset timer
                del st.session_state.start_time
        else:
            st.warning("Please draw on the board before submitting.")

if st.button("Clear Canvas"):
    if "start_time" in st.session_state:
        del st.session_state.start_time
    st.rerun()
