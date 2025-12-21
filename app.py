import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import time
import os

st.set_page_config(page_title="Dyslexia Risk Analyzer", layout="wide")

# --- 1. Age-Based Handwriting Benchmarks ---
# Reference data for normal handwriting speeds and developmental goals
AGE_REFS = {
    5: {"time": 60, "goal": "Basic stroke orientation and shape recognition."},
    6: {"time": 52, "goal": "Consistent letter sizing and baseline alignment."},
    7: {"time": 45, "goal": "Word spacing and consistent letter formation."},
    8: {"time": 38, "goal": "Fluidity and connection between letters."},
    9: {"time": 32, "goal": "Automaticity in letter production."},
    10: {"time": 28, "goal": "Legibility at increased writing speeds."},
    11: {"time": 24, "goal": "Mature motor control and efficient speed."},
    12: {"time": 20, "goal": "Adult-level motor control and speed."},
}

# --- 2. Model Loading (Modified for TFLite Flex Support) ---
@st.cache_resource
def load_tflite_model():
    model_path = "Improved_Hybrid_AgeModel2.tflite"
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found in repository.")
        return None
    try:
        # Standard Interpreter handles Flex Ops when tensorflow-cpu is installed
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Flex Delegate Error: {e}")
        return None

interpreter = load_tflite_model()

# --- 3. Prediction & Severity Logic ---

def run_prediction(img_rgba, age_val):
    """Step 1: Predict with the TFLite model."""
    if interpreter is None: return None
    
    # Preprocess Image: RGBA -> RGB -> 160x160 -> Normalize
    img_rgb = cv2.cvtColor(img_rgba.astype(np.uint8), cv2.COLOR_RGBA2RGB)
    img_resized = cv2.resize(img_rgb, (160, 160))
    img_input = (img_resized / 255.0).astype(np.float32)
    img_input = np.expand_dims(img_input, axis=0)
    
    # Preprocess Age Input
    age_input = np.array([[age_val]], dtype=np.float32)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Dynamic Assignment for Hybrid Inputs
    for detail in input_details:
        if len(detail['shape']) == 4:
            interpreter.set_tensor(detail['index'], img_input)
        else:
            interpreter.set_tensor(detail['index'], age_input)
    
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0][0]

def get_severity(prob):
    """Step 2: Map probability to categorical severity."""
    if prob < 0.25:
        return "Normal", "No Risk", "green"
    elif prob < 0.50:
        return "Normal", "Low Risk", "blue"
    elif prob < 0.75:
        return "Dyslexic", "Moderate Risk", "orange"
    else:
        return "Dyslexic", "High Risk", "red"

def generate_personalized_feedback(status, severity, age, duration):
    """Step 3: Justify prediction with Age and Time parameters."""
    ref = AGE_REFS.get(age, AGE_REFS[12])
    speed_status = "appropriate" if duration <= ref["time"] else "slower than typical"
    
    feedback = f"### Personalized Report (Age {age})\n"
    feedback += f"**Benchmark Goal:** {ref['goal']}\n\n"
    
    if status == "Normal":
        feedback += f"The student shows healthy motor flow. The speed of **{duration}s** is {speed_status} (Target: ~{ref['time']}s). "
    else:
        feedback += f"Analysis detected **{severity}** dyslexic markers. The speed of **{duration}s** is {speed_status} for this age. "
        feedback += "Pattern inconsistencies suggest specific difficulty with spatial letter orientation."
    
    return feedback

# --- 4. User Interface ---
st.title("✍️ Dyslexia Handwriting Board")
st.markdown("Write the sentence using a stylus: **'witty mouse'**")

with st.sidebar:
    st.header("Student Profile")
    student_age = st.slider("Select Age", 5, 12, 8)
    pen_width = st.slider("Pen Thickness", 1, 15, 3)
    st.divider()
    st.info(f"Target for Age {student_age}: {AGE_REFS[student_age]['goal']}")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Interactive Canvas")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=pen_width,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=400,
        width=600,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    # Start timer when first stroke is drawn
    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        if "start_time" not in st.session_state:
            st.session_state.start_time = time.time()

with col2:
    st.subheader("Diagnostic Results")
    if st.button("Submit for Analysis"):
        if canvas_result.image_data is not None and "start_time" in st.session_state:
            total_time = round(time.time() - st.session_state.start_time, 2)
            
            # 1. PREDICT
            prob = run_prediction(canvas_result.image_data, student_age)
            
            if prob is not None:
                # 2. SEVERITY
                status, severity, color = get_severity(prob)
                
                st.markdown(f"### Status: :{color}[{status} - {severity}]")
                st.metric("Writing Speed", f"{total_time}s")
                st.metric("Risk Prob.", f"{round(float(prob)*100, 1)}%")
                
                # 3. FEEDBACK (Age/Time justified)
                st.divider()
                st.markdown(generate_personalized_feedback(status, severity, student_age, total_time))
                
                # Cleanup
                del st.session_state.start_time
        else:
            st.warning("Please draw on the canvas first.")

if st.button("Clear Board"):
    if "start_time" in st.session_state:
        del st.session_state.start_time
    st.rerun()
