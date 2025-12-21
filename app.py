import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import time
import os

st.set_page_config(page_title="Dyslexia Risk Analyzer", layout="wide")

# --- 1. Age-Based Benchmarks ---
# Reference for normal handwriting speed and developmental milestones
AGE_REFS = {
    5: {"avg_time": 60, "goal": "Basic stroke orientation and shape recognition."},
    6: {"avg_time": 52, "goal": "Consistent letter sizing and baseline alignment."},
    7: {"avg_time": 45, "goal": "Improved word spacing and letter formation."},
    8: {"avg_time": 38, "goal": "Fluidity and connection between letters."},
    9: {"avg_time": 32, "goal": "Automaticity in letter production."},
    10: {"avg_time": 28, "goal": "Legibility at increased writing speeds."},
    11: {"avg_time": 24, "goal": "Mature motor control and efficient speed."},
    12: {"avg_time": 20, "goal": "Adult-level handwriting automaticity."}
}

# --- 2. TFLite Model Loading (Flex Delegate Fix) ---
@st.cache_resource
def load_tflite_model():
    model_path = "Improved_Hybrid_AgeModel2.tflite"
    if not os.path.exists(model_path):
        st.error(f"Error: '{model_path}' not found.")
        return None
    try:
        # Full tensorflow import automatically enables Flex delegates for TFLite
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Flex Delegate Error: {e}")
        st.info("Check requirements.txt: You MUST use 'tensorflow', not 'tflite-runtime'.")
        return None

interpreter = load_tflite_model()

# --- 3. Processing Logic ---

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

    # Assign Tensors
    for detail in input_details:
        if len(detail['shape']) == 4:
            interpreter.set_tensor(detail['index'], img_input)
        else:
            interpreter.set_tensor(detail['index'], age_input)
    
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0][0]

def get_severity(prob):
    """Step 2: Assign severity based on model output."""
    if prob < 0.25:
        return "Normal", "No Risk", "green"
    elif prob < 0.50:
        return "Normal", "Low Risk", "blue"
    elif prob < 0.75:
        return "Dyslexic", "Moderate Risk", "orange"
    else:
        return "Dyslexic", "High Risk", "red"

def generate_custom_feedback(status, severity, age, elapsed_time):
    """Step 3: Proceed with Age/Time parameters for personalized feedback."""
    ref = AGE_REFS.get(age, AGE_REFS[12])
    speed_status = "appropriate" if elapsed_time <= ref["avg_time"] else "slower than average"
    
    feedback = f"### Personalized Analysis for Age {age}\n"
    feedback += f"**Developmental Context:** At this age, the goal is: *{ref['goal']}*.\n\n"
    
    if status == "Normal":
        feedback += f"The student's handwriting patterns are healthy. The completion time of **{elapsed_time}s** is {speed_status} (Benchmark: {ref['avg_time']}s)."
    else:
        feedback += f"The analysis detected **{severity}** indicators. The writing speed was **{elapsed_time}s**, which is {speed_status}. "
        feedback += "This combination suggests difficulty with motor-spatial coordination or letter-form recall."
    
    return feedback

# --- 4. User Interface ---
st.title("🧠 Neuro-Writing Diagnostic Board")
st.markdown("Write: **'The quick brown fox'** on the board below.")

with st.sidebar:
    st.header("Student Settings")
    age = st.slider("Select Age", 5, 12, 8)
    brush = st.slider("Pen Thickness", 1, 10, 3)
    st.divider()
    st.info(f"Target for age {age}: {AGE_REFS[age]['goal']}")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Writing Canvas")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=brush,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=400,
        width=600,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    # Start timer on first stroke
    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        if "start_time" not in st.session_state:
            st.session_state.start_time = time.time()

with col2:
    st.subheader("Report")
    if st.button("Submit Analysis"):
        if canvas_result.image_data is not None and "start_time" in st.session_state:
            # Step 1 & 2: Prediction and Severity
            duration = round(time.time() - st.session_state.start_time, 2)
            prob = run_prediction(canvas_result.image_data, age)
            
            if prob is not None:
                status, severity, color = get_severity(prob)
                
                # Display Severity Result
                st.markdown(f"### Status: :{color}[{status} - {severity}]")
                st.progress(float(prob))
                
                # Step 3: Detailed Feedback with Age/Time
                st.divider()
                report = generate_custom_feedback(status, severity, age, duration)
                st.markdown(report)
                
                # Clear session
                del st.session_state.start_time
        else:
            st.warning("Please write on the board first.")

if st.button("Reset Board"):
    if "start_time" in st.session_state:
        del st.session_state.start_time
    st.rerun()
