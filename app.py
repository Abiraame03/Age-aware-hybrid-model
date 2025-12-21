import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import time
import os

st.set_page_config(page_title="Dyslexia Risk Screener (5-12y)", layout="wide")

# --- 1. Age-Appropriate Practice Sentences ---
PRACTICE_SENTENCES = {
    "Age 5-6 (Early)": "Cat sat on a mat.",
    "Age 7-8 (Basic)": "The big red dog ran fast.",
    "Age 9-10 (Intermediate)": "Pack my box with five dozen jugs.",
    "Age 11-12 (Advanced)": "The quick brown fox jumps over the lazy dog."
}

AGE_REFS = {
    5: {"time": 65, "goal": "Basic shape recognition and vertical/horizontal strokes."},
    6: {"time": 55, "goal": "Letter sizing and basic baseline alignment."},
    7: {"time": 48, "goal": "Word spacing and distinguishing 'b' from 'd'."},
    8: {"time": 40, "goal": "Fluidity and connection between letter strokes."},
    9: {"time": 32, "goal": "Automaticity; writing becomes faster and more consistent."},
    10: {"time": 28, "goal": "Legibility maintained at efficient speeds."},
    11: {"time": 24, "goal": "Mature motor control for efficient note-taking."},
    12: {"time": 20, "goal": "Adult-level automaticity and personalized style."}
}

# --- 2. TFLite Model Loading (Flex Op Fix) ---
@st.cache_resource
def load_tflite_model():
    model_path = "Improved_Hybrid_AgeModel2.tflite"
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found.")
        return None
    try:
        # Full TF library is required for Flex Delegate ops
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"TFLite Error: {e}")
        return None

interpreter = load_tflite_model()

# --- 3. Inference Logic ---
def run_prediction(img_rgba, age_val):
    if interpreter is None: return None
    
    # Preprocess Image: RGBA to RGB
    img_rgb = cv2.cvtColor(img_rgba.astype(np.uint8), cv2.COLOR_RGBA2RGB)
    
    # Resize and Normalize
    img_resized = cv2.resize(img_rgb, (160, 160))
    img_input = (img_resized / 255.0).astype(np.float32)
    img_input = np.expand_dims(img_input, axis=0)
    
    # Age Input
    age_input = np.array([[age_val]], dtype=np.float32)

    input_details = interpreter.get_input_details()
    for detail in input_details:
        if "input_1" in detail['name'] or len(detail['shape']) == 4:
            interpreter.set_tensor(detail['index'], img_input)
        else:
            interpreter.set_tensor(detail['index'], age_input)
    
    interpreter.invoke()
    output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0][0]
    return output

def get_diagnostic_report(prob, age, duration):
    # Convert raw probability (0-1) to Risk Score (0-100)
    risk_score = round(float(prob) * 100, 2)
    
    # NEW USER-DEFINED THRESHOLDS
    if risk_score <= 30:
        status, sev, color = "Normal", "Low Risk", "green"
    elif 10 < risk_score <= 40:
        status, sev, color = "Normal", "Mild Risk", "blue"
    elif 30 < risk_score <= 70:
        status, sev, color = "At Risk", "Moderate Risk", "orange"
    else:
        status, sev, color = "At Risk", "Severe Risk", "red"
    
    ref = AGE_REFS.get(age, AGE_REFS[12])
    speed_status = "Appropriate" if duration <= ref["time"] else "Delayed"
    
    feedback = f"### Diagnostic Report (Age {age})\n"
    feedback += f"**Developmental Target:** {ref['goal']}\n\n"
    feedback += f"**Result Analysis:** The writing patterns indicate a **{sev}** profile with a score of **{risk_score}/100**. "
    feedback += f"The completion time of **{duration}s** is considered **{speed_status}** (Avg: {ref['time']}s).\n\n"
    
    if risk_score > 30:
        feedback += "The model identified tremors, irregular letter spacing, or directional reversals often associated with dyslexia."
    else:
        feedback += "Handwriting formation and spatial orientation align closely with developmental milestones."
        
    return status, sev, color, feedback, risk_score

# --- 4. User Interface ---
st.title("🧠 Handwriting Analysis Board")

with st.sidebar:
    st.header("Settings")
    age = st.slider("Child's Age", 5, 12, 8)
    group = "Age 5-6 (Early)" if age <= 6 else ("Age 7-8 (Basic)" if age <= 8 else "Age 9-10 (Intermediate)" if age <= 10 else "Age 11-12 (Advanced)")
    st.success(f"**Target Sentence:**\n{PRACTICE_SENTENCES[group]}")
    
    pen_size = st.slider("Pen Thickness", 1, 10, 3)
    st.divider()
    st.info(f"Goal for Age {age}: {AGE_REFS[age]['goal']}")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Writing Surface")
    canvas_result = st_canvas(
        stroke_width=pen_size, stroke_color="#000", background_color="#FFF",
        height=400, width=650, drawing_mode="freedraw", key="canvas"
    )
    
    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        if "start" not in st.session_state:
            st.session_state.start = time.time()

with col_right:
    st.subheader("Diagnostic Results")
    if st.button("Submit & Predict Dyslexic Risk"):
        if canvas_result.image_data is not None and "start" in st.session_state:
            duration = round(time.time() - st.session_state.start, 2)
            
            with st.spinner("Analyzing handwriting metrics..."):
                prob = run_prediction(canvas_result.image_data, age)
                
            if prob is not None:
                status, sev, color, report, score = get_diagnostic_report(prob, age, duration)
                
                st.markdown(f"## Status: :{color}[{status}]")
                st.metric("Risk Level", sev)
                st.metric("Risk Score", f"{score}/100")
                st.metric("Time Taken", f"{duration}s", delta=f"{AGE_REFS[age]['time']}s avg", delta_color="inverse")
                st.divider()
                st.markdown(report)
                
                # Progress bar for visual score representation
                st.progress(min(score/100, 1.0))
                
                del st.session_state.start
        else:
            st.warning("Please begin writing on the board first.")

if st.button("Clear Board"):
    if "start" in st.session_state: del st.session_state.start
    st.rerun()
