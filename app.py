import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import time
import os

st.set_page_config(page_title="Dyslexia Risk Analyzer", layout="wide")

# --- 1. Benchmarks & Sentences ---
PRACTICE_SENTENCES = {
    "Age 5-6": "Cat sat on a mat.",
    "Age 7-8": "The big red dog ran fast.",
    "Age 9-10": "Pack my box with five dozen jugs.",
    "Age 11-12": "The quick brown fox jumps over the lazy dog."
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

# --- 2. Model Loading ---
@st.cache_resource
def load_tflite_model():
    model_path = "Improved_Hybrid_AgeModel2.tflite"
    if not os.path.exists(model_path):
        st.error("Model file not found.")
        return None
    try:
        # Full TF library handles the Flex Delegate kernels automatically
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"TFLite Initialization Error: {e}")
        return None

interpreter = load_tflite_model()

# --- 3. Fixed Inference Logic ---
def run_prediction(img_rgba, age_val):
    if interpreter is None: return None
    
    # Preprocessing to fix "Always Severe" error:
    # 1. Strip Alpha channel and convert to Grayscale
    img_gray = cv2.cvtColor(img_rgba.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
    
    # 2. AUTO-INVERT: If background is white (>127), flip it to black for the model
    if np.mean(img_gray) > 127:
        img_gray = cv2.bitwise_not(img_gray)
    
    # 3. Resize and convert to RGB (3-channel) as required by hybrid model
    img_resized = cv2.resize(img_gray, (160, 160))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    
    # 4. Normalize to 0.0 - 1.0
    img_input = (img_rgb / 255.0).astype(np.float32)
    img_input = np.expand_dims(img_input, axis=0)
    
    # 5. Age Input
    age_input = np.array([[age_val]], dtype=np.float32)

    input_details = interpreter.get_input_details()
    for detail in input_details:
        if len(detail['shape']) == 4:
            interpreter.set_tensor(detail['index'], img_input)
        else:
            interpreter.set_tensor(detail['index'], age_input)
    
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0][0]

def get_severity_report(prob, age, duration):
    # Convert 0.0-1.0 probability to 0-100 score
    score = round(float(prob) * 100, 2)
    
    # Apply your specific thresholds
    if score <= 10:
        status, sev, color = "Normal", "Low Risk", "green"
    elif 10 < score <= 30:
        status, sev, color = "Normal", "Mild Risk", "blue"
    elif 30 < score <= 50:
        status, sev, color = "At Risk", "Moderate Risk", "orange"
    else:
        status, sev, color = "At Risk", "Severe Risk", "red"
    
    ref = AGE_REFS.get(age, AGE_REFS[12])
    speed_status = "appropriate" if duration <= ref["time"] else "delayed"
    
    feedback = f"### Diagnostic Insight (Age {age})\n"
    feedback += f"**Developmental Target:** {ref['goal']}\n\n"
    feedback += f"**Result Analysis:** The writing patterns indicate a **{sev}** profile with a score of **{score}/100**. "
    feedback += f"The completion time of **{duration}s** is **{speed_status}** compared to the age benchmark of {ref['time']}s.\n\n"
    
    if score > 30:
        feedback += "The model identified tremors, irregular sizing, or spatial disorientation often associated with dyslexia."
    else:
        feedback += "Handwriting formation aligns closely with standard motor-spatial expectations for this age group."
        
    return status, sev, color, feedback, score

# --- 4. User Interface ---
st.title("🧠 Handwriting Analysis Board")

with st.sidebar:
    st.header("Settings")
    student_age = st.slider("Child's Age", 5, 12, 5)
    
    group = "Age 5-6" if student_age <= 6 else ("Age 7-8" if student_age <= 8 else "Age 9-10" if student_age <= 10 else "Age 11-12")
    st.success(f"**Target Sentence:**\n{PRACTICE_SENTENCES[group]}")
    
    pen_width = st.slider("Pen Thickness", 1, 10, 3)
    st.divider()
    st.info(f"Goal for Age {student_age}: {AGE_REFS[student_age]['goal']}")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Writing Surface")
    canvas_result = st_canvas(
        stroke_width=pen_width, stroke_color="#000", background_color="#FFF",
        height=400, width=650, drawing_mode="freedraw", key="canvas"
    )
    
    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        if "start" not in st.session_state:
            st.session_state.start = time.time()

with col_right:
    st.subheader("Diagnostic Results")
    if st.button("Submit & Predict Dyslexic Risk"):
        if canvas_result.image_data is not None and "start" in st.session_state:
            total_time = round(time.time() - st.session_state.start, 2)
            
            with st.spinner("Analyzing handwriting metrics..."):
                prob = run_prediction(canvas_result.image_data, student_age)
                
            if prob is not None:
                status, sev, color, report, final_score = get_severity_report(prob, student_age, total_time)
                
                st.markdown(f"## Status: :{color}[{status}]")
                st.metric("Risk Level", sev)
                st.metric("Risk Score", f"{final_score}/100")
                st.metric("Time Taken", f"{total_time}s", delta=f"{AGE_REFS[student_age]['time']}s avg", delta_color="inverse")
                
                st.divider()
                st.markdown(report)
                st.progress(min(final_score/100, 1.0))
                
                del st.session_state.start
        else:
            st.warning("Please begin writing on the board first.")

if st.button("Clear Board"):
    if "start" in st.session_state: del st.session_state.start
    st.rerun()
