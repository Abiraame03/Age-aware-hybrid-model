import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import time
import os

st.set_page_config(page_title="Dyslexia Risk Analyzer", layout="wide")

# --- Developmental Benchmarks ---
# Reference data for normal handwriting speeds and characteristics by age
AGE_BENCHMARKS = {
    5: {"speed_limit": 60, "desc": "Focus on letter orientation and basic shapes."},
    6: {"speed_limit": 50, "desc": "Consistent sizing and baseline alignment start to form."},
    7: {"speed_limit": 45, "desc": "Improved spacing and cursive transitions may begin."},
    8: {"speed_limit": 40, "desc": "Fluidity increases; focus on consistent slant."},
    9: {"speed_limit": 35, "desc": "Handwriting becomes more automatic and faster."},
    10: {"speed_limit": 30, "desc": "Strong legibility with personalized style emerging."},
    11: {"speed_limit": 25, "desc": "Adult-like speed and automaticity expected."},
    12: {"speed_limit": 20, "desc": "Efficient note-taking speed achieved."},
    13: {"speed_limit": 18, "desc": "Stable and rapid handwriting patterns."},
    14: {"speed_limit": 15, "desc": "Highly consistent and fast motor execution."},
    15: {"speed_limit": 15, "desc": "Mature handwriting with high automaticity."}
}

# --- Model Loading with Flex Op Fix ---
@st.cache_resource
def load_tflite_model():
    model_path = "Improved_Hybrid_AgeModel2.tflite"
    if not os.path.exists(model_path):
        return None
    try:
        # Full TensorFlow import usually registers Flex Ops automatically.
        # We initialize the interpreter here.
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Initialization Error: {e}. Ensure 'tensorflow' (not tflite-runtime) is installed.")
        return None

interpreter = load_tflite_model()

# --- Logic & Processing ---
def generate_personalized_report(prob, age, time_taken):
    """Custom feedback based on model probability, age benchmarks, and time."""
    benchmark = AGE_BENCHMARKS.get(age, {"speed_limit": 30, "desc": "Standard development."})
    
    # Speed Analysis
    is_slow = time_taken > benchmark["speed_limit"]
    speed_feedback = "The writing speed is slower than typical for this age group." if is_slow else "The writing speed is within the expected range."
    
    # Severity & Feedback
    if prob < 0.25:
        status, color = "No Risk", "green"
        insight = f"Excellent! At age {age}, the handwriting follows typical developmental markers: {benchmark['desc']}"
    elif prob < 0.50:
        status, color = "Low Risk", "blue"
        insight = f"Normal variation for a {age}-year-old. {speed_feedback}"
    elif prob < 0.75:
        status, color = "Moderate Risk", "orange"
        insight = f"Some dyslexic markers detected. For age {age}, we expect {benchmark['desc']}. The patterns suggest spatial processing delays."
    else:
        status, color = "High Risk", "red"
        insight = f"Significant dyslexic indicators found. {speed_feedback} This level of difficulty at age {age} suggests a need for professional evaluation."

    return status, color, insight

def run_inference(img_array, age_val):
    if interpreter is None: return None
    
    # Preprocess Image: Resize to 160x160 and handle Alpha channel from canvas
    img_resized = cv2.resize(img_array, (160, 160))
    if img_resized.shape[-1] == 4: # Remove alpha channel if present
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGBA2RGB)
    
    img_norm = (img_resized / 255.0).astype(np.float32)
    img_input = np.expand_dims(img_norm, axis=0)
    
    # Preprocess Age
    age_input = np.array([[age_val]], dtype=np.float32)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Dynamic Assignment for Hybrid Model
    for detail in input_details:
        if len(detail['shape']) == 4:
            interpreter.set_tensor(detail['index'], img_input)
        else:
            interpreter.set_tensor(detail['index'], age_input)
    
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0][0]

# --- UI Layout ---
st.title("✍️ Dyslexia Handwriting Analysis Board")
st.write("Please use a stylus to write the phrase: **'The quick brown fox'**")

# Sidebar
st.sidebar.header("Student Profile")
age = st.sidebar.slider("Student Age", 5, 15, 8)
stroke_width = st.sidebar.slider("Pen Thickness", 1, 10, 3)

if 'start_time' not in st.session_state:
    st.session_state.start_time = None

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Interactive Writing Board")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=stroke_width,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=450,
        width=700,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    # Start timer when the first stroke is made
    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        if st.session_state.start_time is None:
            st.session_state.start_time = time.time()

with col2:
    st.subheader("Diagnostic Results")
    if st.button("Submit for Analysis"):
        if canvas_result.image_data is not None and st.session_state.start_time is not None:
            time_taken = round(time.time() - st.session_state.start_time, 2)
            prob = run_inference(canvas_result.image_data, age)
            
            if prob is not None:
                status, color, feedback = generate_personalized_report(prob, age, time_taken)
                
                st.markdown(f"### Status: :{color}[{status}]")
                st.metric("Completion Time", f"{time_taken}s", help=f"Expected for age {age}: <{AGE_BENCHMARKS[age]['speed_limit']}s")
                st.progress(float(prob))
                
                st.info(f"**Personalized Insight:**\n\n{feedback}")
                
                # Cleanup for next attempt
                st.session_state.start_time = None
            else:
                st.error("Model failure. Ensure the .tflite file is valid.")
        else:
            st.warning("No handwriting detected. Please write on the board.")

if st.button("Reset Board"):
    st.session_state.start_time = None
    st.rerun()
