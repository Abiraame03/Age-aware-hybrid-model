import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import time
import os

st.set_page_config(page_title="Dyslexia Risk Analyzer", layout="wide")

# --- Model Loading ---
@st.cache_resource
def load_tflite_model():
    model_path = "Improved_Hybrid_AgeModel2.tflite"
    if not os.path.exists(model_path):
        return None
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Model Error: {e}")
        return None

interpreter = load_tflite_model()

# --- Logic & Processing ---
def get_personalized_feedback(prob, age, time_taken):
    """Generates custom feedback based on age, speed, and probability."""
    # Logic for expected time: younger kids take longer
    expected_time = 30 if age < 8 else 15 
    speed_status = "slow" if time_taken > expected_time else "steady"
    
    if prob < 0.25:
        return f"Excellent! For a {age}-year-old, the handwriting flow is {speed_status} and shows strong motor control. No dyslexic markers detected."
    elif prob < 0.50:
        return f"Normal patterns observed. The {time_taken}s completion time is within the expected range for age {age}. Minor irregularities are likely developmental."
    elif prob < 0.75:
        return f"Moderate risk detected. At age {age}, the combination of a {speed_status} pace and specific stroke patterns suggests possible spatial orientation challenges. Consider a professional screening."
    else:
        return f"High-risk markers identified. The completion time of {time_taken}s combined with pattern analysis indicates significant difficulty with letter formation. We recommend consulting an educational specialist."

def run_inference(img_array, age_val):
    if interpreter is None: return None
    
    # Preprocess Image (160x160)
    img_resized = cv2.resize(img_array, (160, 160))
    # Convert to RGB if needed and normalize
    img_norm = (img_resized[:, :, :3] / 255.0).astype(np.float32)
    img_input = np.expand_dims(img_norm, axis=0)
    
    # Preprocess Age
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

# --- UI Layout ---
st.title("🧠 Digital Writing Analysis Board")
st.markdown("Write the sentence: **'The quick brown fox'** or draw shapes below.")

# Sidebar Settings
st.sidebar.header("Student Profile")
age = st.sidebar.slider("Student Age", 5, 15, 8)
stroke_width = st.sidebar.slider("Stroke width: ", 1, 10, 3)

# Timer logic
if 'start_time' not in st.session_state:
    st.session_state.start_time = None

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Writing Canvas")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=400,
        width=600,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    # Start timer on first stroke
    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        if st.session_state.start_time is None:
            st.session_state.start_time = time.time()

with col2:
    st.subheader("Analysis")
    if st.button("Submit & Analyze"):
        if canvas_result.image_data is not None and st.session_state.start_time is not None:
            # Calculate Time
            end_time = time.time()
            time_taken = round(end_time - st.session_state.start_time, 2)
            
            # Run Model
            prob = run_inference(canvas_result.image_data, age)
            
            if prob is not None:
                # Severity Logic
                if prob < 0.5:
                    st.success(f"Result: Normal Range")
                else:
                    st.error(f"Result: Dyslexic Indicators Found")
                
                # Metrics
                st.metric("Completion Time", f"{time_taken} seconds")
                st.metric("Probability", f"{round(float(prob)*100, 2)}%")
                
                # Feedback
                st.info("**Personalized Feedback:**")
                feedback = get_personalized_feedback(prob, age, time_taken)
                st.write(feedback)
                
                # Reset timer for next run
                st.session_state.start_time = None
            else:
                st.error("Model Error: Could not run inference.")
        else:
            st.warning("Please write something on the board first!")

if st.button("Clear Canvas"):
    st.session_state.start_time = None
    st.rerun()
