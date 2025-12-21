import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import time
import os

st.set_page_config(page_title="Dyslexia Risk Analyzer", layout="wide")

# --- 1. Age-Based Handwriting Benchmarks ---
# Reference data for normal handwriting speeds and goals by age
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

# --- 2. TFLite Model Loading (Fixing Flex Op Error) ---
@st.cache_resource
def load_tflite_model():
    model_path = "Improved_Hybrid_AgeModel2.tflite"
    if not os.path.exists(model_path):
        st.error("Model file not found!")
        return None
    try:
        # Full tensorflow import registers the Flex Delegate kernels automatically.
        # This handles operations like FlexTensorListReserve.
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        st.info("Check your requirements.txt: You MUST use 'tensorflow', not 'tflite-runtime'.")
        return None

interpreter = load_tflite_model()

# --- 3. Processing & Logic ---

def run_prediction(img_rgba, age_val):
    """Step 1: Get raw probability from TFLite model."""
    if interpreter is None: return None
    
    # Preprocess Image: RGBA -> RGB -> 160x160 -> Float32
    img_rgb = cv2.cvtColor(img_rgba.astype(np.uint8), cv2.COLOR_RGBA2RGB)
    img_resized = cv2.resize(img_rgb, (160, 160))
    img_input = (img_resized / 255.0).astype(np.float32)
    img_input = np.expand_dims(img_input, axis=0)
    
    # Preprocess Age Input
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

def generate_report(status, severity, age, elapsed_time):
    """Step 3: Integrate Age and Time for personalized feedback."""
    ref = AGE_REFS.get(age, AGE_REFS[12])
    speed_status = "appropriate" if elapsed_time <= ref["time"] else "slower than expected"
    
    feedback = f"### Personalized Report for Age {age}\n"
    feedback += f"**Benchmark Goal:** {ref['goal']}\n\n"
    
    if status == "Normal":
        feedback += f"The student shows healthy motor patterns. The writing speed of **{elapsed_time}s** is {speed_status} (Avg: {ref['time']}s). "
        feedback += "The stroke connection is consistent with standard developmental progress."
    else:
        feedback += f"The analysis detected **{severity}** dyslexic indicators. "
        feedback += f"The writing speed of **{elapsed_time}s** is {speed_status} for this age group. "
        feedback += "The feedback suggests specific difficulty with letter-form spatial orientation and motor fluidity."
    
    return feedback

# --- 4. User Interface ---
st.title("🧠 Neuro-Writing Analysis Board")
st.write("Please write the phrase: **'The quick brown fox'**")

with st.sidebar:
    st.header("Student Parameters")
    age = st.slider("Select Age", 5, 12, 8)
    pen_size = st.slider("Pen Thickness", 1, 10, 3)
    st.divider()
    st.write(f"**Target for age {age}:**")
    st.info(AGE_REFS[age]["goal"])

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Writing Surface")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=pen_size,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=400,
        width=650,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    # Start timer when drawing begins
    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        if "start_time" not in st.session_state:
            st.session_state.start_time = time.time()

with col_right:
    st.subheader("Results & Feedback")
    if st.button("Submit & Analyze"):
        if canvas_result.image_data is not None and "start_time" in st.session_state:
            duration = round(time.time() - st.session_state.start_time, 2)
            
            # Step 1: Prediction
            prob = run_prediction(canvas_result.image_data, age)
            
            if prob is not None:
                # Step 2: Severity
                status, severity, color = get_severity(prob)
                
                st.markdown(f"### Status: :{color}[{status} - {severity}]")
                st.metric("Risk Probability", f"{round(float(prob)*100, 1)}%")
                st.metric("Writing Speed", f"{duration}s", delta=f"{ref['time']}s avg", delta_color="inverse")
                
                # Step 3: Personalized Feedback
                st.divider()
                st.markdown(generate_report(status, severity, age, duration))
                
                # Reset for next run
                del st.session_state.start_time
            else:
                st.error("Prediction failed. Ensure model is correctly loaded.")
        else:
            st.warning("Please write on the board first.")

if st.button("Clear Board"):
    if "start_time" in st.session_state:
        del st.session_state.start_time
    st.rerun()
