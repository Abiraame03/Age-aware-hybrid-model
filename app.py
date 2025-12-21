import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import time
import os

st.set_page_config(page_title="Dyslexia Risk Analyzer", layout="wide")

# --- Developmental Benchmarks ---
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

# --- Model Loading with Flex Support ---
@st.cache_resource
def load_tflite_model():
    # Looks for the file in the same directory as app.py
    model_path = "Improved_Hybrid_AgeModel2.tflite"
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found in the repository folder.")
        return None
    try:
        # Standard Interpreter automatically links Flex Ops when full tensorflow is imported
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Flex Delegate Error: {e}")
        st.info("Check if your requirements.txt includes the full 'tensorflow' library.")
        return None

interpreter = load_tflite_model()

# --- Analysis Logic ---
def get_personalized_report(prob, age, elapsed_time):
    """Generates a custom report based on age, time, and model probability."""
    ref = AGE_REFS.get(age, AGE_REFS[12])
    is_delayed = elapsed_time > ref["time"]
    speed_status = "delayed" if is_delayed else "appropriate"
    
    if prob < 0.3:
        status, color = "No Significant Risk", "green"
        insight = f"Excellent! For a {age}-year-old, the speed ({elapsed_time}s) is {speed_status}. Motor control aligns with: {ref['goal']}."
    elif prob < 0.7:
        status, color = "Moderate Risk Observed", "orange"
        insight = f"Moderate markers found. While {ref['goal']} is expected at age {age}, the {speed_status} speed and stroke irregularities suggest a need for monitoring."
    else:
        status, color = "High Risk Indicators", "red"
        insight = f"Significant dyslexic indicators detected. Writing speed is {speed_status} for age {age}. Difficulty with the target goal ({ref['goal']}) suggests professional screening."
    
    return status, color, insight

def run_inference(img_rgba, age_val):
    if interpreter is None: return None
    
    # Preprocess Image: RGBA -> RGB -> Resize -> Normalize
    img_rgb = cv2.cvtColor(img_rgba.astype(np.uint8), cv2.COLOR_RGBA2RGB)
    img_resized = cv2.resize(img_rgb, (160, 160))
    img_input = (img_resized / 255.0).astype(np.float32)
    img_input = np.expand_dims(img_input, axis=0)
    
    # Preprocess Age: Scalar to float32 tensor
    age_input = np.array([[age_val]], dtype=np.float32)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Dynamic Assignment for Image vs Age inputs
    for detail in input_details:
        if len(detail['shape']) == 4:
            interpreter.set_tensor(detail['index'], img_input)
        else:
            interpreter.set_tensor(detail['index'], age_input)
    
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0][0]

# --- UI Layout ---
st.title("🧠 Digital Handwriting Diagnostic Board")
st.markdown("Write the sentence below with a stylus: **'The quick brown fox'**")

with st.sidebar:
    st.header("Student Parameters")
    age = st.slider("Select Age", 5, 12, 8)
    pen_width = st.slider("Pen Thickness", 1, 15, 4)
    st.write(f"**Target Goal for Age {age}:**")
    st.caption(AGE_REFS[age]["goal"])

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Writing Surface")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=pen_width,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=400,
        width=650,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    # Capture start time when the child begins writing
    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        if "start_time" not in st.session_state:
            st.session_state.start_time = time.time()

with col_right:
    st.subheader("Analysis Results")
    if st.button("Predict Dyslexia Risk"):
        if canvas_result.image_data is not None and "start_time" in st.session_state:
            total_time = round(time.time() - st.session_state.start_time, 2)
            prob = run_inference(canvas_result.image_data, age)
            
            if prob is not None:
                status, color, feedback = get_personalized_report(prob, age, total_time)
                
                st.markdown(f"### Status: :{color}[{status}]")
                st.metric("Completion Time", f"{total_time}s")
                st.metric("Risk Probability", f"{round(float(prob)*100, 1)}%")
                st.info(f"**Personalized Justification:**\n\n{feedback}")
                
                # Reset timer for next session
                del st.session_state.start_time
        else:
            st.warning("Please draw on the board first.")

if st.button("Clear Canvas"):
    if "start_time" in st.session_state:
        del st.session_state.start_time
    st.rerun()
