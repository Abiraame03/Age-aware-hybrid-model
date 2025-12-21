import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import time
import os

st.set_page_config(page_title="Dyslexia Risk Analyzer", layout="wide")

# --- 1. Age-Based Reference Benchmarks ---
# These benchmarks serve as the "ground truth" for age-wise handwriting performance
AGE_BENCHMARKS = {
    5: {"avg_time": 60, "milestone": "Basic stroke orientation and shape recognition."},
    6: {"avg_time": 52, "milestone": "Consistent letter sizing and baseline alignment."},
    7: {"avg_time": 45, "milestone": "Improved word spacing and letter formation."},
    8: {"avg_time": 38, "milestone": "Fluidity and connection between letters."},
    9: {"avg_time": 32, "milestone": "Automaticity in letter production (less mental effort)."},
    10: {"avg_time": 28, "milestone": "Legibility maintained at higher writing speeds."},
    11: {"avg_time": 24, "milestone": "Mature motor control and efficient note-taking speed."},
    12: {"avg_time": 20, "milestone": "Adult-level handwriting automaticity."}
}

# --- 2. Model Loading (Fixing the Flex Delegate Error) ---
@st.cache_resource
def load_tflite_model():
    model_path = "Improved_Hybrid_AgeModel2.tflite"
    if not os.path.exists(model_path):
        st.error(f"Error: '{model_path}' not found in the current directory.")
        return None
    try:
        # Standard tf.lite.Interpreter automatically handles Flex Ops 
        # when the full 'tensorflow' library is imported.
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Flex Delegate Error: {e}")
        st.info("Ensure your requirements.txt uses 'tensorflow' and not 'tflite-runtime'.")
        return None

interpreter = load_tflite_model()

# --- 3. Prediction & Severity Logic ---
def run_inference(img_rgba, age_val):
    if interpreter is None: return None
    
    # Preprocess Image: RGBA -> RGB -> 160x160 -> Float32
    img_rgb = cv2.cvtColor(img_rgba.astype(np.uint8), cv2.COLOR_RGBA2RGB)
    img_resized = cv2.resize(img_rgb, (160, 160))
    img_input = (img_resized / 255.0).astype(np.float32)
    img_input = np.expand_dims(img_input, axis=0)
    
    # Preprocess Age Input (Scalar to float32 tensor)
    age_input = np.array([[age_val]], dtype=np.float32)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Assign inputs dynamically (Image vs Age)
    for detail in input_details:
        if len(detail['shape']) == 4:
            interpreter.set_tensor(detail['index'], img_input)
        else:
            interpreter.set_tensor(detail['index'], age_input)
    
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0][0]

def get_severity(prob):
    """Step 1: Determine prediction category and severity level."""
    if prob < 0.25:
        return "Normal", "No Risk", "green"
    elif prob < 0.50:
        return "Normal", "Low Risk", "blue"
    elif prob < 0.75:
        return "Dyslexic", "Moderate Risk", "orange"
    else:
        return "Dyslexic", "High Risk", "red"

def generate_feedback(prob, status, severity, age, elapsed_time):
    """Step 2: Proceed with Age and Time parameters for personalized feedback."""
    ref = AGE_BENCHMARKS.get(age, AGE_BENCHMARKS[12])
    speed_diff = elapsed_time - ref["avg_time"]
    
    # Speed evaluation
    if speed_diff > 10:
        speed_critique = f"The writing speed is significantly slower ({elapsed_time}s) than the age-appropriate average of {ref['avg_time']}s."
    else:
        speed_critique = f"The writing speed is within or near the expected range for an {age}-year-old."

    # Feedback customization
    if status == "Normal":
        feedback = f"Great progress! At age {age}, the child has achieved the target milestone: **{ref['milestone']}**. {speed_critique} No significant dyslexic markers were found in the stroke patterns."
    else:
        feedback = f"The analysis suggests a {severity} of dyslexic patterns. At age {age}, we typically expect **{ref['milestone']}**. {speed_critique} The combination of pattern irregularity and temporal delay indicates a need for professional pedagogical support."
    
    return feedback

# --- 4. User Interface ---
st.title("🧠 Neuro-Writing Analysis Board")
st.markdown("Please write the following phrase: **'The quick brown fox'**")

# Sidebar
with st.sidebar:
    st.header("Student Profile")
    age = st.slider("Select Student Age", 5, 12, 8)
    pen_size = st.slider("Pen Thickness", 1, 15, 3)
    st.divider()
    st.write(f"**Age {age} Benchmark:**")
    st.caption(AGE_BENCHMARKS[age]["milestone"])

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Interactive Writing Surface")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=pen_size,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=450,
        width=700,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    # Start timer when the first object is drawn
    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        if "start_time" not in st.session_state:
            st.session_state.start_time = time.time()

with col_right:
    st.subheader("Diagnostic Report")
    if st.button("Submit & Analyze"):
        if canvas_result.image_data is not None and "start_time" in st.session_state:
            # 1. Prediction
            total_time = round(time.time() - st.session_state.start_time, 2)
            prob = run_inference(canvas_result.image_data, age)
            
            if prob is not None:
                # 2. Severity
                status, severity, color = get_severity(prob)
                
                # 3. Feedback (Age & Time context)
                feedback_text = generate_feedback(prob, status, severity, age, total_time)
                
                # Display Results
                st.markdown(f"### Status: :{color}[{status} - {severity}]")
                st.metric("Risk Probability", f"{round(float(prob)*100, 1)}%")
                st.metric("Writing Speed", f"{total_time}s", delta=f"{ref['avg_time']}s avg", delta_color="inverse")
                
                st.divider()
                st.markdown("#### Personalized Feedback")
                st.info(feedback_text)
                
                # Clear session for next attempt
                del st.session_state.start_time
            else:
                st.error("Prediction failed. Ensure the model is correctly loaded.")
        else:
            st.warning("Please write on the board before submitting for analysis.")

if st.button("Clear Canvas"):
    
    if "start_time" in st.session_state:
        del st.session_state.start_time
    st.rerun()
