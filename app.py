import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import time
import os
from streamlit_drawable_canvas import st_canvas

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Dyslexia Handwriting Risk Analyzer",
    layout="centered"
)

st.title("🧠 Dyslexia Handwriting Pattern Analyzer")
st.caption("Stylus-based handwriting analysis with age & time awareness")

# --------------------------------------------------
# Load TFLite Model
# --------------------------------------------------
@st.cache_resource
def load_tflite_model():
    model_path = "Improved_Hybrid_AgeModel2.tflite"
    if not os.path.exists(model_path):
        return None
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

# --------------------------------------------------
# Severity Mapping
# --------------------------------------------------
def get_severity_info(prob):
    if prob < 0.25:
        return "Normal", "No Risk", "green"
    elif prob < 0.50:
        return "Normal", "Low Risk", "blue"
    elif prob < 0.75:
        return "Dyslexic", "Moderate Risk", "orange"
    else:
        return "Dyslexic", "High Risk", "red"

# --------------------------------------------------
# Personalized Feedback
# --------------------------------------------------
def personalized_feedback(prob, age, writing_time):
    if prob < 0.25:
        return (
            f"✅ Handwriting is appropriate for age {age}.\n\n"
            f"- Writing speed is within normal range ({writing_time:.1f}s)\n"
            f"- Stroke formation is smooth and consistent\n"
            f"- No dyslexic indicators detected"
        )
    elif prob < 0.50:
        return (
            f"⚠️ Minor handwriting variations observed.\n\n"
            f"- Writing time slightly higher than peers ({writing_time:.1f}s)\n"
            f"- Small inconsistencies in letter spacing\n"
            f"- Still within acceptable age norms"
        )
    elif prob < 0.75:
        return (
            f"❗ Moderate dyslexic indicators detected.\n\n"
            f"- Writing speed slower than expected ({writing_time:.1f}s)\n"
            f"- Irregular stroke continuity\n"
            f"- May benefit from guided handwriting practice"
        )
    else:
        return (
            f"🚨 High dyslexia risk detected.\n\n"
            f"- Writing time significantly exceeds age norms ({writing_time:.1f}s)\n"
            f"- Poor stroke consistency and motor planning\n"
            f"- Professional evaluation is strongly recommended"
        )

# --------------------------------------------------
# Run Inference
# --------------------------------------------------
def run_inference(img_array, age_val, time_val):
    if interpreter is None:
        return None

    # Image preprocessing
    img_resized = cv2.resize(img_array, (160, 160))
    img_norm = (img_resized / 255.0).astype(np.float32)
    img_input = np.expand_dims(img_norm, axis=0)

    # Age & time inputs
    age_input = np.array([[age_val]], dtype=np.float32)
    time_input = np.array([[time_val]], dtype=np.float32)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Dynamic input assignment
    for detail in input_details:
        shape_len = len(detail["shape"])
        if shape_len == 4:
            interpreter.set_tensor(detail["index"], img_input)
        elif shape_len == 2:
            # Heuristic: first scalar = age, second = time
            if "age_used" not in locals():
                interpreter.set_tensor(detail["index"], age_input)
                age_used = True
            else:
                interpreter.set_tensor(detail["index"], time_input)

    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    return float(output[0][0])

# --------------------------------------------------
# Sidebar Controls
# --------------------------------------------------
st.sidebar.header("Student Details")
age = st.sidebar.slider("Student Age", 5, 15, 8)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Instructions**\n"
    "- Write a letter or word on the board\n"
    "- Use stylus / finger / mouse\n"
    "- Click *Analyze Handwriting*"
)

# --------------------------------------------------
# Writing Canvas
# --------------------------------------------------
st.subheader("✍️ Writing Board")

if "start_time" not in st.session_state:
    st.session_state.start_time = None

canvas = st_canvas(
    fill_color="rgba(255,255,255,0)",
    stroke_width=4,
    stroke_color="black",
    background_color="white",
    width=500,
    height=250,
    drawing_mode="freedraw",
    key="canvas",
)

# Start timing when user begins writing
if canvas.json_data is not None and st.session_state.start_time is None:
    st.session_state.start_time = time.time()

# --------------------------------------------------
# Analyze Button
# --------------------------------------------------
if st.button("🔍 Analyze Handwriting"):
    if canvas.image_data is None:
        st.warning("Please write something on the board first.")
    elif interpreter is None:
        st.error("TFLite model not found.")
    else:
        end_time = time.time()
        writing_time = end_time - st.session_state.start_time

        img = canvas.image_data[:, :, :3].astype(np.uint8)

        prob = run_inference(img, age, writing_time)

        if prob is not None:
            result, severity, color = get_severity_info(prob)

            st.markdown(f"## Status: :{color}[{result} – {severity}]")
            st.progress(prob)
            st.write(f"**Risk Probability:** {prob*100:.2f}%")
            st.write(f"**Writing Time:** {writing_time:.2f} seconds")

            st.markdown("### 🧾 Personalized Feedback")
            st.info(personalized_feedback(prob, age, writing_time))

        # Reset timer for next attempt
        st.session_state.start_time = None
