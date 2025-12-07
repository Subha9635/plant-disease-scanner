import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ----------------------------------------------------------------------------------
# CONFIGURATION AND STYLING
# ----------------------------------------------------------------------------------

# 1. Custom CSS Styling
st.set_page_config(page_title="Plant Disease Scanner", layout="wide")
st.markdown("""
    <style>
    /* Main App Background Color */
    .stApp {
        background-color: #121212; /* Very Dark Gray/Near Black */
    }
    /* Header/Title Styling */
    h1 {
        color: #1a5e1a; /* Dark Green */
        text-align: center;
        font-weight: 700;
    }
    /* Button Styling */
    .stButton>button {
        background-color: #4CAF50; /* Medium Green */
        color: white;
        border-radius: 8px;
        border: 2px solid #388E3C;
        padding: 10px 20px;
        font-size: 16px;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #388E3C; /* Darker Green on Hover */
    }
    </style>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------------
# MODEL SETUP
# ----------------------------------------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('plant_village_model.h5')

model = load_model()

# Define Classes and Threshold
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 
    'Tomato___healthy'
]

CONFIDENCE_THRESHOLD = 0.90 # 90% confidence needed for a positive ID

# Initialize session state for camera control
if 'camera_on' not in st.session_state:
    st.session_state.camera_on = False

# ----------------------------------------------------------------------------------
# UI AND INPUT LOGIC (UPDATED)
# ----------------------------------------------------------------------------------

st.title("🌿 PlantVillage AI Disease Scanner")
st.markdown("---")

# Use columns for clear input separation
col1, col2 = st.columns(2)
source_file = None

with col1:
    uploaded_file = st.file_uploader("📁 1. Upload Image from Gallery", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        source_file = uploaded_file

with col2:
    st.markdown("#### 2. Scan Leaf with Camera")
    # Button to control camera access
    if st.button("Start Camera Scan"):
        st.session_state.camera_on = True
    
    # Conditional camera rendering
    if st.session_state.camera_on:
        camera_input = st.camera_input("Take Photo", label_visibility="collapsed")
        if camera_input:
            source_file = camera_input
            # Turn off camera flag after photo is taken to reset the widget
            st.session_state.camera_on = False 

# --- Prediction Logic (Only runs if a source file is available) ---
if source_file is not None:
    
    # Display the input image
    image = Image.open(source_file)
    st.image(image, caption="Captured/Uploaded Image", width=300)

    # Image Processing
    img_array = np.array(image.resize((224, 224)))
    img_array = img_array / 255.0 
    img_array = np.expand_dims(img_array, axis=0) 

    # Prediction
    with st.spinner('Analyzing image...'):
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = np.max(predictions[0]) * 100

    st.markdown("---")
    
    # --- NON-PLANT IMAGE DETECTION (Immediate Error) ---
    if confidence < CONFIDENCE_THRESHOLD:
        st.error("❌ Non-Plant Image Detected")
        st.subheader("Analysis Failed: Non-Plant or Low Confidence")
        st.warning(f"The model's confidence ({confidence:.2f}%) is too low. Please ensure the image is a clear, single plant leaf.")
        # STOP HERE: Do not display further results or diagnostics
        st.stop()
        
    # --- Display Positive Diagnosis ---
    else:
        # Clean up the name for a user-friendly display
        clean_name = predicted_class.replace("___", ": ").replace("_", " ")
        
        st.subheader(f"Diagnosis: **{clean_name}**")
        st.caption(f"Confidence: **{confidence:.2f}%**")
        
        if "healthy" in predicted_class.lower():
            st.balloons()
            st.success("✅ The plant appears healthy! Keep monitoring it.")
        else:
            st.error("🚨 DISEASE DETECTED!")
            st.info("Immediate action is recommended. Please isolate the plant and seek specific treatment.")
