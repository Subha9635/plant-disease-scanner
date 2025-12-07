import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Load Model (Cached to prevent reloading on every interaction)
@st.cache_resource
def load_model():
    # Ensure this name matches your downloaded file in the project folder
    return tf.keras.models.load_model('plant_village_model.h5')

model = load_model()

# 2. Define Classes (Copied directly from your Colab output)
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

# --- UI Setup ---
st.set_page_config(page_title="Plant Disease Scanner", layout="wide")
st.title("🌿 PlantVillage AI Disease Scanner")
st.markdown("Use your webcam to scan a leaf and identify potential diseases.")

st.markdown("---")

# 3. Camera Input
uploaded_file = st.camera_input("Take a picture of the plant leaf")

if uploaded_file is not None:
    # --- Image Processing ---
    image = Image.open(uploaded_file)
    st.image(image, caption="Captured Leaf Image", width=300)
    
    # Resize to 224x224 (Must match training size)
    img_array = np.array(image.resize((224, 224)))
    img_array = img_array / 255.0  # Normalize to 0-1
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

    # --- Prediction ---
    with st.spinner('Analyzing image...'):
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = np.max(predictions[0]) * 100

    # --- Display Results ---
    st.markdown("---")
    
    # Clean up the name for a user-friendly display
    clean_name = predicted_class.replace("___", ": ").replace("_", " ").replace("Two-spotted spider mite", "Two-Spotted Spider Mite")
    
    st.subheader(f"Diagnosis: **{clean_name}**")
    st.caption(f"Confidence: **{confidence:.2f}%**")
    
    if "healthy" in predicted_class.lower():
        st.balloons()
        st.success("✅ The plant appears healthy! Keep monitoring it.")
    else:
        st.error("🚨 DISEASE DETECTED!")
        st.warning("Immediate action is recommended. Please isolate the plant and seek specific treatment.")
