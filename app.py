{\rtf1\ansi\ansicpg1252\cocoartf2865
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\fnil\fcharset0 Menlo-Regular;}
{\colortbl;\red255\green255\blue255;\red24\green24\blue24;\red255\green255\blue255;}
{\*\expandedcolortbl;;\cssrgb\c12157\c12157\c12157;\cssrgb\c100000\c100000\c100000;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import streamlit as st\
import tensorflow as tf\
import numpy as np\
from PIL import Image\
\
# 1. Load Model\
@st.cache_resource\
def load_model():\
    return tf.keras.models.load_model('plant_village_model.h5')\
\
model = load_model()\
\
# 2. Define Classes\
# \uc0\u9888 \u65039  REPLACE THE LIST BELOW with the one you copied from Colab! \u9888 \u65039 \
# It will look something like this (but much longer):\
CLASS_NAMES = [\
    
\f1 \cf2 \cb3 \expnd0\expndtw0\kerning0
'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
\f0 \cf0 \cb1 \kerning1\expnd0\expndtw0 \
]\
\
st.title("\uc0\u55356 \u57151  PlantVillage Disease Scanner")\
st.markdown("---")\
\
# 3. Camera Input\
uploaded_file = st.camera_input("Scan a Leaf")\
\
if uploaded_file is not None:\
    # Processing\
    image = Image.open(uploaded_file)\
    st.image(image, caption="Captured Image", use_column_width=True)\
    \
    # Resize to 224x224 (Standard for MobileNetV2)\
    img_array = np.array(image.resize((224, 224)))\
    img_array = img_array / 255.0  # Normalize to 0-1\
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension\
\
    # Prediction\
    predictions = model.predict(img_array)\
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]\
    confidence = np.max(predictions[0]) * 100\
\
    # Display\
    st.write("---")\
    st.subheader(f"Detected: **\{predicted_class\}**")\
    st.write(f"Confidence: **\{confidence:.2f\}%**")\
\
    # Clean up the output text for the user\
    clean_name = predicted_class.replace("_", " ")\
    st.info(f"Diagnosis: \{clean_name\}")\
    \
    if "healthy" in predicted_class:\
        st.success("\uc0\u9989  Plant appears healthy.")\
    else:\
        st.warning("\uc0\u9888 \u65039  Disease detected.")}