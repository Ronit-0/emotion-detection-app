import os
import urllib.request
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Emotion Detector AI", page_icon="🎭", layout="centered")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0px;
        background: -webkit-linear-gradient(#4facfe, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-title {
        text-align: center;
        font-size: 1.1rem;
        color: #A0AEC0;
        margin-bottom: 30px;
    }
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
    }
    img {
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.5);
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD MODELS (HUGGING FACE) ---
MODEL_URL = "https://huggingface.co/Ronit-0/fer2013-emotion-model/resolve/main/final_emotion_model.h5?download=true"
MODEL_PATH = "final_emotion_model.h5"

@st.cache_resource
def load_emotion_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📦 Downloading AI Weights (66MB)... Please wait."):
            try:
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            except Exception as e:
                st.error(f"Failed to download model: {e}")
                return None
    try:
        return load_model(MODEL_PATH)
    except:
        return None

model = load_emotion_model()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotion_dict = {0: "Angry 😠", 1: "Disgusted 🤢", 2: "Fearful 😨", 3: "Happy 😄", 4: "Neutral 😐", 5: "Sad 😢", 6: "Surprised 😲"}

# --- MAIN UI HEADER ---
st.markdown('<div class="main-title">Facial Emotion Analysis AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload photos or use your camera to detect emotional states.</div>', unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model failed to load. Check the Hugging Face link.")

# --- THE AI ENGINE (Helper Function) ---
def run_analysis(image_file, file_name="Captured Image"):
    # Create a visual container for each image so they don't blend together
    with st.container():
        st.markdown(f"#### 📄 Analyzing: `{file_name}`")
        
        with st.spinner("Analyzing facial features..."):
            image = Image.open(image_file)
            img_array = np.array(image)
            
            if len(img_array.shape) == 3 and img_array.shape[2] == 4:
                 img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                 
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
                img_array = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            if len(faces) == 0:
                st.warning("No face detected! Please try an image with a clearer face.")
            else:
                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_gray = cv2.resize(roi_gray, (48, 48))
                    roi_gray = roi_gray / 255.0 
                    roi_gray = np.reshape(roi_gray, (1, 48, 48, 1))

                    if model:
                        prediction = model.predict(roi_gray, verbose=0)
                        max_index = int(np.argmax(prediction))
                        predicted_emotion = emotion_dict[max_index]
                        confidence = np.max(prediction) * 100
                        
                        cv2.rectangle(img_array, (x, y), (x+w, y+h), (0, 255, 150), 3)
                        cv2.putText(img_array, predicted_emotion, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 150), 2)

                col1, col2 = st.columns([1.5, 1], gap="large")
                with col1:
                    st.image(img_array, use_container_width=True)
                with col2:
                    st.write("") 
                    st.write("") 
                    st.metric(label="Primary Emotion", value=predicted_emotion)
                    st.metric(label="AI Confidence", value=f"{confidence:.2f}%")
        
        st.markdown("---") # Adds a clean dividing line between multiple images

# --- TABBED INTERFACE ---
tab1, tab2 = st.tabs(["📸 Take a Picture", "🖼️ Upload Images (Batch)"])

with tab1:
    st.write("Use your device's camera to analyze your current expression.")
    camera_img = st.camera_input("Smile for the camera!", label_visibility="collapsed")
    if camera_img is not None:
        run_analysis(camera_img, "Webcam Capture")

with tab2:
    st.write("Upload one or more clear photos of faces for analysis.")
    # Notice the new 'accept_multiple_files=True' parameter!
    uploaded_imgs = st.file_uploader("Drag and drop images here", type=["jpg", "png", "jpeg"], accept_multiple_files=True, label_visibility="collapsed")
    
    # If the user uploads files, this loops through every single one dynamically
    if uploaded_imgs:
        st.success(f"Successfully loaded {len(uploaded_imgs)} image(s) into the pipeline.")
        for img in uploaded_imgs:
            run_analysis(img, img.name)
