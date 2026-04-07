import os
import urllib.request
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Emotion Detector", page_icon="🎭", layout="centered")

# --- LOAD MODELS (HUGGING FACE) ---
# Your exact raw download link
MODEL_URL = "https://huggingface.co/Ronit-0/fer2013-emotion-model/resolve/main/final_emotion_model.h5?download=true"
MODEL_PATH = "final_emotion_model.h5"

@st.cache_resource
def load_emotion_model():
    # If the model isn't on the Streamlit server yet, download it!
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading Deep Learning Model (approx. 66MB)... This only happens once!"):
            try:
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
                st.success("Model downloaded successfully!")
            except Exception as e:
                st.error(f"Failed to download model: {e}")
                return None
    
    # Load the model into memory
    try:
        model = load_model(MODEL_PATH)
        return model
    except:
        return None

# Load the model and face tracker
model = load_emotion_model()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotion_dict = {0: "Angry 😠", 1: "Disgusted 🤢", 2: "Fearful 😨", 3: "Happy 😄", 4: "Neutral 😐", 5: "Sad 😢", 6: "Surprised 😲"}

# --- UI DESIGN ---
st.title("🎭 Human Emotion Detection")
st.markdown("Developed by the team. Upload an image or take a picture to analyze facial expressions!")

if model is None:
    st.error("⚠️ Model failed to load. Check the Hugging Face link.")

# Options for user input
option = st.radio("Choose Input Method:", ("Take a Picture", "Upload Image"))

image_file = None
if option == "Upload Image":
    image_file = st.file_uploader("Upload a face image", type=["jpg", "png", "jpeg"])
elif option == "Take a Picture":
    # This automatically triggers the selfie camera on mobile and the webcam on desktop!
    image_file = st.camera_input("Take a picture")

# --- PROCESSING ---
if image_file is not None:
    # 1. Read the image from the UI
    image = Image.open(image_file)
    img_array = np.array(image)
    
    # Fix for PNG images with transparency (RGBA to RGB)
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
         img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
         
    # Convert to grayscale for OpenCV face detection
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
        img_array = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # 2. Detect Faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        st.warning("No face detected! Please try an image with a clearer face.")
        st.image(image, caption="Original Image", use_column_width=True)
    else:
        # 3. Draw rectangles and predict
        for (x, y, w, h) in faces:
            # Crop the face and resize to 48x48
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            
            # Normalize pixel values to 0-1
            roi_gray = roi_gray / 255.0 
            roi_gray = np.reshape(roi_gray, (1, 48, 48, 1))

            # Make Prediction
            if model:
                prediction = model.predict(roi_gray, verbose=0)
                max_index = int(np.argmax(prediction))
                predicted_emotion = emotion_dict[max_index]
                
                # Draw Box and Text on the colored image
                cv2.rectangle(img_array, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.putText(img_array, predicted_emotion, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        # 4. Display the Result
        st.success("Analysis Complete!")
        st.image(img_array, caption="Detected Emotion", use_column_width=True)
