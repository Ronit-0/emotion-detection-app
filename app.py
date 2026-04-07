import os
import urllib.request
import streamlit as st
import cv2
import numpy as np
import av
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# --- 🚨 CRITICAL FIX FOR SEGMENTATION FAULT 🚨 ---
# This prevents OpenCV from fighting WebRTC for memory/threads on the cloud server
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Live Emotion Detector", page_icon="🎭", layout="centered")

# --- LOAD MODELS (HUGGING FACE) ---
MODEL_URL = "https://huggingface.co/Ronit-0/fer2013-emotion-model/resolve/main/final_emotion_model.h5?download=true"
MODEL_PATH = "final_emotion_model.h5"

@st.cache_resource
def load_emotion_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading Deep Learning Model (approx. 66MB)..."):
            try:
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            except Exception as e:
                st.error(f"Failed to download model: {e}")
                return None
    try:
        return load_model(MODEL_PATH)
    except:
        return None

# Load model and cascade globally
model = load_emotion_model()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotion_dict = {0: "Angry 😠", 1: "Disgusted 🤢", 2: "Fearful 😨", 3: "Happy 😄", 4: "Neutral 😐", 5: "Sad 😢", 6: "Surprised 😲"}

# --- LIVE VIDEO PROCESSOR ---
class EmotionProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            # Crop, resize, and normalize
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48)) / 255.0
            roi_gray = np.reshape(roi_gray, (1, 48, 48, 1))
            
            # Predict
            if model:
                prediction = model.predict(roi_gray, verbose=0)
                max_index = int(np.argmax(prediction))
                predicted_emotion = emotion_dict[max_index]
                
                # Draw box and text
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI DESIGN ---
st.title("🎭 Live Emotion Detection")
st.markdown("Click **START** to open your camera and see live emotion tracking!")

if model is None:
    st.error("⚠️ Model failed to load.")

# WEBRTC CONFIGURATION
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Render the live video player
webrtc_streamer(
    key="emotion-tracker",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

st.info("💡 Tip: Make sure you are in a well-lit room and looking directly at the camera.")
