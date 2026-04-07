import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Emotion Detector", page_icon="🎭", layout="wide")

# --- LOAD MODELS ---
@st.cache_resource
def load_emotion_model():
    try:
        model = load_model("final_emotion_model.h5")
        return model
    except:
        return None

model = load_emotion_model()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotion_dict = {0: "Angry 😠", 1: "Disgusted 🤢", 2: "Fearful 😨", 3: "Happy 😄", 4: "Neutral 😐", 5: "Sad 😢", 6: "Surprised 😲"}

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🎭 Navigation")
app_mode = st.sidebar.radio("Choose a Mode:", ["Live Camera", "Upload Image"])

st.sidebar.markdown("---")
st.sidebar.info("CNN trained on FER-2013 Dataset.\n\nDeveloped by the project group.")

if model is None:
    st.error("⚠️ Model not found! Please place 'final_emotion_model.h5' in the folder.")

# ==========================================
# MODE 1: LIVE CAMERA
# ==========================================
if app_mode == "Live Camera":
    st.title("🎥 Real-Time Emotion Detection")
    st.write("Click the checkbox below to turn on your webcam. Ensure no other apps (like Zoom) are using it!")
    
    # Checkbox to start/stop the camera
    run_camera = st.checkbox("Turn On Webcam")
    
    # Create an empty placeholder to display the video frames
    FRAME_WINDOW = st.image([])
    
    if run_camera:
        # 0 is the default laptop camera
        cap = cv2.VideoCapture(0)
        
        while run_camera:
            ret, frame = cap.read()
            if not ret:
                st.error("Could not read frame from camera. Is it blocked?")
                break
            
            # Streamlit expects RGB colors, but OpenCV uses BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Detect Faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Crop and Predict
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48)) / 255.0
                roi_gray = np.reshape(roi_gray, (1, 48, 48, 1))
                
                if model:
                    prediction = model.predict(roi_gray, verbose=0)
                    max_index = int(np.argmax(prediction))
                    predicted_emotion = emotion_dict[max_index]
                    
                    cv2.putText(frame, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Display the frame in the Streamlit placeholder
            FRAME_WINDOW.image(frame)
            
        # Release the camera if the checkbox is unchecked
        cap.release()

# ==========================================
# MODE 2: UPLOAD IMAGE
# ==========================================
elif app_mode == "Upload Image":
    st.title("🖼️ Image Analysis")
    st.write("Upload a static photo to analyze the facial expressions.")
    
    image_file = st.file_uploader("Upload a face image", type=["jpg", "png", "jpeg"])
    
    if image_file is not None:
        image = Image.open(image_file)
        img_array = np.array(image)
        
        # Convert to RGB (Streamlit display) and Grayscale (Processing)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
            img_array = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            st.warning("No face detected! Try a clearer image.")
            st.image(image, use_column_width=True)
        else:
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48)) / 255.0
                roi_gray = np.reshape(roi_gray, (1, 48, 48, 1))

                if model:
                    prediction = model.predict(roi_gray, verbose=0)
                    max_index = int(np.argmax(prediction))
                    predicted_emotion = emotion_dict[max_index]
                    
                    cv2.rectangle(img_array, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    cv2.putText(img_array, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            st.success("Analysis Complete!")
            st.image(img_array, caption="Processed Image", use_column_width=True)