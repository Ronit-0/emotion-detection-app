import os
import urllib.request
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import google.generativeai as genai

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Emotion Detector AI", page_icon="🎭", layout="centered")

# --- CONFIGURE CHATBOT & VISION API ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    chatbot_model = genai.GenerativeModel('gemini-2.5-flash-lite')
except Exception as e:
    chatbot_model = None

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Analyze an image first, and I can give you advice, quotes, or music recommendations based on your mood!"}]
if "current_emotion" not in st.session_state:
    st.session_state.current_emotion = "Neutral"

# --- 🎨 BULLETPROOF CUSTOM CSS 🎨 ---
st.markdown("""
    <style>
    /* 1. HIDE HEADER & FOOTER */
    header {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    
    /* 2. THE BACKGROUND (Forcing Streamlit's inner container to be transparent) */
    .stApp {
        background-color: #0B0F19 !important;
    }
    [data-testid="stAppViewContainer"] {
        background-color: transparent !important;
        background-image: 
            radial-gradient(at 10% 10%, rgba(30, 58, 138, 0.4) 0px, transparent 50%),
            radial-gradient(at 90% 90%, rgba(88, 28, 135, 0.3) 0px, transparent 50%) !important;
        background-attachment: fixed !important;
        color: #F8FAFC;
    }

    /* 3. PILL-SHAPED TABS */
    /* The container holding the tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 50px;
        padding: 5px 15px;
        display: flex;
        justify-content: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    /* The individual tabs */
    .stTabs [data-baseweb="tab"] {
        border-radius: 50px !important;
        padding: 10px 24px !important;
        background-color: transparent;
        color: #94A3B8;
        border: none !important;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.1);
        color: #F8FAFC;
    }
    /* The active tab */
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.5) !important;
    }
    /* Hide the annoying blue underline */
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }

    /* 4. TYPOGRAPHY */
    .main-title { 
        font-size: 3rem; 
        font-weight: 800; 
        text-align: center; 
        margin-top: -50px; 
        margin-bottom: 5px; 
        background: linear-gradient(to right, #60a5fa, #c084fc); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
    }
    .sub-title { 
        text-align: center; 
        font-size: 1.1rem; 
        color: #94a3b8; 
        margin-bottom: 20px; 
    }
    img { 
        border-radius: 12px; 
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD LOCAL CNN MODEL ---
MODEL_URL = "https://huggingface.co/Ronit-0/fer2013-emotion-model/resolve/main/final_emotion_model.h5?download=true"
MODEL_PATH = "final_emotion_model.h5"

@st.cache_resource
def load_emotion_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📦 Downloading AI Weights (66MB)... Please wait."):
            try:
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            except:
                return None
    try:
        return load_model(MODEL_PATH)
    except:
        return None

model = load_emotion_model()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emoji_map = {"Angry": "😠", "Disgusted": "🤢", "Fearful": "😨", "Happy": "😄", "Neutral": "😐", "Sad": "😢", "Surprised": "😲"}
cnn_emotion_list = ["Angry", "Disgusted", "Fearful", "Happy", "Sad", "Surprised", "Neutral"]

suggestion_dict = {
    "Happy": ["Give me a happy quote! ☀️", "Recommend an upbeat song 🎵", "Tell me a joke! 😂"],
    "Sad": ["Give me a comforting quote 🌧️", "How can I cheer up?", "Recommend a calming song 🎧"],
    "Angry": ["How to calm down? 🧘", "Give me a peaceful quote 🍃", "Recommend relaxing music 🎶"],
    "Fearful": ["Give me a courageous quote 🦁", "How to overcome anxiety?", "Recommend a soothing song 🎹"],
    "Surprised": ["Tell me a mind-blowing fact! 🤯", "Recommend an exciting movie 🍿", "Give me a fun trivia question 🎲"],
    "Disgusted": ["Tell me a funny story! 🤣", "How to clear my mind?", "Give me a random fun fact 💡"],
    "Neutral": ["Tell me a fun fact! 🧠", "Give me a motivational quote 🚀", "Recommend a good book 📚"]
}

# --- MAIN UI HEADER ---
st.markdown('<div class="main-title">Facial Emotion Analysis AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Detect emotional states and chat with an AI assistant.</div>', unsafe_allow_html=True)

colA, colB, colC = st.columns([1, 2, 1])
with colB:
    use_gemini = st.toggle("🚀 Enable High-Accuracy Mode (Gemini Vision AI)", value=False)
st.write("") 

# --- THE AI ENGINE ---
def run_analysis(image_file, file_name="Captured Image"):
    with st.container(border=True): # Adds a clean built-in Streamlit border
        st.markdown(f"#### 📄 Analyzing: `{file_name}`")
        with st.spinner("Processing facial features..."):
            image = Image.open(image_file) 
            img_array = np.array(image)
            
            if len(img_array.shape) == 3 and img_array.shape[2] == 4:
                 img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
                img_array = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

            gray = cv2.equalizeHist(gray)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))

            if len(faces) == 0:
                st.warning("No face detected! Please try an image with a clearer face.")
                st.image(image, use_container_width=True)
            else:
                for (x, y, w, h) in faces:
                    if use_gemini and chatbot_model is not None:
                        try:
                            vision_prompt = "Analyze the facial expression of the primary person in this image. Classify their emotion into exactly one of these words: Angry, Disgusted, Fearful, Happy, Sad, Surprised, Neutral. Also estimate your confidence from 0 to 100. Respond strictly in this format: Emotion,Confidence (Example: Happy,95)"
                            response = chatbot_model.generate_content([vision_prompt, image])
                            
                            response_text = response.text.strip()
                            if "," in response_text:
                                parts = response_text.split(",")
                                base_emotion = parts[0].strip().capitalize()
                                confidence_display = f"{parts[1].strip().replace('%', '')}%"
                            else:
                                base_emotion = response_text.capitalize()
                                confidence_display = "99.0%"

                            if base_emotion not in cnn_emotion_list:
                                base_emotion = "Neutral"
                                
                            predicted_emotion_ui = f"{base_emotion} {emoji_map.get(base_emotion, '')}"
                            model_used_text = "Gemini Vision"
                            color = (255, 200, 0)
                            
                        except Exception as e:
                            error_msg = str(e).lower()
                            if "429" in error_msg or "quota" in error_msg or "exhausted" in error_msg:
                                st.toast("⏳ Server overload! Gemini AI limit reached. Temporarily using Custom CNN.", icon="⚠️")
                            else:
                                st.toast("⚠️ Gemini Vision unavailable. Falling back to Custom CNN.", icon="⚠️")
                            
                            base_emotion = "Neutral"
                            predicted_emotion_ui = "Neutral 😐"
                            confidence_display = "N/A"
                            model_used_text = "API Limit Exceeded"
                            color = (0, 0, 255)
                    else:
                        roi_gray = gray[y:y+h, x:x+w]
                        roi_gray = cv2.resize(roi_gray, (48, 48)) / 255.0 
                        roi_gray = np.reshape(roi_gray, (1, 48, 48, 1))

                        if model:
                            prediction = model.predict(roi_gray, verbose=0)
                            max_index = int(np.argmax(prediction))
                            base_emotion = cnn_emotion_list[max_index]
                            
                            predicted_emotion_ui = f"{base_emotion} {emoji_map.get(base_emotion, '')}"
                            confidence_display = f"{(np.max(prediction) * 100):.2f}%"
                            model_used_text = "Custom CNN"
                            color = (0, 255, 150)
                    
                    st.session_state.current_emotion = base_emotion
                    
                    cv2.rectangle(img_array, (x, y), (x+w, y+h), color, 3)
                    cv2.putText(img_array, base_emotion, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                col1, col2 = st.columns([1.5, 1], gap="large")
                with col1:
                    st.image(img_array, use_container_width=True)
                with col2:
                    st.write("") 
                    st.write("") 
                    st.metric(label="Primary Emotion", value=predicted_emotion_ui)
                    st.metric(label=f"Confidence ({model_used_text})", value=confidence_display)
        st.write("") # Spacer

# --- 3-TABBED INTERFACE ---
tab1, tab2, tab3 = st.tabs(["📸 Camera", "🖼️ Upload Images", "💬 AI Assistant"])

with tab1:
    camera_img = st.camera_input("Smile for the camera!", label_visibility="collapsed")
    if camera_img is not None:
        run_analysis(camera_img, "Webcam Capture")

with tab2:
    uploaded_imgs = st.file_uploader("Upload images", type=["jpg", "png", "jpeg"], accept_multiple_files=True, label_visibility="collapsed")
    if uploaded_imgs:
        st.success(f"Successfully loaded {len(uploaded_imgs)} image(s) into the pipeline.")
        for img in uploaded_imgs:
            run_analysis(img, img.name)

# --- RE-DESIGNED CHATBOT UI ---
with tab3:
    current_mood = st.session_state.current_emotion
    emoji = emoji_map.get(current_mood, '')
    
    # Stylish Mood Header
    st.info(f"### Detected Mood: **{current_mood}** {emoji}")
    
    if chatbot_model is None:
        st.error("⚠️ Gemini API Key missing or invalid! Please check your Streamlit Secrets.")
    else:
        st.write("✨ **What would you like to do?**")
        suggestions = suggestion_dict.get(current_mood, suggestion_dict["Neutral"])
        
        suggestion_clicked = None
        # Tighter column layout for buttons
        sug_col1, sug_col2, sug_col3 = st.columns(3, gap="small")
        if sug_col1.button(suggestions[0]): suggestion_clicked = suggestions[0]
        if sug_col2.button(suggestions[1]): suggestion_clicked = suggestions[1]
        if sug_col3.button(suggestions[2]): suggestion_clicked = suggestions[2]
        
        st.divider() # Clean horizontal line

        # Chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt = st.chat_input(f"Ask me something about feeling {current_mood}...")
        
        if suggestion_clicked:
            prompt = suggestion_clicked

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        system_prompt = f"The user's face was scanned by an AI and they look {current_mood}. Keep this in mind when answering: {prompt}"
                        response = chatbot_model.generate_content(system_prompt)
                        st.markdown(response.text)
                        st.session_state.messages.append({"role": "assistant", "content": response.text})
                    except Exception as e:
                        error_msg = str(e).lower()
                        if "429" in error_msg or "quota" in error_msg or "exhausted" in error_msg:
                            st.warning("⏳ **Server Overload:** The AI has reached its rate limit. Please wait a minute and try again!")
                        else:
                            st.error("⚠️ Oops! The chatbot encountered a slight issue. Please try again.")
