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
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your AI Emotion Assistant. Scan an image first, and I will tailor my responses, quotes, and advice to your current mood!"}]
if "current_emotion" not in st.session_state:
    st.session_state.current_emotion = "Neutral"

AI_AVATAR = "https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Smilies/Robot.png"

# --- 🎨 AGGRESSIVE UI OVERRIDE CSS 🎨 ---
st.markdown("""
    <style>
    /* 1. HIDE HEADER & FOOTER */
    header {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    
    /* Expand the main container & Fix Title Clipping */
    .block-container {
        max-width: 950px !important;
        padding-top: 3rem !important; /* Gives the title room to breathe */
        padding-bottom: 5rem !important;
    }

    /* 2. SYMMETRICAL PREMIUM BACKGROUND */
    .stApp {
        background: radial-gradient(circle at center, #1e293b 0%, #0B0F19 100%) !important;
        background-attachment: fixed !important;
    }
    [data-testid="stAppViewContainer"] {
        background-color: transparent !important;
    }

    /* 3. WIDE SYMMETRICAL PILL TABS */
    /* Force the Streamlit radio wrapper to take full width */
    [data-testid="stRadio"] {
        width: 100% !important;
        max-width: 850px !important;
        margin: 0 auto !important;
    }
    [data-testid="stRadio"] > div {
        width: 100% !important;
    }
    div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        justify-content: space-between !important;
        background-color: rgba(255, 255, 255, 0.03) !important;
        border-radius: 50px !important;
        padding: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        margin: 0 auto 40px auto !important;
        width: 100% !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
        backdrop-filter: blur(10px) !important;
    }
    [data-testid="stRadio"] div[role="radiogroup"] > label > div:first-of-type {
        display: none !important;
    }
    div[role="radiogroup"] > label {
        flex: 1 1 100% !important; /* Stretches tabs equally */
        width: 100% !important;
        text-align: center !important;
        justify-content: center !important;
        background-color: transparent !important;
        padding: 12px 0px !important;
        border-radius: 50px !important;
        color: #94A3B8 !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
        border: 1px solid transparent !important;
        margin: 0 5px !important;
    }
    div[role="radiogroup"] > label:hover {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: #F8FAFC !important;
        transform: translateY(-2px) !important;
    }
    div[role="radiogroup"] > label[data-checked="true"] {
        background-color: #3B82F6 !important;
        color: white !important;
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.4) !important;
    }

    /* 4. CAMERA UI HACK (Transparent Box & Fixed Circular Shutter) */
    [data-testid="stCameraInput"], 
    [data-testid="stCameraInput"] > div, 
    [data-testid="stCameraInput"] > div > div {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        overflow: visible !important; 
    }
    [data-testid="stCameraInput"] video {
        transform: scaleX(-1) !important; 
        border-radius: 20px !important;
        box-shadow: 0 10px 40px rgba(0,0,0,0.5) !important;
        background-color: black !important;
    }
    [data-testid="stCameraInput"] button {
        width: 65px !important;
        height: 65px !important;
        border-radius: 50% !important;
        background-color: rgba(255,255,255,0.1) !important;
        border: 5px solid #ffffff !important;
        color: transparent !important; 
        margin: 15px auto !important;
        display: block !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3) !important;
    }
    [data-testid="stCameraInput"] button:hover {
        background-color: white !important;
        transform: scale(1.1) !important;
    }
    [data-testid="stCameraInput"] button:active {
        transform: scale(0.9) !important;
    }

    /* 5. TRANSPARENT UPLOAD BOX FIX */
    [data-testid="stFileUploader"] {
        background-color: transparent !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 2px dashed rgba(255, 255, 255, 0.2) !important;
        border-radius: 20px !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: rgba(255, 255, 255, 0.5) !important;
        background-color: rgba(255, 255, 255, 0.1) !important;
    }

    /* 6. CONTENT FADE-IN & GLASS CONTAINERS FOR RESULTS */
    @keyframes fadeIn {
        from { opacity: 0; transform: scale(0.98); }
        to { opacity: 1; transform: scale(1); }
    }
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] {
        background: rgba(15, 23, 42, 0.5);
        border-radius: 20px;
        padding: 25px;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        animation: fadeIn 0.4s ease-out forwards;
    }

    /* 7. TYPOGRAPHY SYMMETRY */
    .main-title { 
        font-size: 3.2rem; 
        font-weight: 800; 
        text-align: center; 
        margin-top: -10px; /* Reduced negative margin */
        margin-bottom: 5px; 
        background: linear-gradient(to right, #4facfe, #00f2fe); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
    }
    .sub-title { text-align: center; font-size: 1.1rem; color: #94a3b8; margin-bottom: 20px; }
    img { border-radius: 12px; }

    /* 8. KILL THE BLACK CHAT DIVIDER / BOTTOM CONTAINER */
    .stBottom {
        background-color: transparent !important;
        background: transparent !important;
    }
    .stBottom > div {
        background-color: transparent !important;
        background: transparent !important;
    }
    [data-testid="stBottom"] {
        background-color: transparent !important;
        background: transparent !important;
    }
    [data-testid="stBottom"] > div {
        background-color: transparent !important;
        background: transparent !important;
    }
    
    /* Floating chat input bar */
    [data-testid="stChatInput"] {
        padding-bottom: 15px !important;
        background: transparent !important;
    }
    [data-testid="stChatInput"] > div {
        background-color: rgba(15, 23, 42, 0.85) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 30px !important;
        backdrop-filter: blur(15px) !important;
        box-shadow: 0 8px 30px rgba(0,0,0,0.5) !important;
    }
    [data-testid="stChatInputTextArea"] { color: #F8FAFC !important; }
    [data-testid="stChatInputSubmitButton"] { color: #3B82F6 !important; }
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
    "Happy": ["Give me a happy quote! ☀️", "Recommend an upbeat song 🎵", "Tell me a joke! 😂", "What's a fun fact about happiness?"],
    "Sad": ["Give me a comforting quote 🌧️", "How can I cheer up? 🫂", "Recommend a calming song 🎧", "Write me a short uplifting poem ✨"],
    "Angry": ["How to calm down? 🧘", "Give me a peaceful quote 🍃", "Recommend relaxing ambient music 🎶", "Guide me through a breathing exercise 🌬️"],
    "Fearful": ["Give me a courageous quote 🦁", "How to overcome anxiety? 🛡️", "Recommend a soothing song 🎹", "Tell me an inspiring story of bravery 🦸"],
    "Surprised": ["Tell me a mind-blowing fact! 🤯", "Recommend an unpredictable movie 🍿", "Give me a fun trivia question 🎲", "What is the universe's biggest mystery? 🌌"],
    "Disgusted": ["Tell me a funny story to clear my mind! 🤣", "Give me a random weird fact 💡", "Recommend a wholesome video topic 🐶", "How to reset my mood? 🔄"],
    "Neutral": ["Tell me a fun fact! 🧠", "Give me a motivational quote 🚀", "Recommend a good book 📚", "Teach me something new today 🎓"]
}

# --- MAIN UI HEADER ---
st.markdown('<div class="main-title">Facial Emotion Analysis AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Advanced Emotion Recognition & Real-Time AI Companion</div>', unsafe_allow_html=True)

colA, colB, colC = st.columns([1, 2, 1])
with colB:
    use_gemini = st.toggle("🚀 Enable High-Accuracy Mode (Gemini Vision AI)", value=False)
st.write("") 

# --- THE CUSTOM "ROUTER" TABS ---
selected_tab = st.radio(
    "Navigation", 
    ["📸 Camera", "🖼️ Upload Images", "💬 AI Assistant"], 
    horizontal=True, 
    label_visibility="collapsed"
)

# --- THE AI ENGINE ---
def run_analysis(image_file, file_name="Captured Image"):
    with st.container(): 
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

                # Draw the final results
                col1, col2 = st.columns([1.5, 1], gap="large")
                with col1:
                    st.image(img_array, use_container_width=True)
                with col2:
                    st.write("") 
                    st.write("") 
                    st.metric(label="Primary Emotion", value=predicted_emotion_ui)
                    st.metric(label=f"Confidence ({model_used_text})", value=confidence_display)
        st.write("") 

# --- ROUTER LOGIC ---
if selected_tab == "📸 Camera":
    st.markdown("<h5 style='text-align: center; color: #94A3B8; font-weight: normal; margin-bottom: 10px;'>Align your face in the center</h5>", unsafe_allow_html=True)
    camera_img = st.camera_input("Smile for the camera!", label_visibility="collapsed")
    if camera_img is not None:
        run_analysis(camera_img, "Webcam Capture")

elif selected_tab == "🖼️ Upload Images":
    uploaded_imgs = st.file_uploader("Drag and drop images here", type=["jpg", "png", "jpeg"], accept_multiple_files=True, label_visibility="collapsed")
    if uploaded_imgs:
        st.success(f"Successfully loaded {len(uploaded_imgs)} image(s) into the pipeline.")
        for img in uploaded_imgs:
            run_analysis(img, img.name)

elif selected_tab == "💬 AI Assistant":
    current_mood = st.session_state.current_emotion
    emoji = emoji_map.get(current_mood, '')
    
    st.info(f"### Detected Mood: **{current_mood}** {emoji}")
    
    if chatbot_model is None:
        st.error("⚠️ Gemini API Key missing or invalid! Please check your Streamlit Secrets.")
    else:
        st.write("✨ **What would you like to do?**")
        suggestions = suggestion_dict.get(current_mood, suggestion_dict["Neutral"])
        
        suggestion_clicked = None
        sug_col1, sug_col2 = st.columns(2, gap="small")
        if sug_col1.button(suggestions[0], use_container_width=True): suggestion_clicked = suggestions[0]
        if sug_col2.button(suggestions[1], use_container_width=True): suggestion_clicked = suggestions[1]
        
        sug_col3, sug_col4 = st.columns(2, gap="small")
        if sug_col3.button(suggestions[2], use_container_width=True): suggestion_clicked = suggestions[2]
        if sug_col4.button(suggestions[3], use_container_width=True): suggestion_clicked = suggestions[3]
        
        st.write("") 

        for message in st.session_state.messages:
            avatar = AI_AVATAR if message["role"] == "assistant" else "👤"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        prompt = st.chat_input(f"Ask me something about feeling {current_mood}...")
        
        if suggestion_clicked:
            prompt = suggestion_clicked

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)

            with st.chat_message("assistant", avatar=AI_AVATAR):
                with st.spinner("Processing..."):
                    try:
                        system_prompt = f"The user's face was scanned by an AI and they look {current_mood}. Keep this in mind when answering: {prompt}"
                        response = chatbot_model.generate_content(system_prompt)
                        st.markdown(response.text)
                        st.session_state.messages.append({"role": "assistant", "content": response.text})
                    except Exception as e:
                        st.error("⚠️ Oops! The chatbot encountered a slight issue. Please try again.")
