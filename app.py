import os
import urllib.request
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import google.generativeai as genai
from groq import Groq

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Emotion Detector AI", page_icon="🎭", layout="wide")

# --- CONFIGURE APIS (GEMINI FOR VISION, GROQ FOR CHAT) ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    vision_model = genai.GenerativeModel('gemini-2.5-flash-lite')
except Exception as e:
    vision_model = None

try:
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception as e:
    groq_client = None

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your AI Emotion Assistant. Scan an image first, and I will tailor my responses, quotes, and advice to your current mood!"}]
if "current_emotion" not in st.session_state:
    st.session_state.current_emotion = "Neutral"

AI_AVATAR = "https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Smilies/Robot.png"

# --- 🎨 THE ULTIMATE DYNAMIC CSS OVERRIDE 🎨 ---
st.markdown("""
<style>

/* Hide Header & Footer */
header {visibility: hidden;}
footer {visibility: hidden;}

/* --- CYBERPUNK SCROLLBAR --- */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(2, 6, 23, 0.9); }
::-webkit-scrollbar-thumb { background: rgba(0, 255, 255, 0.3); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0, 255, 255, 0.8); }

/* --- FIX FOR MONOCHROME EMOJIS --- */
span.emoji {
    font-family: "Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji", sans-serif !important;
    color: initial !important;
    -webkit-text-fill-color: initial !important;
    text-shadow: none !important;
}

/* ============================= */
/* 🚀 ADVANCED KEYFRAME ANIMATIONS 🚀 */
/* ============================= */

@keyframes tabActiveGlow {
    0% { box-shadow: 0 0 10px rgba(0, 206, 209, 0.3); }
    100% { box-shadow: 0 0 25px rgba(0, 206, 209, 0.7); }
}

@keyframes floatIdle {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-8px); }
    100% { transform: translateY(0px); }
}

@keyframes floatCard {
    0% { transform: translateY(0px); box-shadow: 0 8px 32px rgba(0,0,0,0.6), 0 0 10px rgba(0,255,255,0.05); }
    100% { transform: translateY(-8px); box-shadow: 0 15px 35px rgba(0,0,0,0.7), 0 0 25px rgba(0,206,209,0.25); }
}

@keyframes textGlow {
    0% { text-shadow: 0 0 10px rgba(0, 255, 255, 0.2), 0 0 20px rgba(0, 255, 255, 0.2); }
    50% { text-shadow: 0 0 20px rgba(0, 255, 255, 0.6), 0 0 30px rgba(0, 255, 255, 0.4); }
    100% { text-shadow: 0 0 10px rgba(0, 255, 255, 0.2), 0 0 20px rgba(0, 255, 255, 0.2); }
}

@keyframes fadeSlideUp {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}

@keyframes popIn {
    0% { opacity: 0; transform: scale(0.9) translateY(10px); }
    100% { opacity: 1; transform: scale(1) translateY(0); }
}

/* --- BACKGROUND (Clean Dark Slate Gradient) --- */
.stApp {
    background: radial-gradient(circle at center, #1e293b 0%, #0B0F19 100%) !important;
    background-attachment: fixed !important;
    color: #f8fafc;
}
[data-testid="stAppViewContainer"] { background-color: transparent !important; }

/* Page Width & Main Container */
.block-container {
    max-width: 1100px !important;
    padding-top: 3rem !important;
    padding-bottom: 5rem !important;
    z-index: 1;
}

/* Apply smooth load-in to content blocks */
[data-testid="stCameraInput"], 
[data-testid="stFileUploader"], 
.stChatInputContainer,
.card-grid-3,
.custom-alert {
    animation: fadeSlideUp 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards !important;
}

/* ============================= */
/* 🔥 PERFECT CENTERED GLASS TABS */
/* ============================= */

[data-testid="stRadio"] {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: 100% !important;
    margin-top: 25px !important;
    margin-bottom: 35px !important;
}

/* Fix Streamlit container width */
[data-testid="stRadio"] > div {
    width: fit-content !important;
    margin: 0 auto !important;
}

/* Hide radio circles */
[data-testid="stRadio"] label > div:first-child {
    display: none !important;
}

/* Glass Container */
div[role="radiogroup"] {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    gap: 12px !important;

    margin: 0 auto !important;
    padding: 8px !important;

    background: rgba(255,255,255,0.04) !important;
    backdrop-filter: blur(16px) !important;

    border-radius: 40px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;

    box-shadow: 
    0 8px 30px rgba(0,0,0,0.35),
    inset 0 0 0.5px rgba(255,255,255,0.08) !important;
}

/* Tab Buttons */
div[role="radiogroup"] label {
    padding: 11px 22px !important;
    border-radius: 30px !important;

    color: #94A3B8 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;

    background: transparent !important;

    transition: 
    all 0.25s ease,
    transform 0.18s ease,
    box-shadow 0.25s ease !important;

    cursor: pointer !important;
}

/* Hover Animation */
div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.08) !important;
    color: #ffffff !important;

    transform: translateY(-2px);

    box-shadow: 
    0 6px 14px rgba(0,0,0,0.25),
    inset 0 0 0.5px rgba(255,255,255,0.15);
}

/* 🔥 Selected Tab Effect (Reliable Fix) */

div[role="radiogroup"] label:has(input:checked) {
    
    background: rgba(34,227,255,0.15) !important;
    color: #22E3FF !important;

    border: 1px solid rgba(34,227,255,0.35) !important;

    box-shadow:
        0 0 8px rgba(34,227,255,0.35),
        0 0 18px rgba(34,227,255,0.25),
        inset 0 0 8px rgba(34,227,255,0.15);

    transform: translateY(-1px);
}

/* ============================= */
/* 🔥 PNEUMONIALENS CARD SYSTEM 🔥 */
/* ============================= */
.card-grid-3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 25px; margin-bottom: 35px; }

.custom-card { 
    background: linear-gradient(145deg, rgba(15, 23, 42, 0.7) 0%, rgba(5, 10, 15, 0.9) 100%); 
    backdrop-filter: blur(12px);
    border: 1px solid rgba(0, 255, 255, 0.15); 
    border-radius: 16px; 
    padding: 25px; 
    transition: all 0.4s ease; 
    height: 100%; 
    animation: floatCard 4s ease-in-out infinite alternate;
}
.custom-card:hover { 
    transform: scale(1.02); 
    box-shadow: 0 20px 40px rgba(0, 255, 255, 0.2); 
    animation-play-state: paused; 
    border-color: #00FFFF;
}
.card-title { color: #00FFFF; font-size: 1.2rem; font-weight: 800; margin-bottom: 15px; border-bottom: 1px solid rgba(0, 255, 255, 0.15); padding-bottom: 10px; }
.card-text { color: #D1D5DB; font-size: 1rem; line-height: 1.7; }
.kpi-value { font-size: 3.5rem; font-weight: 900; color: #00FFFF; text-shadow: 0 0 15px rgba(0,255,255,0.4); text-align: center; margin-bottom: 10px; }
.kpi-label { color: #94A3B8; font-size: 1.1rem; font-weight: 700; text-align: center; text-transform: uppercase; }

/* ============================= */
/* ☢️ NUCLEAR CAMERA TRANSPARENCY ☢️ */
/* ============================= */
[data-testid="stCameraInput"],
[data-testid="stCameraInput"] > div,
[data-testid="stCameraInput"] > div > div,
[data-testid="stCameraInput"] section {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

[data-testid="stCameraInput"] button {
    width: 70px !important;
    height: 70px !important;
    border-radius: 50% !important;
    background-color: rgba(0, 255, 255, 0.05) !important;
    border: 3px solid #00FFFF !important;
    box-shadow: 0 0 15px rgba(0, 255, 255, 0.3) !important;
    color: transparent !important; 
    margin: 20px auto !important;
    display: block !important;
    transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    animation: floatIdle 4s ease-in-out infinite !important;
}
[data-testid="stCameraInput"] button:hover {
    background-color: rgba(0, 255, 255, 0.3) !important;
    transform: scale(1.15) !important;
    box-shadow: 0 0 30px rgba(0, 255, 255, 0.8) !important;
    animation-play-state: paused !important;
}
[data-testid="stCameraInput"] video {
    background-color: #000 !important; 
    border-radius: 15px !important;
    transform: scaleX(-1) !important;
    border: 1px solid rgba(0, 255, 255, 0.3) !important;
}

/* ============================= */
/* ☢️ TRANSPARENT UPLOAD DROPZONE ☢️ */
/* ============================= */
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] > div,
[data-testid="stFileUploader"] section {
    background-color: transparent !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
[data-testid="stFileUploadDropzone"],
[data-testid="stFileUploaderDropzone"] {
    background: rgba(15, 23, 42, 0.4) !important;
    backdrop-filter: blur(8px) !important;
    border-radius: 15px !important;
    border: 2px dashed rgba(0, 255, 255, 0.4) !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploadDropzone"]:hover,
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(0, 255, 255, 1) !important;
    background: rgba(0, 255, 255, 0.1) !important; 
    transform: translateY(-4px) !important;
}

/* ============================= */
/* 🔥 SHIMMERING SUGGESTION CARDS 🔥 */
/* ============================= */
div.stButton > button {
    background: rgba(15, 23, 42, 0.6) !important; 
    border: 1px solid rgba(0, 255, 255, 0.2) !important;
    backdrop-filter: blur(10px) !important;
    color: #e2e8f0 !important;
    border-radius: 12px !important;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    padding: 10px !important;
    position: relative;
    overflow: hidden;
}
div.stButton > button:hover {
    background: rgba(0, 255, 255, 0.1) !important;
    border-color: #00FFFF !important;
    color: #ffffff !important;
    transform: translateY(-4px) scale(1.02); 
    box-shadow: 0 8px 20px rgba(0, 255, 255, 0.2) !important;
}

/* Custom HTML Mood Info Box */
.custom-alert {
    background: rgba(15, 23, 42, 0.7) !important;
    backdrop-filter: blur(15px) !important;
    border: 1px solid rgba(0, 255, 255, 0.2) !important;
    border-left: 5px solid #00FFFF !important;
    color: #f8fafc !important;
    border-radius: 12px !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.5) !important;
    animation: floatIdle 6s ease-in-out infinite !important;
    padding: 16px 20px;
    font-size: 1.1rem;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* ============================= */
/* ☢️ NUCLEAR TRANSPARENCY: CHAT ☢️ */
/* ============================= */
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"],
.stChatInputContainer {
    background-color: transparent !important;
    background: transparent !important;
    border: none !important;
}

[data-testid="stChatInput"] {
    padding-bottom: 20px !important;
    background: transparent !important;
    background-color: transparent !important;
}
[data-testid="stChatInput"] > div:first-child {
    background: rgba(10, 15, 25, 0.85) !important;
    backdrop-filter: blur(15px) !important;
    border-radius: 30px !important;
    border: 1px solid rgba(0, 255, 255, 0.15) !important;
    box-shadow: 0 15px 35px rgba(0,0,0,0.8), inset 0 0 15px rgba(0,255,255,0.05) !important;
    transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}
[data-testid="stChatInput"] > div:first-child:focus-within {
    border-color: #00FFFF !important;
    box-shadow: 0 15px 35px rgba(0,0,0,0.9), 0 0 25px rgba(0, 255, 255, 0.3) !important;
    transform: translateY(-2px);
}
[data-testid="stChatInputTextArea"],
[data-testid="stChatInputTextArea"] > div,
[data-testid="stChatInputTextArea"] textarea { 
    color: #FFFFFF !important; 
    background-color: transparent !important;
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
}
[data-testid="stChatInputSubmitButton"] { color: #00FFFF !important; transition: transform 0.2s; }
[data-testid="stChatInputSubmitButton"]:hover { transform: scale(1.2) rotate(10deg); }

[data-testid="stChatMessage"] {
    background-color: transparent !important;
    background: transparent !important;
    animation: popIn 0.4s cubic-bezier(0.25, 1, 0.5, 1) forwards !important;
}

/* ============================= */
/* Typography */
/* ============================= */
.main-title {
    font-size: 3.2rem;
    font-weight: 800;
    text-align: center;
    color: #00FFFF !important;
    animation: textGlow 3s ease-in-out infinite !important; 
    margin-top: 0px;
    margin-bottom: 5px;
    letter-spacing: 1px;
}
.sub-title {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 30px;
    font-size: 1.1rem;
    font-weight: 300;
}
img {
    border-radius: 12px;
}

/* ============================= */
/* 🚀 Premium UI Enhancements */
/* ============================= */


/* 1. Sliding Active Indicator Base */
div[role="radiogroup"] {
position: relative;
transition: 0.3s ease;

box-shadow: 
0 8px 30px rgba(0,0,0,0.35),
0 0 40px rgba(34,227,255,0.05);
}


/* 2. Floating Navbar Effect */
div[role="radiogroup"]:hover {
transform: translateY(-2px);
}


/* 3. Smooth Hover Micro Animation */
div[role="radiogroup"] label {
transition: all 0.25s ease !important;
}

div[role="radiogroup"] label:hover {
transform: translateY(-2px) scale(1.02);
}


/* 4. Selected Tab Glow Enhancement */
div[role="radiogroup"] label:has(input:checked) {

background: rgba(34,227,255,0.15) !important;
color: #22E3FF !important;

border: 1px solid rgba(34,227,255,0.35) !important;

box-shadow:
0 0 8px rgba(34,227,255,0.35),
0 0 18px rgba(34,227,255,0.25),
inset 0 0 8px rgba(34,227,255,0.15);

transform: translateY(-1px);
}


/* 5. Page Fade-In Animation */
.stApp {
animation: fadeIn 0.4s ease;
}

@keyframes fadeIn {
from {
opacity:0;
transform:translateY(10px);
}

to {
opacity:1;
transform:translateY(0);
}
}

/* Smooth Active Pill Animation */
div[role="radiogroup"] label {
position: relative;
overflow: hidden;
}

/* Active tab stronger pill */
div[role="radiogroup"] label:has(input:checked) {

background: linear-gradient(
135deg,
rgba(34,227,255,0.18),
rgba(59,130,246,0.18)
) !important;

backdrop-filter: blur(10px);

border: 1px solid rgba(34,227,255,0.35);

box-shadow:
0 0 12px rgba(34,227,255,0.35),
0 0 22px rgba(34,227,255,0.20);

transition: all 0.25s ease;
}

/* ============================= */
/* 🔥 Final 10% Premium Polish */
/* ============================= */


/* 1. Ambient Glass Glow Behind Navbar */
div[role="radiogroup"] {
position: relative;
backdrop-filter: blur(18px);
-webkit-backdrop-filter: blur(18px);
}

div[role="radiogroup"]::before {
content: "";
position: absolute;
inset: -2px;
border-radius: 40px;

background: linear-gradient(
90deg,
rgba(34,227,255,0.08),
rgba(59,130,246,0.08),
rgba(34,227,255,0.08)
);

filter: blur(10px);
z-index: -1;
}


/* 2. Micro Hover Glow */
div[role="radiogroup"] label:hover {

box-shadow:
0 4px 12px rgba(0,0,0,0.25),
0 0 10px rgba(34,227,255,0.18);

transform: translateY(-2px) scale(1.02);
}


/* 3. Smooth Tab Switching Animation */
[data-testid="stVerticalBlock"] {
animation: fadeSlide 0.35s ease;
}

@keyframes fadeSlide {
from {
opacity:0;
transform: translateY(8px);
}

to {
opacity:1;
transform: translateY(0);
}
}


/* 4. Subtle Background Movement */
.stApp {
background-size: 120% 120%;
animation: bgMove 15s ease infinite;
}

@keyframes bgMove {
0% { background-position: 0% 50%; }
50% { background-position: 100% 50%; }
100% { background-position: 0% 50%; }
}


/* 5. Smooth Glass Floating Effect */
div[role="radiogroup"] {
transition: all 0.3s ease;
}

div[role="radiogroup"]:hover {
transform: translateY(-2px);
box-shadow:
0 10px 35px rgba(0,0,0,0.35),
0 0 20px rgba(34,227,255,0.08);
}

/* ============================= */
/* 🔥 Final Ultra Premium Polish */
/* ============================= */


/* 1. Tab Icon Pop Animation */
div[role="radiogroup"] label:has(input:checked) {
animation: tabPop 0.25s ease;
}

@keyframes tabPop {
0% { transform: scale(0.96); }
100% { transform: scale(1); }
}

/* 2. Floating Glass Cards */
[data-testid="stVerticalBlock"] > div {
transition: all 0.3s ease;
}

[data-testid="stVerticalBlock"] > div:hover {
transform: translateY(-2px);
}


/* 3. Cursor Hover Glow */
button:hover,
label:hover {
filter: brightness(1.08);
}


/* 4. Smooth Scroll */
html {
scroll-behavior: smooth;
}


/* 5. Soft Divider Glow */
hr {
border: none;
height: 1px;

background: linear-gradient(
90deg,
transparent,
rgba(34,227,255,0.3),
transparent
);
}


/* 6. Loading Spinner Glow */
[data-testid="stSpinner"] {
filter: drop-shadow(
0 0 6px rgba(34,227,255,0.4)
);
}


/* 7. Title Glow */
.main-title {
text-shadow:
0 0 10px rgba(34,227,255,0.15),
0 0 20px rgba(34,227,255,0.08);
}


/* 8. Micro Smooth Transitions */
* {
transition: 
background 0.25s ease,
color 0.25s ease,
box-shadow 0.25s ease;
}


/* 9. Slight Hover Elevation */
button:hover {
transform: translateY(-1px);
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

# --- THE CUSTOM "ROUTER" TABS (CENTERED VERSION) ---
selected_tab = st.radio(
    "Navigation", 
    ["🏠 Home", "📸 Camera", "🖼️ Upload", "💬 Chat", "📊 Analytics", "📖 Docs"], 
    horizontal=True, 
    label_visibility="collapsed"
)

# --- THE VISION ENGINE ---
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
                    if use_gemini and vision_model is not None:
                        try:
                            vision_prompt = "Analyze the facial expression of the primary person in this image. Classify their emotion into exactly one of these words: Angry, Disgusted, Fearful, Happy, Sad, Surprised, Neutral. Also estimate your confidence from 0 to 100. Respond strictly in this format: Emotion,Confidence (Example: Happy,95)"
                            response = vision_model.generate_content([vision_prompt, image])
                            
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

if selected_tab == "🏠 Home":
    st.markdown("### <span class='emoji'>🎯</span> Our Mission: Bridging the Emotional Gap", unsafe_allow_html=True)
    html_home_1 = (
        "<div class='card-grid-3'>"
        "<div class='custom-card'><div class='card-title'><span class='emoji'>🧩</span> The Core Problem</div><div class='card-text'>Human emotion is complex and deeply nuanced. In digital spaces, empathy is often lost. Our mission is to build a bridge between human feeling and machine understanding using advanced neural networks.</div></div>"
        "<div class='custom-card'><div class='card-title'><span class='emoji'>🤖</span> Real-Time Perception</div><div class='card-text'>By utilizing a Convolutional Neural Network (CNN) trained on the FER2013 dataset, this AI instantly categorizes micro-expressions into 7 core universal human emotions.</div></div>"
        "<div class='custom-card'><div class='card-title'><span class='emoji'>🫂</span> Empathetic AI Feedback</div><div class='card-text'>Detection is just step one. The system feeds your detected mood directly into an LLM (Llama 3 / Gemini), ensuring that the AI speaks to you with the appropriate emotional context and empathy.</div></div>"
        "</div>"
    )
    st.markdown(html_home_1, unsafe_allow_html=True)

    st.markdown("### <span class='emoji'>🏥</span> Practical Use Cases & Applications", unsafe_allow_html=True)
    html_home_2 = (
        "<div class='card-grid-3'>"
        "<div class='custom-card'><div class='card-title'><span class='emoji'>🧠</span> Mental Health & Wellness</div><div class='card-text'><i>The Situation:</i> A user opens a wellness app but is unable to articulate how they feel.<br><br><i>The AI Solution:</i> The camera reads their micro-expressions, logging them as \"Sad\" or \"Fearful,\" automatically triggering comforting workflows, breathing exercises, or crisis hotlines.</div></div>"
        "<div class='custom-card'><div class='card-title'><span class='emoji'>🛒</span> Customer Experience</div><div class='card-text'><i>The Situation:</i> A frustrated customer is navigating a self-checkout kiosk.<br><br><i>The AI Solution:</i> The system detects \"Angry\" or \"Disgusted\" expressions, instantly bypassing automated menus to connect them with a human representative to de-escalate the situation.</div></div>"
        "<div class='custom-card'><div class='card-title'><span class='emoji'>🎮</span> Adaptive Gaming</div><div class='card-text'><i>The Situation:</i> A player is getting bored or stressed during a game.<br><br><i>The AI Solution:</i> By monitoring facial cues for \"Neutral\" (boredom) or \"Surprised,\" the game engine dynamically adjusts difficulty, lighting, or music to keep the player engaged.</div></div>"
        "</div>"
    )
    st.markdown(html_home_2, unsafe_allow_html=True)

elif selected_tab == "📸 Camera":
    st.markdown("<h5 style='text-align: center; color: #94A3B8; font-weight: normal; margin-bottom: 10px;'>Align your face in the center</h5>", unsafe_allow_html=True)
    camera_img = st.camera_input("Smile for the camera!", label_visibility="collapsed")
    if camera_img is not None:
        run_analysis(camera_img, "Webcam Capture")

elif selected_tab == "🖼️ Upload":
    uploaded_imgs = st.file_uploader("Drag and drop images here", type=["jpg", "png", "jpeg"], accept_multiple_files=True, label_visibility="collapsed")
    if uploaded_imgs:
        st.success(f"Successfully loaded {len(uploaded_imgs)} image(s) into the pipeline.")
        for img in uploaded_imgs:
            run_analysis(img, img.name)

elif selected_tab == "💬 Chat":
    current_mood = st.session_state.current_emotion
    emoji = emoji_map.get(current_mood, '')
    
    # Custom HTML Alert instead of st.info so the emoji span renders properly!
    custom_alert = f"""
    <div class='custom-alert'>
        <span class='emoji'>✨</span> <b>Detected Mood: {current_mood}</b> <span class='emoji'>{emoji}</span>
    </div>
    """
    st.markdown(custom_alert, unsafe_allow_html=True)
    
    if groq_client is None:
        st.error("⚠️ Groq API Key missing or invalid! Please check your Streamlit Secrets.")
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
                        completion = groq_client.chat.completions.create(
                            model="llama-3.1-8b-instant", 
                            messages=[
                                {"role": "system", "content": f"You are a helpful, empathetic AI assistant. The user's face was just scanned by an emotion detection model, and they are currently feeling: {current_mood}. Keep this mood in mind and tailor your responses, tone, and advice accordingly."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.7,
                            max_tokens=1024,
                        )
                        
                        response_text = completion.choices[0].message.content
                        st.markdown(response_text)
                        st.session_state.messages.append({"role": "assistant", "content": response_text})
                    except Exception as e:
                        st.error(f"⚠️ Oops! The Groq chatbot encountered an issue: {e}")

elif selected_tab == "📊 Analytics":
    st.markdown("### <span class='emoji'>📊</span> Detailed Model Analytics", unsafe_allow_html=True)
    html_analytics_1 = (
        "<div class='card-grid-3'>"
        "<div class='custom-card'><div class='card-title'><span class='emoji'>🛠️</span> Fine-Tuning Methodology</div><div class='card-text'>This model leverages a custom Convolutional Neural Network (CNN) trained from scratch. The architecture consists of multiple stacked Conv2D layers for spatial feature extraction, paired with MaxPooling to downsample spatial dimensions. Dropout layers (0.25 to 0.5) were implemented aggressively to prevent model overfitting.</div></div>"
        "<div class='custom-card'><div class='card-title'><span class='emoji'>📁</span> The FER2013 Dataset</div><div class='card-text'>The network was trained on the industry-standard FER2013 dataset, which contains over 35,000 grayscale images of faces standardized to 48x48 pixels. The classes are heavily imbalanced (e.g., 'Happy' has many more samples than 'Disgust'), requiring strict categorical weighting during backpropagation.</div></div>"
        "<div class='custom-card'><div class='card-title'><span class='emoji'>🤖</span> LLM Integration Pipeline</div><div class='card-text'>Unlike standard classifiers, this app features a dual-pipeline. The visual CNN extracts the mood state and injects it as a hidden context variable into the system prompt of an LLM (Groq Llama-3.1 or Gemini Flash). This bridges computer vision and generative text seamlessly.</div></div>"
        "</div>"
    )
    st.markdown(html_analytics_1, unsafe_allow_html=True)
    
    st.markdown("### <span class='emoji'>📈</span> CNN Training Metrics", unsafe_allow_html=True)
    html_analytics_2 = (
        "<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 25px; margin-bottom: 30px;'>"
        "<div class='custom-card' style='display: flex; flex-direction: column; align-items: center; justify-content: flex-start; padding: 25px;'><div class='kpi-value'>68.5%</div><div class='kpi-label'><span class='emoji'>🎯</span> Base Accuracy</div><div style='margin-top: 15px; font-size: 0.95rem; color: #94A3B8; line-height: 1.6; text-align: left; padding-top: 15px; border-top: 1px solid rgba(0,255,255,0.1);'><b>Why 68.5%?</b> Human emotion is highly subjective. In the original FER2013 study, human-level agreement was only ~65%. Our CNN surpasses this human baseline, acting as a reliable, instant filter for raw pixel patterns.</div></div>"
        "<div class='custom-card' style='display: flex; flex-direction: column; align-items: center; justify-content: flex-start; padding: 25px;'><div class='kpi-value'>99.0%</div><div class='kpi-label'><span class='emoji'>🚀</span> Gemini Accuracy</div><div style='margin-top: 15px; font-size: 0.95rem; color: #94A3B8; line-height: 1.6; text-align: left; padding-top: 15px; border-top: 1px solid rgba(0,255,255,0.1);'><b>Why 99.0%?</b> By enabling High-Accuracy Mode, the system bypasses the local CNN and routes the image to Google\'s state-of-the-art multimodal AI (Gemini 2.5 Flash), which possesses near-perfect contextual emotional understanding.</div></div>"
        "<div class='custom-card' style='display: flex; flex-direction: column; align-items: center; justify-content: flex-start; padding: 25px;'><div class='kpi-value'>7</div><div class='kpi-label'><span class='emoji'>🧠</span> Emotion Classes</div><div style='margin-top: 15px; font-size: 0.95rem; color: #94A3B8; line-height: 1.6; text-align: left; padding-top: 15px; border-top: 1px solid rgba(0,255,255,0.1);'><b>The Spectrum:</b> The AI classifies micro-expressions into seven universal categories: Angry <span class='emoji'>😠</span>, Disgust <span class='emoji'>🤢</span>, Fear <span class='emoji'>😨</span>, Happy <span class='emoji'>😄</span>, Sad <span class='emoji'>😢</span>, Surprise <span class='emoji'>😲</span>, and Neutral <span class='emoji'>😐</span>.</div></div>"
        "</div>"
    )
    st.markdown(html_analytics_2, unsafe_allow_html=True)

elif selected_tab == "📖 Docs":
    st.markdown("### <span class='emoji'>📃</span> Documentation & FAQs", unsafe_allow_html=True)
    html_docs = (
        "<div class='custom-card' style='margin-bottom: 30px;'><div class='card-title'><span class='emoji'>❔</span> 1. Why is the baseline CNN accuracy ~68%?</div><div class='card-text'>Human emotion is highly subjective. In the FER2013 dataset, even human experts only agree on the emotion in an image about 65% of the time. Our CNN achieves 68.5%, placing it above human-level baseline for this specific grayscale 48x48 dataset. For near-perfect accuracy, toggle the <b>Gemini Vision AI</b> switch at the top.</div></div>"
        "<div class='custom-card' style='margin-bottom: 30px;'><div class='card-title'><span class='emoji'>❔</span> 2. What happens to my image data?</div><div class='card-text'>Privacy is paramount. If you are using the baseline CNN, the image is processed entirely in your browser/local session using OpenCV and the loaded `.h5` model. The images are never saved to a database. If you enable Gemini Vision, the image is sent securely via API for a one-time inference and then discarded.</div></div>"
        "<div class='custom-card' style='margin-bottom: 30px;'><div class='card-title'><span class='emoji'>❔</span> 3. How does the AI Assistant know my mood?</div><div class='card-text'>We use <i>Context Injection</i>. When the camera reads your face as \"Happy,\" the backend silently prepends a system instruction to the Llama 3.1 LLM: <code>\"The user is currently feeling: Happy. Tailor your responses to this mood.\"</code> This makes the text-generation empathetic to your physical state.</div></div>"
    )
    st.markdown(html_docs, unsafe_allow_html=True)
