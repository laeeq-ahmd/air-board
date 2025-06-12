import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import time
import base64
from streamlit_webrtc import webrtc_streamer
import av

# Page config
st.set_page_config(page_title="Air Board", page_icon="🖐️", layout="centered")

# Set blurred background image
def set_background(image_path):
    try:
        with open(image_path, "rb") as image:
            encoded = base64.b64encode(image.read()).decode()

        st.markdown(f"""
        <style>
        html, body, .stApp {{
            height: 100%;
            margin: 0;
            padding: 0;
            background: none;
        }}

        .stApp::before {{
            content: "";
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            filter: blur(10px);
            z-index: -1;
        }}

        .shrink-title {{
            animation: shrinkTitle 2s forwards;
            font-size: 4em;
            font-weight: bold;
            color: #6C63FF;
            text-align: center;
            margin-top: 35vh;
            font-family: 'Segoe UI', sans-serif;
        }}

        @keyframes shrinkTitle {{
            0% {{ font-size: 4em; margin-top: 35vh; }}
            100% {{ font-size: 2.6em; margin-top: 10px; }}
        }}

        .subtitle {{
            text-align: center;
            font-size: 1.3em;
            margin-top: 5px;
            color: #C4B5FD;
            font-family: 'Segoe UI', sans-serif;
        }}

        .stButton button {{
            background-color: #6C63FF !important;
            color: white !important;
            font-weight: bold;
            padding: 12px 32px;
            border-radius: 8px;
            border: none;
            font-size: 16px;
            font-family: 'Segoe UI', sans-serif;
            transition: background-color 0.3s ease;
            margin-top: 20px;
            display: block;
            margin-left: auto;
            margin-right: auto;
        }}

        .stButton button:hover {{
            background-color: #A66EFF !important;
        }}

        .stButton button:active,
        .stButton button:focus:active,
        .stButton button:focus,
        .stButton button:visited {{
            color: #E0E0E0 !important;
            outline: none !important;
            box-shadow: none !important;
        }}
        </style>
        """, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Background image not found. Please make sure 'background.png' exists.")

# Load background
set_background("background.png")

# Title
st.markdown('<div class="shrink-title">Air Board</div>', unsafe_allow_html=True)
time.sleep(2.3)
st.markdown('<div class="subtitle">A touchless drawing interface</div>', unsafe_allow_html=True)

# Setup MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Frame processor using streamlit-webrtc
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return av.VideoFrame.from_ndarray(image, format="bgr24")

# Centered layout for webcam
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    webrtc_streamer(
        key="airboard",
        video_frame_callback=video_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )