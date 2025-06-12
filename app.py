# app.py

import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image

# Page config
st.set_page_config(page_title="Hand Tracking App", page_icon="🖐️", layout="centered")

st.title("🖐️ Real-time Hand Gesture Tracking using MediaPipe")
st.markdown("This demo captures your hand gestures in real time using your webcam and displays landmarks.")

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Webcam capture function
def main():
    run = st.checkbox('Start Camera')

    FRAME_WINDOW = st.image([])

    if run:
        cap = cv2.VideoCapture(0)

        with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to grab frame")
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = hands.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Draw hand landmarks
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                FRAME_WINDOW.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        cap.release()

main()
