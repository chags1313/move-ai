import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_hands_connections = mp.solutions.hands_connections
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils 

connections = {
    'HAND_CONNECTIONS': mp_hands_connections.HAND_CONNECTIONS,
    'HAND_PALM_CONNECTIONS': mp_hands_connections.HAND_PALM_CONNECTIONS,
    'HAND_THUMB_CONNECTIONS': mp_hands_connections.HAND_THUMB_CONNECTIONS,
    'HAND_INDEX_FINGER_CONNECTIONS': mp_hands_connections.HAND_INDEX_FINGER_CONNECTIONS,
    'HAND_MIDDLE_FINGER_CONNECTIONS': mp_hands_connections.HAND_MIDDLE_FINGER_CONNECTIONS,
    'HAND_RING_FINGER_CONNECTIONS': mp_hands_connections.HAND_RING_FINGER_CONNECTIONS,
    'HAND_PINKY_FINGER_CONNECTIONS': mp_hands_connections.HAND_PINKY_FINGER_CONNECTIONS,
}

draw_background = st.checkbox("Draw background", value=True)
selected_connection = st.selectbox("Select connections to draw", list(connections.keys()))

def process_hands(frame):
  img = frame.to_ndarray(format="bgr24")
  results = hands.process(img)
  output_img = img if draw_background else np.zeros_like(img)  
  if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
      mp_draw.draw_landmarks(output_img, hand_landmarks, connections[selected_connection])    
  return av.VideoFrame.from_ndarray(output_img, format="bgr24")

webrtc_streamer(
  key="streamer", 
  video_frame_callback=process_hands,
  media_stream_constraints={"video": True, "audio": False},
  )
