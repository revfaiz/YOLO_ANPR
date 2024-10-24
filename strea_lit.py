# streamlit_app.py
import io
import streamlit as st
import cv2
import requests
import numpy as np
from PIL import Image

# Set up the Streamlit app
st.title("Live Video Stream from Backend")

# The backend URL that streams the frames
video_url = "http://127.0.0.1:5000/video_feed"

# Create a placeholder for the video frames
frame_placeholder = st.empty()

# Connect to the video stream from the backend
with st.spinner("Connecting to video stream..."):
    video_stream = requests.get(video_url, stream=True)

    # Check if the connection is successful
    if video_stream.status_code == 200:
        st.success("Connected successfully!")

        # Iterate through the frames
        for chunk in video_stream.iter_content(chunk_size=1024):
            if b'Content-Type: image/jpeg' in chunk:
                # Search for frame boundary and extract the frame bytes
                frame_start = chunk.find(b'\xff\xd8')  # JPEG frame start
                frame_end = chunk.find(b'\xff\xd9')  # JPEG frame end
                
                if frame_start != -1 and frame_end != -1:
                    # Extract frame bytes
                    frame_bytes = chunk[frame_start:frame_end+2]

                    # Convert frame bytes to image and display it
                    frame_image = Image.open(io.BytesIO(frame_bytes))
                    frame_placeholder.image(frame_image, caption="Live Video Feed", use_column_width=True)
    else:
        st.error("Failed to connect to video stream.")
