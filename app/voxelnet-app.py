import cv2
import streamlit as st
import time
from utils.cameras import list_available_webcams, get_video_frame

W = 1280
H = 720

# Set up the Streamlit layout
st.title("VOXELnet app")
st.header("MUIA UAX 2023/2024")

# List available webcams
st.header("List of Available Webcams")

# Find available webcams
webcams = list_available_webcams()

if webcams:
    st.write("Available webcams:")
    for idx in webcams:
        st.write(f"Webcam index: {idx}")
else:
    st.write("No webcams found.")

# # Check if the webcams are opened successfully
# if not cap1.isOpened():
#     st.error("Error: Webcam 1 not found or cannot be accessed.")
#     cap1.release()  # Release the capture object if it was opened
#     cap1 = None

# if not cap2.isOpened():
#     st.error("Error: Webcam 2 not found or cannot be accessed.")
#     cap2.release()  # Release the capture object if it was opened
#     cap2 = None

# Step 2: Dropdown for selecting cameras
selected_l_index = st.selectbox("Select Left (L) Camera", webcams, index=0)
selected_r_index = st.selectbox("Select Right (R) Camera", webcams, index=1)

# Initialize video captures for the selected webcams
cap_l = cv2.VideoCapture(selected_l_index)
cap_r = cv2.VideoCapture(selected_r_index)

# Set the width and height (optional)
cap_l.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap_l.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cap_r.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap_r.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

# Display real-time feeds
st.write("Real-time webcam feeds:")

# Using Streamlit columns to display the feeds side by side
col1, col2 = st.columns(2)

# # Display the video feeds continuously
# while True:
#     # Capture frames from the selected webcams
#     frame_l = get_video_frame(cap_l)
#     frame_r = get_video_frame(cap_r)
    
#     # Display the left camera feed
#     with col1:
#         st.header("Left (L) Camera")
#         if frame_l is not None:
#             st.image(cv2.cvtColor(frame_l, cv2.COLOR_BGR2RGB), channels="RGB")
#         else:
#             st.write("Error: Unable to read from Left (L) Camera.")
    
#     # Display the right camera feed
#     with col2:
#         st.header("Right (R) Camera")
#         if frame_r is not None:
#             st.image(cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB), channels="RGB")
#         else:
#             st.write("Error: Unable to read from Right (R) Camera.")
    
#     # Release the capture objects if needed
#     if not cap_l.isOpened() or not cap_r.isOpened():
#         cap_l.release()
#         cap_r.release()
#         st.stop()

# Initialize video captures for the selected webcams
if 'cap_l' not in st.session_state:
    st.session_state.cap_l = cap_l

if 'cap_r' not in st.session_state:
    st.session_state.cap_r = cap_r

left_placeholder = col1.empty()
right_placeholder = col2.empty()

# Display the video feeds continuously
while True:
    # Capture frames from the selected webcams
    frame_l = get_video_frame(st.session_state.cap_l)
    frame_r = get_video_frame(st.session_state.cap_r)
    
    # Update the left camera feed
    if frame_l is not None:
        left_placeholder.image(cv2.cvtColor(frame_l, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
    else:
        left_placeholder.write("Error: Unable to read from Left (L) Camera.")
    
    # Update the right camera feed
    if frame_r is not None:
        right_placeholder.image(cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
    else:
        right_placeholder.write("Error: Unable to read from Right (R) Camera.")
    
    # Sleep for a short time to prevent excessive CPU usage
    time.sleep(0.1)