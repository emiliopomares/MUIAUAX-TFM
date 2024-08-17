import cv2
import streamlit as st

def list_available_webcams(max_devices=10):
    available_webcams = []
    
    for device_index in range(max_devices):
        cap = cv2.VideoCapture(device_index)
        if cap.isOpened():
            available_webcams.append(device_index)
            cap.release()  # Release the capture object
        else:
            # Add this line to see which indices are not available (optional)
            # st.write(f"Webcam index {device_index} not available.")
            pass
    
    return available_webcams

# Streamlit app
st.title("List of Available Webcams")

# Find available webcams
webcams = list_available_webcams()

if webcams:
    st.write("Available webcams:")
    for idx in webcams:
        st.write(f"Webcam index: {idx}")
else:
    st.write("No webcams found.")

