import cv2

def list_available_webcams(max_devices=10):
    """Lists available connected webcams
    Parameters:
    -max_devices (int): the maximum amount to devices to look for
    Returns a list of available cams with id"""
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

def get_video_frame(cap):
    """Capture frames from the webcam"""
    ret, frame = cap.read()
    return frame