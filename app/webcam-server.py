import cv2
from flask import Flask, request, Response
from utils.cameras import list_available_webcams

app = Flask(__name__)

available_webcams = list_available_webcams()
print(f"[Webcam Server] available webcams: {available_webcams}")

print(f"[Webcam Server] opening camera: {available_webcams[0]}")

if len(available_webcams)<2:
    print(f"[Webcam Server] exception: at least two cameras are needed, {len(available_webcams)} found")
    import sys
    sys.exit(0)

# Initialize webcams
camera_0 = cv2.VideoCapture(available_webcams[0])  # Left webcam
camera_1 = cv2.VideoCapture(available_webcams[1])  # Right webcam (change index if needed)

# Set desired resolution
frame_width = 640 # 1280 querter resolution
frame_height = 360 # 720 quarter resolution
camera_0.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
camera_0.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
camera_1.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
camera_1.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Desired output resolution
output_width = 256
output_height = 256

def generate_frames(camera):
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Resize frame to 224x224
            resized_frame = cv2.resize(frame, (output_width, output_height))
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', resized_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed_0')
def video_feed_0():
    mode = request.args.get('mode', 'non-inverted')
    print(mode)
    return Response(generate_frames(camera_0 if mode=="non-inverted" else camera_1),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_1')
def video_feed_1(mode: str ="non-inverted"):
    mode = request.args.get('mode', 'non-inverted')
    return Response(generate_frames(camera_1 if mode=="non-inverted" else camera_0),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(port=5001, debug=False, threaded=True)
