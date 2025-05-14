import cv2
import requests

API_URL = "http://192.168.1.239:8000/generate"

# Send frames and text to API
def send_frames_and_text_to_api(frames, text):
    # Convert each frame to raw bytes
    frame_bytes = [cv2.imencode('.jpg', frame)[1].tobytes() for frame in frames]
    
    # Prepare the payload
    files = {
        'frame1': ('frame1.jpg', frame_bytes[0], 'image/jpeg'),
        'frame2': ('frame2.jpg', frame_bytes[1], 'image/jpeg'),
        'frame3': ('frame3.jpg', frame_bytes[2], 'image/jpeg')
    }
    data = {'text': text}
    
    # Send POST request to API
    response = requests.post(API_URL, files=files, data=data)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None