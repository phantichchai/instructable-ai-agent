import cv2
import json
import os
from utils import copy_file, create_folder_by_datetime

output_logs = "output_logs"

def synchronize_logs_with_frames(
    instruction,
    video_file,
    key_log_file, 
    mouse_log_file, 
    frame_logs_file
):
    # Read logs
    with open(key_log_file, 'r') as f:
        key_log = json.load(f)
    with open(mouse_log_file, 'r') as f:
        mouse_log = json.load(f)

    # Open video file
    cap = cv2.VideoCapture(video_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize frame logs
    frame_logs = {
        "instruction": instruction,
        "actions": []
    }

    folder_path = create_folder_by_datetime(output_logs)
    source = f"{output_logs}/gameplay.mp4"
    dest = f"{folder_path}/"
    copy_file(source, dest)

    for frame_num in range(frame_count):
        # Extract frame
        ret, _ = cap.read()
        if not ret:
            break
        
        # Filter logs for the current frame
        key_events = [event for event in key_log if event['frame'] == frame_num]
        mouse_events = [event for event in mouse_log if event['frame'] == frame_num]
        
        frame_logs["actions"].append({
            'frame': frame_num,
            'key_events': key_events,
            'mouse_events': mouse_events
        })
    
    # Release video capture
    cap.release()

    # Save synchronized logs
    with open(f"{folder_path}/frame_logs.json", 'w') as f:
        json.dump(frame_logs, f, indent=4)

if __name__ == "__main__":
    output_frames = "output_frames"
    output_logs = "output_logs"
    video_file = f"{output_logs}/gameplay.mp4"
    key_log_file = f"{output_logs}/key_log.json"
    mouse_log_file = f"{output_logs}/mouse_log.json"
    frame_logs_file = f"{output_logs}/frame_logs.json"

    if not os.path.exists(output_frames):
        os.makedirs(output_frames)

    synchronize_logs_with_frames(video_file, key_log_file, mouse_log_file, frame_logs_file, output_logs)