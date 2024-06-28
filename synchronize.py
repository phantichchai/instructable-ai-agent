import cv2
import json
import os
from utils import create_folder_by_datetime
from tkinter import messagebox

output_logs = "output_logs"

def synchronize_logs_with_frames(
    instruction,
    video_file,
    key_log_file, 
    mouse_log_file
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
    dest = f"{folder_path}/gameplay.mp4"

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

    cut_off_frame = find_frames_with_keys(frame_logs)
    cut_log_after_frame(frame_logs, folder_path, cut_off_frame)
    cut_video_after_frame(source, dest, cut_off_frame)

    messagebox.showinfo("Info", "Synchronize success.")


def cut_log_after_frame(frame_logs, folder_path, cut_off_frame):
    with open(f"{folder_path}/frame_logs.json", 'w') as f:
        frame_logs['actions'] = frame_logs['actions'][:cut_off_frame]
        json.dump(frame_logs, f, indent=4)

def cut_video_after_frame(input_video_path, output_video_path, cut_off_frame):
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    
    # Get the width and height of the frames
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs
    out = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (frame_width, frame_height))

    # Read and write frames until the specified cut-off frame
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_number > cut_off_frame - 1:
            break
        out.write(frame)
        frame_number += 1

    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def find_frames_with_keys(frame_logs):
    actions = frame_logs["actions"]

    matching_frames = []

    for action in actions[::-1]:
        key_events = action['key_events']
        for key_event in key_events:
            if key_event.get("key") in ["Key.alt_l", "Key.tab"]:
                frame_number = key_event.get("frame")
                matching_frames.append(frame_number)
        if len(matching_frames) >= 2:
            break
    return matching_frames[-1]

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