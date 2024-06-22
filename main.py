import cv2
import numpy as np
import time
import json
import threading
from pynput import keyboard, mouse
import mss
import pygetwindow as gw
from tkinter import Label

# Initialize variables
key_log = []
mouse_log = []
frame_number = 0
keyboard_listener = None
mouse_listener = None
recording = False

# Keyboard event handler
def on_keyboard_press(key):
    global key_log, frame_number
    try:
        key_log.append({'frame': frame_number, 'key': key.char})
    except AttributeError:
        key_log.append({'frame': frame_number, 'key': str(key)})

# Mouse event handlers
def on_mouse_move(x, y):
    global mouse_log, frame_number
    mouse_log.append({'frame': frame_number, 'event_type': 'move', 'position': (x, y)})

def on_mouse_click(x, y, button, pressed):
    global mouse_log, frame_number
    if pressed:
        mouse_log.append({'frame': frame_number, 'event_type': 'click', 'position': (x, y)})

# Function to start keyboard and mouse listeners
def start_listeners():
    global keyboard_listener, mouse_listener
    keyboard_listener = keyboard.Listener(on_press=on_keyboard_press)
    mouse_listener = mouse.Listener(on_move=on_mouse_move, on_click=on_mouse_click)
    keyboard_listener.start()
    mouse_listener.start()

# Function to stop keyboard and mouse listeners
def stop_listeners():
    global keyboard_listener, mouse_listener
    keyboard_listener.stop()
    mouse_listener.stop()

def stop_recording():
    global recording
    recording = False
    stop_listeners()
    save_log("output_logs")

# Function to get the coordinates of a specific window
def get_window_coordinates(window_title):
    window = gw.getWindowsWithTitle(window_title)[0]
    return {
        'left': window.left,
        'top': window.top,
        'width': window.width,
        'height': window.height
    }

def start_screen_recording(output_folder: str, window_title: str, label: Label,fps: int=30):
    # Run the recording in a separate thread to avoid blocking the main Tkinter thread
    threading.Thread(target=record_window, args=(output_folder, window_title, label, fps)).start()

# Function to record a specific window in MP4 format with frame-based logging
def record_window(output_folder: str, window_title: str, label: Label, fps: int):
    global frame_number, recording
    window_coords = get_window_coordinates(window_title)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f"{output_folder}/gameplay.mp4", fourcc, fps, (window_coords['width'], window_coords['height']))

    recording = True

    with mss.mss() as sct:
        
        frame_duration = 1 / fps
        while recording:
            loop_start_time = time.time()
            img = np.array(sct.grab(window_coords))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            out.write(frame)
            elapsed_time = time.time() - loop_start_time
            record_elapsed_time = frame_number / fps
            mins, secs = divmod(int(record_elapsed_time), 60)
            time_format = f"Recording: {mins:02d}:{secs:02d}"
            label.config(text=time_format)
            time.sleep(max(0, frame_duration - elapsed_time))
            frame_number += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    out.release()
    label.config(text="Recording Stopped")

def save_log(output_logs):
    with open(f"{output_logs}/key_log.json", 'w') as f:
        json.dump(key_log, f, indent=4)
    with open(f"{output_logs}/mouse_log.json", 'w') as f:
        json.dump(mouse_log, f, indent=4)

# Function to start recording and logging
def main(output_logs, window_title):
    # Start listeners in a separate thread
    listener_thread = threading.Thread(target=start_listeners)
    listener_thread.start()

    # Start window recording
    record_window(output_logs, window_title)

    # Stop listeners after recording is done
    stop_listeners()

    # Wait for listeners to finish
    listener_thread.join()

    # Save logs
    save_log(output_logs=output_logs)

if __name__ == "__main__":
    main()
