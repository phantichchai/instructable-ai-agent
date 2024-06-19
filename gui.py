import tkinter as tk
from tkinter import messagebox
import threading
import os
import main
import synchronize
import pygetwindow as gw

def start_recording():
    record_time = int(entry_record_time.get())
    window_title = clicked.get()

    if not window_title:
        messagebox.showerror("Error", "Please enter the application window title.")
        return

    frame_folder = "output_frames"
    log_folder = "output_logs"

    # Create directories if they don't exist
    os.makedirs(frame_folder, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)

    # Start recording in a separate thread to avoid blocking the GUI
    threading.Thread(target=main.main, args=(log_folder, window_title, record_time)).start()

def synchronize_logs():
    frame_folder = "output_frames"
    log_folder = "output_logs"
    video_file = f"{log_folder}/gameplay.mp4"
    key_log_file = f"{log_folder}/key_log.json"
    mouse_log_file = f"{log_folder}/mouse_log.json"

    if not os.path.exists(video_file):
        messagebox.showerror("Error", "Gameplay video not found.")
        return
    if not os.path.exists(key_log_file):
        messagebox.showerror("Error", "Key log file not found.")
        return
    if not os.path.exists(mouse_log_file):
        messagebox.showerror("Error", "Mouse log file not found.")
        return

    # Synchronize logs in a separate thread to avoid blocking the GUI
    threading.Thread(target=synchronize.synchronize_logs_with_frames, args=(video_file, key_log_file, mouse_log_file, frame_folder, log_folder)).start()

# Create the main window
root = tk.Tk()
root.title("Gameplay Recorder")
options = [
    "Genshin Impact"
]
clicked = tk.StringVar()
clicked.set("Genshin Impact")

# Create and place the widgets
tk.Label(root, text="Window Title:").grid(row=0, column=0, padx=10, pady=10)
entry_window_title = tk.OptionMenu(root, clicked, *options)
entry_window_title.grid(row=0, column=1, padx=10, pady=10)

tk.Label(root, text="Record Time (seconds):").grid(row=1, column=0, padx=10, pady=10)
entry_record_time = tk.Entry(root)
entry_record_time.grid(row=1, column=1, padx=10, pady=10)

btn_start_recording = tk.Button(root, text="Start Recording", command=start_recording)
btn_start_recording.grid(row=2, column=0, columnspan=2, pady=10)

btn_synchronize_logs = tk.Button(root, text="Synchronize Logs", command=synchronize_logs)
btn_synchronize_logs.grid(row=3, column=0, columnspan=2, pady=10)

# Start the GUI event loop
root.mainloop()
