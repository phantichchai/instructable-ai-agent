import tkinter as tk
from tkinter import messagebox
import threading
import os
import main
import synchronize

output_frames = "output_frames"
output_logs = "output_logs"

# Function to start the countdown
def start_countdown():
    try:
        countdown_time = int(entry_record_time.get())
        countdown(countdown_time)
    except ValueError:
        recording_label.config(text="Please enter a valid integer")

# Function to handle the countdown before recording
def countdown(time_left):
    if time_left > 0:
        mins, secs = divmod(time_left, 60)
        time_format = f"Countdown: {mins:02d}:{secs:02d}"
        recording_label.config(text=time_format)
        root.after(1000, countdown, time_left - 1)
    else:
        recording_label.config(text="Recording Started!")
        
        window_title = clicked.get()
        if not window_title:
            messagebox.showerror("Error", "Please enter the application window title.")
            return
        main.start_screen_recording(output_logs, window_title, recording_label)
        main.start_listeners()
    
def synchronize_logs():
    instruction = entry_instruction_text.get()
    video_file = f"{output_logs}/gameplay.mp4"
    key_log_file = f"{output_logs}/key_log.json"
    mouse_log_file = f"{output_logs}/mouse_log.json"

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
    threading.Thread(target=synchronize.synchronize_logs_with_frames, args=(instruction, video_file, key_log_file, mouse_log_file, output_frames)).start()

# Create the main window
root = tk.Tk()
root.title("Gameplay Recorder")
options = [
    "Genshin Impact",
    "Whthering Waves"
]
clicked = tk.StringVar()
clicked.set("Genshin Impact")

# Create and place the widgets
tk.Label(root, text="Window Title:").grid(row=0, column=0, padx=10, pady=10)
entry_window_title = tk.OptionMenu(root, clicked, *options)
entry_window_title.grid(row=0, column=1, columnspan=1, padx=10, pady=10)

tk.Label(root, text="Instruction:").grid(row=1, column=0, padx=10, pady=10)
entry_instruction_text = tk.Entry(root)
entry_instruction_text.grid(row=1, column=1, columnspan=1, padx=10, pady=10)

tk.Label(root, text="Countdown Time (seconds):").grid(row=2, column=0, padx=10, pady=10)
entry_record_time = tk.Entry(root)
entry_record_time.grid(row=2, column=1, columnspan=1, padx=10, pady=10)

btn_start_recording = tk.Button(root, text="Start Recording", command=start_countdown)
btn_start_recording.grid(row=3, column=1, columnspan=1, pady=10)

btn_stop_recording = tk.Button(root, text="Stop Recording", command=main.stop_recording)
btn_stop_recording.grid(row=3, column=2, columnspan=1, padx=10, pady=10)

recording_label = tk.Label(root, text="")
recording_label.grid(row=3, column=0, padx=10, pady=10)

btn_synchronize_logs = tk.Button(root, text="Synchronize Logs", command=synchronize_logs)
btn_synchronize_logs.grid(row=4, column=1, columnspan=1, padx=10, pady=10)

# Start the GUI event loop
root.mainloop()
