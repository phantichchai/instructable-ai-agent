import os
from queue import Queue
import time
import cv2
import mss
import numpy as np
import threading
from tools.genshin_impact_controller import GenshinImpactController
from tools.window import get_window_coordinates

class GenerateDataset:
    def __init__(self, controller: GenshinImpactController, dataset_dir="dataset", actual_fps=30, buffer_size=30):
        self.controller = controller
        self.dataset_dir = dataset_dir
        self.actual_fps = actual_fps
        self.frame_buffer = Queue(maxsize=buffer_size)
        self.is_recording = False
        self.actions = []
        self.text_prompts = []

        # Ensure dataset directory exists
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
    
    def capture_screen(self):
        """Captures screen using mss."""
        with mss.mss() as sct:
            screen_shot = sct.grab(get_window_coordinates("Genshin Impact"))  # Change to the correct monitor if necessary
            img = np.array(screen_shot)  # Convert screen capture to NumPy array
            img = img[:, :, :3]  # Remove alpha channel if present (RGBA to RGB)
            return img

    def frame_writer(self, video_filename):
        """Thread to write frames from the buffer to the video file."""
        video_writer = None
        while self.is_recording or not self.frame_buffer.empty():
            try:
                screen_img = self.frame_buffer.get(timeout=0.1)  # Get frame from buffer
                if video_writer is None:
                    height, width, _ = screen_img.shape
                    video_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), self.actual_fps, (width, height))
                video_writer.write(screen_img)
            except Exception as e:
                print(f"Error writing frame: {e}")
        
        if video_writer:
            video_writer.release()
        print(f"Video recorded and saved as: {video_filename}")

    def capture_video(self, action, text_prompt, duration=10):
        """Captures video while performing an action, saves video and adds to dataset."""
        video_filename = f"{self.dataset_dir}/video_{len(self.actions)}.mp4"
        self.is_recording = True
        writer_thread = threading.Thread(target=self.frame_writer, args=(video_filename,))
        writer_thread.start()

        frame_count = 0
        delay = 1 / self.actual_fps  # Delay in seconds between frames
        start_time = time.time()  # Start timer for duration

        while frame_count < duration * self.actual_fps:  # Record for specified duration
            screen_img = self.capture_screen()  # Capture screen using mss
            
            # Add frame to the buffer
            if not self.frame_buffer.full():
                self.frame_buffer.put(screen_img)
                frame_count += 1
            
            # Calculate time spent capturing and sleeping
            elapsed_time = time.time() - start_time
            remaining_time = delay * frame_count - elapsed_time
            
            # Sleep to maintain correct frame rate
            if remaining_time > 0:
                time.sleep(remaining_time)

        self.is_recording = False
        writer_thread.join()  # Wait for the writer thread to finish
        self.actions.append(action)
        self.text_prompts.append(text_prompt)
        print(f"Captured video for action: {action}, saved to {video_filename}")

    def perform_action(self, action, duration):
        """Performs the action using the controller."""
        if action == "move_forward":
            self.controller.move("move_forward", duration)
        elif action == "move_left":
            self.controller.move("move_left", duration)
        elif action == "move_right":
            self.controller.move("move_right", duration)
        elif action == "move_backward":
            self.controller.move("move_backward", duration)
        elif action == "jump":
            self.controller.jump()
        # Add more actions as needed from your controller

    def generate(self, action, text_prompt, duration):
        """Performs the action and records it with text prompt in a separate thread."""
        video_thread = threading.Thread(target=self.capture_video, args=(action, text_prompt, duration))
        video_thread.start()  # Start the video capture in a separate thread

        self.perform_action(action, duration)  # Perform the action

        video_thread.join()  # Wait for the video thread to finish

    def save_metadata(self):
        """Saves metadata about the dataset (actions and prompts)."""
        metadata_file = os.path.join(self.dataset_dir, "metadata.csv")
        with open(metadata_file, 'w') as f:
            f.write("Video File,Action,Text Prompt\n")
            for i, action in enumerate(self.actions):
                video_filename = f"video_{i}.mp4"
                text_prompt = self.text_prompts[i]
                f.write(f"{video_filename},{action},{text_prompt}\n")
        print(f"Saved metadata to {metadata_file}")
