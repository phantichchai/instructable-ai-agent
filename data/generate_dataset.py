import json
import os
from queue import Queue
import queue
import time
import cv2
import mss
import numpy as np
import threading
import pyautogui
from tools.action_key_mapping import ActionMapping, KeyBinding
from tools.genshin_impact_controller import GenshinImpactController
from tools.window import get_window_coordinates

class GenerateDataset:
    def __init__(self, controller: GenshinImpactController, dataset_dir="dataset", fps=30, buffer_size=30):
        self.controller = controller
        self.dataset_dir = dataset_dir
        self.fps = fps
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
                    video_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (width, height))
                video_writer.write(screen_img)
            except queue.Empty:
                print("Buffer is empty; waiting for frames...")
            except Exception as e:
                print(f"Error writing frame: {e}")
        
        if video_writer:
            video_writer.release()
        print(f"Video recorded and saved as: {video_filename}")

    def capture_video(self, actions, text_prompt, duration=10):
        """Captures video while performing an action, saves video and adds to dataset."""
        video_filename = f"{self.dataset_dir}/video_{len(self.actions)}.mp4"
        self.is_recording = True
        writer_thread = threading.Thread(target=self.frame_writer, args=(video_filename,))
        writer_thread.start()

        frame_count = 0
        delay = 1 / self.fps  # Delay in seconds between frames
        start_time = time.time()  # Start timer for duration

        while frame_count < duration * self.fps:  # Record for specified duration
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
        self.actions.append([action.value for action in actions])
        self.text_prompts.append(text_prompt)
        print(f"Captured video for action: {actions}, saved to {video_filename}")

    def perform_action(self, actions, duration):
        """Performs multiple actions using the controller."""
        # Press and hold all actions for the duration
        for action in actions:
            key_binding = KeyBinding[action.name]  # Get the corresponding key binding
            # Perform the action by using the key binding
            if key_binding in [KeyBinding.MOVE_FORWARD, KeyBinding.MOVE_LEFT, KeyBinding.MOVE_RIGHT, KeyBinding.MOVE_BACKWARD]:
                self.controller.move(key_binding)
            elif key_binding == KeyBinding.JUMP:
                self.controller.jump()
            elif key_binding == KeyBinding.SPRINT:
                self.controller.sprint()
            elif key_binding == KeyBinding.NORMAL_ATTACK:
                self.controller.normal_attack()
            elif key_binding == KeyBinding.ELEMENTAL_SKILL:
                self.controller.elemental_skill()
            elif key_binding == KeyBinding.ELEMENTAL_BURST:
                self.controller.elemental_burst()
            elif key_binding == KeyBinding.INTERACT:
                self.controller.interact()
            elif key_binding == KeyBinding.OPEN_MAP:
                self.controller.open_map()
            elif key_binding == KeyBinding.OPEN_INVENTORY:
                self.controller.open_inventory()
            elif key_binding.name.startswith("SWITCH_CHARACTER"):
                self.controller.switch_character(key_binding.name[-1])
            
        # Wait for the specified duration
        time.sleep(duration)

        # Release all keys after the duration
        for action in actions:
            key_binding = KeyBinding[action.name]
            if key_binding in [KeyBinding.MOVE_FORWARD, KeyBinding.MOVE_LEFT, KeyBinding.MOVE_RIGHT, KeyBinding.MOVE_BACKWARD]:
                pyautogui.keyUp(key_binding.value)  # Release the key after the duration
            elif action == ActionMapping.SPRINT:
                pyautogui.keyUp(KeyBinding.SPRINT.value)  # Release sprint key

    def generate(self, actions, text_prompt, duration):
        """Performs the action and records it with text prompt in a separate thread."""
        if isinstance(actions, ActionMapping):  # Allow single action as well
            actions = [actions]
        video_thread = threading.Thread(target=self.capture_video, args=(actions, text_prompt, duration))
        video_thread.start()  # Start the video capture in a separate thread

        self.perform_action(actions, duration)  # Perform the action

        video_thread.join()  # Wait for the video thread to finish

    def save_metadata(self):
        """Saves metadata about the dataset (actions and prompts) as a JSON file with actions as lists."""
        metadata_file = os.path.join(self.dataset_dir, "metadata.json")
        metadata = []

        # Build the metadata as a list of dictionaries
        for i, action in enumerate(self.actions):
            video_filename = f"video_{i}.mp4"
            text_prompt = self.text_prompts[i]
            metadata.append({
                "video_file": video_filename,
                "text_prompt": text_prompt,
                "actions": action  # Save the action as a list
            })

        # Write the metadata to a JSON file
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

        print(f"Saved metadata to {metadata_file}")