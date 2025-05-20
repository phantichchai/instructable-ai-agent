import json
import os
import queue
import time
import cv2
import mss
import numpy as np
import threading
import shutil
from queue import Queue
from tools.genshin.mapping import ActionMapping
from tools.genshin.controller import GenshinImpactController
from tools.window import get_window_coordinates
from typing import List, Union

class GenerateDataset:
    def __init__(
        self,
        controller: GenshinImpactController,
        dataset_dir="dataset",
        fps=30,
        buffer_size=30,
        overwrite=False
    ):
        self.controller = controller
        self.dataset_dir = dataset_dir
        self.fps = fps
        self.frame_buffer: Queue[np.ndarray] = Queue(maxsize=buffer_size)
        self.is_recording = False
        self.actions = []
        self.text_prompts = []
        self.video_filenames = []
        self.overwrite = overwrite

        if overwrite and os.path.exists(self.dataset_dir):
            shutil.rmtree(self.dataset_dir)

        os.makedirs(self.dataset_dir, exist_ok=True)

        # Determine starting index
        if overwrite:
            self.next_video_index = 0
        else:
            self.next_video_index = self._count_existing_videos()

    def _count_existing_videos(self):
        """Counts .mp4 files in the dataset directory."""
        return len([f for f in os.listdir(self.dataset_dir) if f.endswith(".mp4")])

    
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

    def capture_video(self, actions: list[ActionMapping], text_prompt, duration=10):
        """Captures video while performing an action, saves video and adds to dataset."""
        video_filename = os.path.join(self.dataset_dir, f"video_{self.next_video_index}.mp4")
        self.next_video_index += 1  # Increment for the next video
        self.video_filenames.append(os.path.basename(video_filename))  
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

    def perform_action(self, action_plan: list[tuple[ActionMapping, float]]):
        def hold_and_release(action, delay):
            self.controller.execute_action(action)
            time.sleep(delay)
            self.controller.release_action(action)

        threads: List[threading.Thread] = []
        for action, delay in action_plan:
            t = threading.Thread(target=hold_and_release, args=(action, delay))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()
    
    def perform_action_timeline(self, action_plan: list[dict]):
        """
        Execute overlapping timed actions.

        Each action dict should have:
        - 'action': ActionMapping
        - 'start': float (when to press, in seconds)
        - 'duration': float (how long to hold)
        """
        import threading
        import time

        def execute_timed(action: ActionMapping, start: float, duration: float):
            time.sleep(start)
            self.controller.execute_action(action)
            time.sleep(duration)
            self.controller.release_action(action)

        threads: List[threading.Thread] = []
        for entry in action_plan:
            action = entry["action"]
            start = entry.get("start", 0.0)
            duration = entry.get("duration", 0.2)

            t = threading.Thread(target=execute_timed, args=(action, start, duration))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()


    def generate(
        self,
        action_plan: Union[
            list[Union[ActionMapping, tuple[ActionMapping, float]]],
            list[dict]
        ],
        text_prompt: str,
        perform_duration: float,
        video_duration: float = None,
        record_video: bool = True,
    ):
        """
        Executes a single or multiple actions with optional video recording.

        Supports both simple list-based action plans and timeline-based overlapping action plans.

        Args:
            action_plan: A list of ActionMappings, (ActionMapping, duration) tuples, or timeline dicts.
            text_prompt: Text describing the action for labeling the video.
            perform_duration: Default duration for actions that don't specify one.
            video_duration: Duration of the recording. Defaults to perform_duration.
            record_video: Whether to record video.
        """

        is_timeline = (
            isinstance(action_plan, list)
            and len(action_plan) > 0
            and isinstance(action_plan[0], dict)
            and "action" in action_plan[0]
        )

        if is_timeline:
            # Validate timeline structure
            for entry in action_plan:
                if not isinstance(entry, dict) or "action" not in entry or "start" not in entry or "duration" not in entry:
                    raise ValueError(f"Invalid timeline entry: {entry}")

            actions = [entry["action"] for entry in action_plan]

        else:
            # Normalize simple action list into (ActionMapping, duration) tuples
            if isinstance(action_plan, ActionMapping):
                action_plan = [action_plan]

            normalized_actions = []
            for item in action_plan:
                if isinstance(item, ActionMapping):
                    normalized_actions.append((item, perform_duration))
                elif isinstance(item, tuple) and isinstance(item[0], ActionMapping):
                    normalized_actions.append(item)
                else:
                    raise ValueError(f"Invalid action entry: {item}")
            action_plan = normalized_actions
            actions = [a for a, _ in action_plan]

        if video_duration is None:
            video_duration = perform_duration

        if record_video:
            video_thread = threading.Thread(
                target=self.capture_video, args=(actions, text_prompt, video_duration)
            )
            video_thread.start()

        # Perform actions
        if is_timeline:
            self.perform_action_timeline(action_plan)
        else:
            self.perform_action(action_plan)

        if record_video:
            video_thread.join()


    def save_metadata(self):
        """Saves metadata about the dataset (actions and prompts) as a JSON file with actions as lists."""
        metadata_file = os.path.join(self.dataset_dir, "metadata.json")
        metadata = []

        # Load existing metadata if not overwriting
        if self.next_video_index > 0 and os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

        # Append new entries
        for i, action in enumerate(self.actions):
            metadata.append({
                "video_file": self.video_filenames[i],
                "text_prompt": self.text_prompts[i],
                "actions": action
            })

        # Save back to file
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

        print(f"Saved metadata to {metadata_file}")
