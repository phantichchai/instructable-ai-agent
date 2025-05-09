import torch
from torch.utils.data import Dataset
import os
import cv2
import json
import numpy as np
from tools.action_key_mapping import ActionMapping
from tools.mapping import ACTION_MAP, EVENT_TYPE, MOUSE_BUTTON

class GameplayActionPairVideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, image_size=None):
        """
        Args:
            root_dir (string): Root directory containing the subdirectories with JSON and MP4 files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data = self._load_data()
        self.image_size = image_size

    def _load_data(self):
        data = []
        for subdir in os.listdir(self.root_dir):
            subdir_path = os.path.join(self.root_dir, subdir)
            if os.path.isdir(subdir_path):
                frame_logs = os.path.join(subdir_path, 'frame_logs.json')
                gameplay = os.path.join(subdir_path, 'gameplay.mp4')
                if os.path.isfile(frame_logs) and os.path.isfile(gameplay):
                    with open(frame_logs, 'r') as file:
                        annotation = json.load(file)
                        annotation['video_path'] = gameplay
                        data.append(annotation)
        return data

    def frame_logs_to_actions_tensor(self, actions, max_width, max_height):
        actions_tensor = []
        num_actions = len(ACTION_MAP) + 4
        tensor_size = (num_actions,)
        
        for action in actions:
            key_events = [key_event['key'] for key_event in action['key_events']]
            mouse_events = []
            action_tensor = torch.zeros(tensor_size)
    
            for key in key_events:
                action_index = ACTION_MAP.get(key, -1)
                if action_index != -1:
                    action_tensor[action_index] = 1
            
            for mouse_event in action['mouse_events']:
                mouse_events = [mouse_event['event_type'], mouse_event['position'][0], mouse_event['position'][1], mouse_event.get('button', 'idle')]
                action_tensor[8] = EVENT_TYPE.get(mouse_events[0], 0)
                action_tensor[9] = MOUSE_BUTTON.get(mouse_events[3], 0)
                action_tensor[10] = mouse_events[1] / max_width
                action_tensor[11] = mouse_events[2] / max_height
                
            actions_tensor.append(action_tensor)
        actions_tensor = np.array(actions_tensor)
        actions_tensor = torch.tensor(actions_tensor)
        return actions_tensor

    def preprocess_frame(self, frame, target_size):
        # Resize the frame
        resized_frame = cv2.resize(frame, target_size)
        
        # Normalize the frame (assuming model expects input in range [0, 1])
        normalized_frame = resized_frame / 255.0
    
        return normalized_frame
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> dict[str, str | torch.Tensor]:
        video_info = self.data[idx]
        video_path = video_info['video_path']

        # Read the video using OpenCV
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = self.preprocess_frame(frame, self.image_size)
            frames.append(processed_frame)
        cap.release()

        # Convert list of frame into NumPy array
        frames = np.array(frames)

        # Convert to tensor and permute dimensions to T, C, H, W
        frames = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2)
        instruction = self.data[idx]['instruction']
        actions = self.frame_logs_to_actions_tensor(self.data[idx]['actions'], frame_width, frame_height)

        if self.transform:
            frames = self.transform(frames)

        return instruction, frames, actions


class CommonGameplayPromptActionDataset(Dataset):
    def __init__(self, root_dir, image_size, transform=None) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.image_size = image_size
        self.transform = transform
        self.metadata_file = os.path.join(self.root_dir, "metadata.json")
        self.action_size = len(ActionMapping)
        self.data = []

        # Load metadata from CSV
        self.load_metadata()
    
    def load_metadata(self):
        """Load metadata from a JSON file into a list of dictionaries."""
        if os.path.exists(self.metadata_file):
            # Open and read the JSON file
            with open(self.metadata_file, mode='r', encoding='utf-8') as file:
                self.data = json.load(file)  # Load JSON data into the data list

            # Print loaded metadata for confirmation
            print("Loaded metadata:")
            for record in self.data:
                print(record)
        else:
            print(f"Metadata file '{self.metadata_file}' not found!")

        def __len__(self):
            """Returns the total number of samples."""
            return len(self.data)
    
    def actions_to_one_hot(self, actions):
        """Convert action indices to a one-hot encoded tensor."""
        one_hot = torch.zeros(self.action_size, dtype=torch.float32)
        for action in actions:
            one_hot[action] = 1.0  # Set the corresponding index to 1
        return one_hot

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        record = self.data[index]
        video_path = os.path.join(self.root_dir, record['video_file'])
        text_prompt = record['text_prompt']
        actions = record['actions']

        # Read video using OpenCV
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, self.image_size)
            normalize_frame = resized_frame / 255.0
            frames.append(normalize_frame)
        cap.release()

        # Convert list of frame into NumPy array
        frames = np.array(frames)
        # Convert to tensor and permute dimensions to T, C, H, W
        frames = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2)
        
        one_hot_actions = self.actions_to_one_hot(actions=actions)

        sample = {
            "frames": frames,
            "text_prompt": text_prompt,
            "action": one_hot_actions
        }

        return sample
