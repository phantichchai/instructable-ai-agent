import torch
from torch.utils.data import Dataset
import os
import cv2
import json
import numpy as np

ACTION_MAP = {
    'w': 0,
    'a': 1,
    's': 2,
    'd': 3,
    'Key.space': 4,
    'Key.shift': 5,
    'q': 6,
    'e': 7
}

EVENT_TYPE = {
    'idle': 0,
    'click': 1,
    'release': 2,
    'move': 3
}

MOUSE_BUTTON = {
    'idle': 0,
    'Button.left': 1,
    'Button.right': 2
}

class GameplayActionPairVideoDataset(Dataset):
    def __init__(self, root_dir, tokenizer, transform=None):
        """
        Args:
            root_dir (string): Root directory containing the subdirectories with JSON and MP4 files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data = self._load_data()
        self.tokenizer = tokenizer

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

    def frame_logs_to_actions_tensor(self, actions):
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
                action_tensor[9] = mouse_events[1]
                action_tensor[10] = mouse_events[2]
                action_tensor[11] = MOUSE_BUTTON.get(mouse_events[3], 0)
                
            actions_tensor.append(action_tensor)
        actions_tensor = np.array(actions_tensor)
        actions_tensor = torch.tensor(actions_tensor)
        return actions_tensor

    def preprocess_frame(self, frame, target_size=(64, 64)):
        # Resize the frame
        resized_frame = cv2.resize(frame, target_size)
        
        # Normalize the frame (assuming model expects input in range [0, 1])
        normalized_frame = resized_frame / 255.0
    
        return normalized_frame
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_info = self.data[idx]
        video_path = video_info['video_path']

        # Read the video using OpenCV
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = self.preprocess_frame(frame)
            frames.append(processed_frame)
        cap.release()

        # Convert list of frame into NumPy array
        frames = np.array(frames)

        # Convert to tensor and permute dimensions to T, C, H, W
        frames = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2)
        instruction = self.tokenizer(text=self.data[idx]['instruction'],
                                     return_tensors='pt',
                                     padding=True,
                                     truncation=True,
                                     max_length=128
                                    )['input_ids'].squeeze(dim=0)
        actions = self.frame_logs_to_actions_tensor(self.data[idx]['actions'])

        if self.transform:
            frames = self.transform(frames)

        return instruction, frames, actions