import shutil
import os
import torch
import numpy as np
import cv2
from datetime import datetime
from typing import Optional, Tuple


def count_folders(directory):
    try:
        # List all entries in the directory
        entries = os.listdir(directory)
        
        # Filter out files, only keep directories
        folders = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]
        
        # Count the number of folders
        num_folders = len(folders)
        
        return num_folders
    except FileNotFoundError:
        return "The specified directory does not exist."
    except Exception as e:
        return f"An error occurred: {e}"

def copy_file(src, dst):
    try:
        # Ensure the source file exists
        if not os.path.isfile(src):
            return f"Source file '{src}' does not exist."
        
        # Ensure the destination directory exists
        dst_dir = os.path.dirname(dst)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        
        # Copy the file
        shutil.copy2(src, dst)  # copy2 preserves file metadata
        return f"File copied from '{src}' to '{dst}'."
    except Exception as e:
        return f"An error occurred: {e}"

def create_folder_by_datetime(base_dir):
    try:
        # Get the current date and time in a specific format
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Create the folder name
        folder_name = os.path.join(base_dir, current_datetime)
        
        # Create the folder
        os.makedirs(folder_name, exist_ok=True)
        
        return folder_name
    except Exception as e:
        return f"An error occurred: {e}"
    
def pad_tensor(tensor, target_num_frames, pad_value=0):
        """
        Pads the time dimension of a tensor to a specified number of frames.
    
        Parameters:
        tensor (torch.Tensor): The original tensor of shape (time, ...).
        target_num_frames (int): The desired number of frames after padding.
        pad_value (float, optional): The value to use for padding. Default is 0.
    
        Returns:
        torch.Tensor: The padded tensor of shape (target_num_frames, ...).
        """
        num_frames_to_pad = target_num_frames - tensor.shape[0]
    
        if num_frames_to_pad > 0:
            # Split the padding equally on both sides (or adjust as needed)
            pad_before = num_frames_to_pad // 2
            pad_after = num_frames_to_pad - pad_before
    
            # Create padding tensors with the specified pad_value
            padding_shape_before = (pad_before, *tensor.shape[1:])
            pad_tensor_before = torch.full(padding_shape_before, pad_value)
            padding_shape_after = (pad_after, *tensor.shape[1:])
            pad_tensor_after = torch.full(padding_shape_after, pad_value)
    
            # Concatenate the padding tensors with the original tensor
            padded_tensor = torch.cat((pad_tensor_before, tensor, pad_tensor_after), dim=0)
        else:
            padded_tensor = tensor
    
        return padded_tensor


def pad_tensor(tensor, target_num_frames, pad_value=0):
        """
        Pads the time dimension of a tensor to a specified number of frames.
    
        Parameters:
        tensor (torch.Tensor): The original tensor of shape (time, ...).
        target_num_frames (int): The desired number of frames after padding.
        pad_value (float, optional): The value to use for padding. Default is 0.
    
        Returns:
        torch.Tensor: The padded tensor of shape (target_num_frames, ...).
        """
        num_frames_to_pad = target_num_frames - tensor.shape[0]
    
        if num_frames_to_pad > 0:
            # Split the padding equally on both sides (or adjust as needed)
            pad_before = num_frames_to_pad // 2
            pad_after = num_frames_to_pad - pad_before
    
            # Create padding tensors with the specified pad_value
            padding_shape_before = (pad_before, *tensor.shape[1:])
            pad_tensor_before = torch.full(padding_shape_before, pad_value)
            padding_shape_after = (pad_after, *tensor.shape[1:])
            pad_tensor_after = torch.full(padding_shape_after, pad_value)
    
            # Concatenate the padding tensors with the original tensor
            padded_tensor = torch.cat((pad_tensor_before, tensor, pad_tensor_after), dim=0)
        else:
            padded_tensor = tensor
    
        return padded_tensor


def custom_collate_fn(batch):
    # Unzip the batch into instructions, action_tensors, and video_tensors
    instructions, video_tensors, action_tensors  = zip(*batch)
    
    # Find the maximum length for instructions, actions, and video tensors
    max_action_length = max(tensor.shape[0] for tensor in action_tensors)
    max_video_length = max(tensor.shape[0] for tensor in video_tensors)

    # Pad each tensor to the maximum length found above
    padded_action_tensors = np.array([pad_tensor(tensor, max_action_length) for tensor in action_tensors])
    padded_video_tensors = np.array([pad_tensor(tensor, max_video_length) for tensor in video_tensors])

    # Return the padded tensors
    return instructions, torch.tensor(padded_video_tensors), torch.tensor(padded_action_tensors)


def read_video_tensor(
    path: str,
    resize_to: Optional[Tuple[int, int]] = None,
    max_frames: Optional[int] = None
) -> torch.Tensor:
    """
    Reads a video file and returns a tensor of shape [T, C, H, W]

    Args:
        path (str): Path to the video file
        resize_to (tuple[int, int], optional): (width, height) to resize each frame
        max_frames (int, optional): Maximum number of frames to read

    Returns:
        torch.Tensor: Video tensor of shape [T, C, H, W], values in [0, 1]
    """
    cap = cv2.VideoCapture(path)
    frames = []
    count = 0

    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {path}")

    while True:
        ret, frame = cap.read()
        if not ret or (max_frames is not None and count >= max_frames):
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if resize_to is not None:
            frame = cv2.resize(frame, resize_to)

        frames.append(torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0)  # C x H x W
        count += 1

    cap.release()

    if not frames:
        raise ValueError("No frames were read from the video.")

    return torch.stack(frames)  # T x C x H x W


from datetime import datetime

def generate_experiment_name(
    model_name: str,
    experiment: str = "default",
    dataset: str = "default",
    prompt_style: str = "base",
    seed: int = 42,
    include_epoch: bool = False,
    epoch: int = None,
    add_timestamp: bool = True
) -> str:
    parts = [
        model_name,
        f"exp-{experiment}",
        f"data-{dataset}",
        f"prompt-{prompt_style}",
        f"seed-{seed}"
    ]
    if include_epoch and epoch is not None:
        parts.append(f"epoch-{epoch}")
    if add_timestamp:
        parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    return "_".join(parts)

