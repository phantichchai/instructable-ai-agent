import shutil
import os
from datetime import datetime
import torch
import numpy as np


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
    instructions, action_tensors, video_tensors = zip(*batch)
    
    # Find the maximum length for instructions, actions, and video tensors
    max_instructions_length = max(tensor.shape[0] for tensor in instructions)
    max_action_length = max(tensor.shape[0] for tensor in action_tensors)
    max_video_length = max(tensor.shape[0] for tensor in video_tensors)

    # Pad each tensor to the maximum length found above
    padded_instruction_tensors = np.array([pad_tensor(tensor, max_instructions_length) for tensor in instructions])
    padded_action_tensors = np.array([pad_tensor(tensor, max_action_length) for tensor in action_tensors])
    padded_video_tensors = np.array([pad_tensor(tensor, max_video_length) for tensor in video_tensors])

    # Return the padded tensors
    return torch.tensor(padded_instruction_tensors), torch.tensor(padded_action_tensors), torch.tensor(padded_video_tensors)
