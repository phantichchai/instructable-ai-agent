import shutil
import os

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

from datetime import datetime

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