from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from api.core import agent, device
import numpy as np
import torch
import cv2

generate_router = APIRouter()

@generate_router.post("/generate")
async def generate_sequence(
    text: str = Form(...),
    frame1: UploadFile = File(...),
    frame2: UploadFile = File(...), 
    frame3: UploadFile = File(...)
):
    try:
        frames = [frame1, frame2, frame3]
        tensor_frames = []

        for frame in frames:
            # Read the uploaded file
            contents = await frame.read()
            
            # Convert bytes to numpy array
            np_arr = np.frombuffer(contents, np.uint8)
            
            # Decode the numpy array as an image
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # Convert the image to RGB format (PyTorch expects images in RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize the frame
            resized_frame = cv2.resize(frame, (160, 256))

            # Normalize the frame (assuming model expects input in range [0, 1])
            normalized_frame = resized_frame / 255.0
            
            frame = np.array(normalized_frame)
            frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1)

            # Add the tensor to the list
            tensor_frames.append(frame)
        
        tensor_frames = torch.stack(tensor_frames)
        tensor_frames = tensor_frames.unsqueeze(dim=0).to(device)
        
        # Generate the action sequence
        actions = agent.get_actions(tensor_frames)

        actions = actions.tolist()
        return {"actions": actions, "instruction": text}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))