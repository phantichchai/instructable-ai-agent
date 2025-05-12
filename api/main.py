from fastapi import FastAPI, HTTPException, File, UploadFile, Form, WebSocket, WebSocketDisconnect
from model.genshin.policy import PolicyFromMineCLIP
from model.mineclip.mineclip import MineCLIP
from collections import deque
from datetime import datetime
import numpy as np
import torch
import cv2

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = {'arch': 'vit_base_p16_fz.v2.t2', 'hidden_dim': 512, 'image_feature_dim': 512, 'mlp_adapter_spec': 'v0-2.t0', 'pool_type': 'attn.d2.nh8.glusw', 'resolution': [160, 256]}
mineclip = MineCLIP(**cfg).to(device)
mineclip.load_state_dict(torch.load("saved_models/mineclip/attn_new.pth"))

agent = PolicyFromMineCLIP(mineclip, 1024, 512, 22)
agent.load_state_dict(torch.load('saved_models/PolicyFromMineCLIP_20250512_191107_epoch_500.pt'))
agent.to(device)

frame_buffer = deque(maxlen=16)


@app.post("/generate")
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


@app.websocket("/ws/video")
async def video_stream(websocket: WebSocket):
    await websocket.accept()
    print("Client connected for video streaming")

    try:
        while True:
            # Receive a frame (raw binary data)
            data = await websocket.receive_bytes()

            # 1. First 4 bytes = instruction length
            instruction_len = int.from_bytes(data[:4], byteorder='big')

            # 2. Next = instruction
            instruction = data[4:4 + instruction_len].decode('utf-8')

            # 3. Remaining = image bytes
            frame_data = data[4 + instruction_len:]

            # Convert received bytes to numpy array (frame)
            np_frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Frame Buffer length: {len(frame_buffer)}")
            if frame is not None:
                frame_buffer.append(frame)
                
                if len(frame_buffer) == 16:
                    # Process once we have 16 frames
                    action = await process_frames(list(frame_buffer), instruction)

                    # Convert action to bytes
                    action_bytes = np.array(action, dtype=np.uint8).tobytes()
                    await websocket.send_bytes(action_bytes)
            else:
                print("Received empty frame")

    except WebSocketDisconnect:
        print("Client disconnected")


async def process_frames(frames, instruction):
    # List of 16 frames in BGR format
    frame_tensors = [torch.from_numpy(f).permute(2, 0, 1).float() / 255.0 for f in frames]

    # Stack into a batch of shape [16, C, H, W]
    video_tensor = torch.stack(frame_tensors).unsqueeze(0).to(device) # Shape: [16, 3, H, W]

    # Forward through your MineCLIP model
    video_embed = mineclip.encode_video(video_tensor)
    text_embed = mineclip.encode_text(instruction)
    obs_embed = torch.cat([video_embed, text_embed], dim=-1)

    # Predict action
    actions = agent.get_actions(obs_embed)
    return actions.cpu().numpy()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
