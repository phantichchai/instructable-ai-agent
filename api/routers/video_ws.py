from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from collections import deque
from datetime import datetime
from api.generate import process_frames
import numpy as np
import cv2

video_ws_router = APIRouter()
frame_buffer = deque(maxlen=16)

@video_ws_router.websocket("/ws/video")
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