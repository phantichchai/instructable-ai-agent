import cv2
import numpy as np
import time
import threading
import mss
from tools.basic.controller import tensor_to_action
from tools.genshin.controller import GenshinImpactController
from tools.window import get_window_coordinates
from collections import deque
from datetime import datetime
import websockets
import asyncio
from queue import Queue
import pyautogui
from clients.generate import send_frames_and_text_to_api


frame_buffer = deque(maxlen=16)

def start_capturing(instruction: str, window_title: str, max_frames: int, fps: int):
    def runner():
        asyncio.run(capture_from_window_ws(instruction, window_title, max_frames, fps))
    # Run the capturing in a separate thread to avoid blocking the main Tkinter thread
    pyautogui.keyDown('alt')
    pyautogui.press('tab')
    time.sleep(0.1)
    pyautogui.keyUp('alt')
    capturing = threading.Thread(target=runner)
    capturing.start()


def capture_from_window(instruction: str, window_title: str, max_frames: int, fps: int):
    window_coords = get_window_coordinates(window_title)
    if not window_coords:
        raise Exception(f'Window with title "{window_title}" not found')

    frame_buffer = deque(maxlen=max_frames)
    interval = 1 / fps

    with mss.mss() as sct:
        while True:
            start_time = time.time()
            img = np.array(sct.grab(window_coords))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            frame = cv2.resize(frame, (160, 256))
            frame_buffer.append(frame)

            if len(frame_buffer) == max_frames:
                response = send_frames_and_text_to_api(list(frame_buffer), instruction)
                tensor_to_action(response['actions'])

            elapsed_time = time.time() - start_time
            time.sleep(max(0, interval - elapsed_time))

            cv2.imshow("Captured Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

# --------------------------------------WebSocket----------------------------------------------

WS_URL = "ws://192.168.1.239:8000/ws/video"
action_queue = Queue()
controller = GenshinImpactController(debug=True)

async def receive_actions(websocket):
    while True:
        try:
            action_bytes = await websocket.recv()
            action = list(np.frombuffer(action_bytes, dtype=np.uint8))
            action_queue.put(action)
        except Exception as e:
            print(f"[Error] Receiving action failed: {e}")
            break

async def capture_from_window_ws(instruction: str, window_title: str, max_frames: int, fps: int):
    window_coords = get_window_coordinates(window_title)
    if not window_coords:
        raise Exception(f'Window with title "{window_title}" not found')

    interval = 1 / fps
    instruction_bytes = instruction.encode('utf-8')
    instruction_len = len(instruction_bytes).to_bytes(4, byteorder='big')
    
    with mss.mss() as sct:
        async with websockets.connect(WS_URL) as websocket:
            
            asyncio.create_task(receive_actions(websocket))

            while True:
                start_time = time.time()

                # Capture frame
                img = np.array(sct.grab(window_coords))
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                frame = cv2.resize(frame, (160, 256))

                # Encode frame to JPEG bytes
                success, frame_buf = cv2.imencode(".jpg", frame)
                if success:
                    message = instruction_len + instruction_bytes + frame_buf.tobytes()
                    await websocket.send(message)
                
                while not action_queue.empty():
                    action = action_queue.get()
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Action received: {action}")
                    controller.control_from_action(action)

                # Maintain FPS
                elapsed_time = time.time() - start_time
                time.sleep(max(0, interval - elapsed_time))

                # Optional: display
                cv2.imshow("Captured Frame", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    with action_queue.mutex:
                        action_queue.queue.clear()
                        action_queue.unfinished_tasks = 0
                    break

    cv2.destroyAllWindows()