import cv2
import numpy as np
import time
import threading
import mss
from api import generate
from tools.controller import tensor_to_action
from tools.window import get_window_coordinates

def start_capturing(instruction: str, window_title: str, max_frames: int, fps: int):
    # Run the capturing in a separate thread to avoid blocking the main Tkinter thread
    capturing = threading.Thread(target=capture_from_window, args=(instruction, window_title, max_frames, fps))
    capturing.start()


def capture_from_window(instruction: str, window_title: str, max_frames: int, fps: int):
    window_coords = get_window_coordinates(window_title)
    if not window_coords:
        raise Exception(f'Window with title "{window_title}" not found')

    frames = []
    frame_count = 0
    interval = 1 / fps

    with mss.mss() as sct:
        while frame_count < max_frames:
            start_time = time.time()
            img = np.array(sct.grab(window_coords))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            frames.append(frame)
            frame_count += 1
            elapsed_time = time.time() - start_time
            time.sleep(max(0, interval - elapsed_time))
            
            if frame_count % 3 == 0:
                response = generate.send_frames_and_text_to_api(frames, instruction)
                frames = []
                tensor_to_action(response['actions'])

            # cv2.imshow("Captured Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    cv2.destroyAllWindows()