import numpy as np
import pyautogui
from tools.basic.mapping import INDEX_TO_ACTION, INDEX_TO_EVENT_TYPE, INDEX_TO_MOUSE_BUTTON

ORIGINAL_WIDTH = 1920
ORIGINAL_HEIGHT = 1080

def tensor_to_action(tensor):
    frames = np.array(tensor)
    for frame in frames:
        keyboard_actions = frame[:8]
        mouse_actions = frame[8:]

        for i in range(len(keyboard_actions)):
            if keyboard_actions[i]:
                button = INDEX_TO_ACTION.get(i)
                print(f"Press button: {button}")
                pyautogui.press(button)
        mouse_event = INDEX_TO_EVENT_TYPE.get(int(mouse_actions[0]))
        mouse_button = INDEX_TO_MOUSE_BUTTON.get(int(mouse_actions[1]))
        x_norm = mouse_actions[2]
        y_norm = mouse_actions[3]
        x_original = int(x_norm * ORIGINAL_WIDTH)
        y_original = int(y_norm * ORIGINAL_HEIGHT)

        if mouse_event == 'click':
            print(f"Mouse click button: {mouse_button}")
            pyautogui.click(mouse_button)

        if mouse_event == 'move':
            print(f"Mouse position: ({x_original}, {y_original})")
            pyautogui.moveRel(x_original, y_original)


held_keys = set()
held_mouse_buttons = set()

def control_from_action(action):
    global held_keys, held_mouse_buttons

    keyboard_actions = action[:8]
    mouse_actions = action[8:]

    new_held_keys = set()
    for i, pressed in enumerate(keyboard_actions):
        if pressed:
            key = INDEX_TO_ACTION.get(i)
            new_held_keys.add(key)
            if key not in held_keys:
                print(f"Press and hold key: {key}")
                pyautogui.keyDown(key)
    
    # Release keys no longer held
    for key in held_keys - new_held_keys:
        print(f"Release key: {key}")
        pyautogui.keyUp(key)

    held_keys = new_held_keys

    # Mouse handling
    mouse_event = INDEX_TO_EVENT_TYPE.get(int(mouse_actions[0]))
    mouse_button = INDEX_TO_MOUSE_BUTTON.get(int(mouse_actions[1]))
    x_norm = mouse_actions[2]
    y_norm = mouse_actions[3]
    x_original = int(x_norm * ORIGINAL_WIDTH)
    y_original = int(y_norm * ORIGINAL_HEIGHT)

    if mouse_event == 'click':
        if mouse_button not in held_mouse_buttons:
            print(f"Mouse click and hold: {mouse_button}")
            pyautogui.mouseDown(button=mouse_button)
            held_mouse_buttons.add(mouse_button)
    else:
        # Release any held mouse buttons if no click event
        for button in held_mouse_buttons:
            print(f"Release mouse button: {button}")
            pyautogui.mouseUp(button=button)
        held_mouse_buttons.clear()

    if mouse_event == 'move':
        print(f"Mouse position: ({x_original}, {y_original})")
        pyautogui.moveRel(x_original, y_original)