import pyautogui
import time
from tools.constants import KEY_BINDINGS, ACTION_MAPPING

class GenshinImpactController:
    def __init__(self, debug=False):
        # Dictionary to store the key bindings
        self.debug = debug

    # Movement functions
    def move(self, direction, duration=1):
        pyautogui.keyDown(KEY_BINDINGS[direction])
        time.sleep(duration)
        pyautogui.keyUp(KEY_BINDINGS[direction])

    def jump(self):
        pyautogui.press(KEY_BINDINGS['jump'])

    def sprint(self, duration=1):
        pyautogui.keyDown(KEY_BINDINGS['sprint'])
        time.sleep(duration)
        pyautogui.keyUp(KEY_BINDINGS['sprint'])

    # Combat functions
    def normal_attack(self):
        pyautogui.click(button=KEY_BINDINGS['normal_attack'])

    def elemental_skill(self):
        pyautogui.press(KEY_BINDINGS['elemental_skill'])

    def elemental_burst(self):
        pyautogui.press(KEY_BINDINGS['elemental_burst'])

    def aim_mode(self):
        pyautogui.press(KEY_BINDINGS['aim_mode'])

    # Interaction functions
    def interact(self):
        pyautogui.press(KEY_BINDINGS['interact'])

    # Character switching
    def switch_character(self, character_number):
        character_key = f'switch_character_{character_number}'
        if character_key in KEY_BINDINGS:
            pyautogui.press(KEY_BINDINGS[character_key])

    # Map and inventory functions
    def open_map(self):
        pyautogui.press(KEY_BINDINGS['open_map'])

    def open_inventory(self):
        pyautogui.press(KEY_BINDINGS['open_inventory'])

    # Method to execute action based on RL model's output
    def execute_action(self, action_value):
        action = ACTION_MAPPING.get(action_value)

        if self.debug:
            print(f"Action: {action}, Key binding: {KEY_BINDINGS[action]}")

        if action:
            if action.startswith("switch_character"):
                character_number = int(action[-1])  # Get the character number from the action name
                self.switch_character(character_number)
            elif action in ['move_forward', 'move_left', 'move_right', 'move_backward']:
                self.move(action)
            elif action == 'jump':
                self.jump()
            elif action == 'sprint':
                self.sprint()
            elif action == 'normal_attack':
                self.normal_attack()
            elif action == 'elemental_skill':
                self.elemental_skill()
            elif action == 'elemental_burst':
                self.elemental_burst()
            elif action == 'aim_mode':
                self.aim_mode()
            elif action == 'interact':
                self.interact()
            elif action == 'open_map':
                self.open_map()
            elif action == 'open_inventory':
                self.open_inventory()

# Example usage
if __name__ == '__main__':
    controller = GenshinImpactController()

    # Example: Execute an action based on an RL model output (e.g., action value 0)
    controller.execute_action(0)  # Moves forward
    controller.execute_action(4)  # Jumps
    time.sleep(1)
    for i in range(6):
        controller.execute_action(6)
        time.sleep(0.5)