import pyautogui
import time

class GenshinImpactController:
    def __init__(self, model):
        # Dictionary to store the key bindings
        self.model = model
        self.key_bindings = {
            "move_forward": 'w',
            "move_left": 'a',
            "move_right": 'd',
            "move_backward": 's',
            "jump": 'space',
            "sprint": 'shift',
            "normal_attack": 'left',  # Left mouse button for normal attack
            "elemental_skill": 'e',
            "elemental_burst": 'q',
            "aim_mode": 'r',
            "interact": 'f',
            "open_map": 'm',
            "open_inventory": 'b',
            "switch_character_1": '1',
            "switch_character_2": '2',
            "switch_character_3": '3',
            "switch_character_4": '4'
        }

        # Define the mapping from action indices to key bindings
        self.action_mapping = {
            0: "move_forward",
            1: "move_left",
            2: "move_right",
            3: "move_backward",
            4: "jump",
            5: "sprint",
            6: "normal_attack",
            7: "elemental_skill",
            8: "elemental_burst",
            9: "aim_mode",
            10: "interact",
            11: "open_map",
            12: "open_inventory",
            13: "switch_character_1",
            14: "switch_character_2",
            15: "switch_character_3",
            16: "switch_character_4"
        }

    # Movement functions
    def move(self, direction, duration=1):
        pyautogui.keyDown(self.key_bindings[direction])
        time.sleep(duration)
        pyautogui.keyUp(self.key_bindings[direction])

    def jump(self):
        pyautogui.press(self.key_bindings['jump'])

    def sprint(self, duration=1):
        pyautogui.keyDown(self.key_bindings['sprint'])
        time.sleep(duration)
        pyautogui.keyUp(self.key_bindings['sprint'])

    # Combat functions
    def normal_attack(self):
        pyautogui.click(button=self.key_bindings['normal_attack'])

    def elemental_skill(self):
        pyautogui.press(self.key_bindings['elemental_skill'])

    def elemental_burst(self):
        pyautogui.press(self.key_bindings['elemental_burst'])

    def aim_mode(self):
        pyautogui.press(self.key_bindings['aim_mode'])

    # Interaction functions
    def interact(self):
        pyautogui.press(self.key_bindings['interact'])

    # Character switching
    def switch_character(self, character_number):
        character_key = f'switch_character_{character_number}'
        if character_key in self.key_bindings:
            pyautogui.press(self.key_bindings[character_key])

    # Map and inventory functions
    def open_map(self):
        pyautogui.press(self.key_bindings['open_map'])

    def open_inventory(self):
        pyautogui.press(self.key_bindings['open_inventory'])

    # Method to execute action based on RL model's output
    def execute_action(self, action_value):
        action = self.action_mapping.get(action_value)
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