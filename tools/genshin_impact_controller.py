import pyautogui
import time
from tools.action_key_mapping import KeyBinding, ActionMapping

class GenshinImpactController:
    def __init__(self, debug=False):
        # Debug mode to print actions
        self.debug = debug

    # Movement functions
    def move(self, direction: KeyBinding):
        pyautogui.keyDown(direction.value)

    def jump(self):
        pyautogui.press(KeyBinding.JUMP.value)

    def sprint(self):
        pyautogui.keyDown(KeyBinding.SPRINT.value)

    # Combat functions
    def normal_attack(self):
        pyautogui.click(button=KeyBinding.NORMAL_ATTACK.value)

    def elemental_skill(self):
        pyautogui.press(KeyBinding.ELEMENTAL_SKILL.value)

    def elemental_burst(self):
        pyautogui.press(KeyBinding.ELEMENTAL_BURST.value)

    def aim_mode(self):
        pyautogui.press(KeyBinding.AIM_MODE.value)

    # Interaction functions
    def interact(self):
        pyautogui.press(KeyBinding.INTERACT.value)

    # Character switching
    def switch_character(self, character_number):
        character_key = KeyBinding[f'SWITCH_CHARACTER_{character_number}']
        if character_key:
            pyautogui.press(character_key.value)

    # Map and inventory functions
    def open_map(self):
        pyautogui.press(KeyBinding.OPEN_MAP.value)

    def open_inventory(self):
        pyautogui.press(KeyBinding.OPEN_INVENTORY.value)

    # Method to execute action based on RL model's output
    def execute_action(self, action_value: ActionMapping):
        action = ActionMapping(action_value)

        if self.debug:
            print(f"Action: {action.name}, Key binding: {KeyBinding[action.name].value}")

        if action:
            if action.name.startswith("SWITCH_CHARACTER"):
                character_number = int(action.name[-1])  # Get the character number from the action name
                self.switch_character(character_number)
            elif action in [ActionMapping.MOVE_FORWARD, ActionMapping.MOVE_LEFT, ActionMapping.MOVE_RIGHT, ActionMapping.MOVE_BACKWARD]:
                self.move(KeyBinding[action.name])
            elif action == ActionMapping.JUMP:
                self.jump()
            elif action == ActionMapping.SPRINT:
                self.sprint()
            elif action == ActionMapping.NORMAL_ATTACK:
                self.normal_attack()
            elif action == ActionMapping.ELEMENTAL_SKILL:
                self.elemental_skill()
            elif action == ActionMapping.ELEMENTAL_BURST:
                self.elemental_burst()
            elif action == ActionMapping.AIM_MODE:
                self.aim_mode()
            elif action == ActionMapping.INTERACT:
                self.interact()
            elif action == ActionMapping.OPEN_MAP:
                self.open_map()
            elif action == ActionMapping.OPEN_INVENTORY:
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
