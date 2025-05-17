import pyautogui
import time
from tools.genshin.mapping import KeyBinding, ActionMapping, HOLDABLE_ACTIONS


class GenshinImpactController:
    def __init__(self, debug=False):
        self.debug = debug
        self.held_keys = set()
        self.held_mouse_buttons = set()

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

    def release_action(self, action_value: ActionMapping):
        pyautogui.keyUp(KeyBinding[action_value.name].value)

    # Main executor
    def execute_action(self, action_value: ActionMapping):
        action = ActionMapping(action_value)

        if self.debug:
            print(f"Action: {action.name}, Key binding: {KeyBinding[action.name].value}")

        if action:
            if action.name.startswith("SWITCH_CHARACTER"):
                character_number = int(action.name[-1])
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

    def control_from_action(self, action):
        keyboard_actions = action
        new_held_keys = set()

        for i, pressed in enumerate(keyboard_actions):
            if pressed:
                action_enum = ActionMapping(i)
                if action_enum in HOLDABLE_ACTIONS:
                    new_held_keys.add(action_enum)
                    if action_enum not in self.held_keys:
                        self.execute_action(action_enum)  # Hold new direction
                else:
                    self.execute_action(action_enum)  # Instant press (jump, skill, etc.)

        # Release only movement keys that are no longer pressed
        for action_enum in self.held_keys - new_held_keys:
            self.release_action(KeyBinding[action_enum.name].value)

        self.held_keys = new_held_keys

    def emergency_stop(self):
        for action_enum in self.held_keys:
            self.release_action(KeyBinding[action_enum.name].value)


# Example usage
if __name__ == '__main__':
    controller = GenshinImpactController()

    # Example: Execute an action based on an RL model output (e.g., action value 0)
    controller.execute_action(0)  # Moves forward
    controller.execute_action(1)  # Moves left
    controller.execute_action(4)  # Jump
    time.sleep(5)
    controller.release_action(action_value=ActionMapping.MOVE_FORWARD)
    controller.release_action(action_value=ActionMapping.MOVE_LEFT)
    
    # for i in range(6):
    #     controller.execute_action(6)
    #     time.sleep(0.5)
