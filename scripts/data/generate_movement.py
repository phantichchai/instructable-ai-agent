import os
import pyautogui
import time
from data.generate_dataset import GenerateDataset
from tools.genshin.mapping import ActionMapping, ActionPromptMapping
from tools.genshin.controller import GenshinImpactController

controller = GenshinImpactController()
dir = os.path.join("dataset", "movement")
dataset_generator = GenerateDataset(controller=controller, dataset_dir=dir, fps=8, overwrite=False)

pyautogui.keyDown('alt')
pyautogui.press('tab')
time.sleep(0.1)
pyautogui.keyUp('alt')

# Example of generating data for moving forward
for switch in [2, 3, 4, 1]:
    dataset_generator.generate(ActionMapping.MOVE_FORWARD, f"[ACTION] {ActionPromptMapping[ActionMapping.MOVE_FORWARD]}", perform_duration=2)
    dataset_generator.generate(ActionMapping.MOVE_BACKWARD, f"[ACTION] {ActionPromptMapping[ActionMapping.MOVE_BACKWARD]}", perform_duration=2)
    dataset_generator.generate(ActionMapping.MOVE_LEFT, f"[ACTION] {ActionPromptMapping[ActionMapping.MOVE_LEFT]}", perform_duration=2)
    dataset_generator.generate(ActionMapping.MOVE_RIGHT, f"[ACTION] {ActionPromptMapping[ActionMapping.MOVE_RIGHT]}", perform_duration=2)
    dataset_generator.generate(ActionMapping[f'SWITCH_CHARACTER_{switch}'], f"[ACTION] {ActionPromptMapping[ActionMapping[f'SWITCH_CHARACTER_{switch}']]}", perform_duration=1, record_video=False)

# Save metadata for the dataset
dataset_generator.save_metadata()
