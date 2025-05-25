import os
import pyautogui
import time
from data.generate_dataset import GenerateDataset
from tools.genshin.mapping import ActionMapping, ActionPromptMapping
from tools.genshin.controller import GenshinImpactController

controller = GenshinImpactController()
dir = os.path.join("dataset", "sprint")
dataset_generator = GenerateDataset(controller=controller, dataset_dir=dir, fps=8, overwrite=False)

pyautogui.keyDown('alt')
pyautogui.press('tab')
time.sleep(0.1)
pyautogui.keyUp('alt')

for switch in [2, 3, 4, 1]:
    dataset_generator.generate([
        {"action": ActionMapping.SPRINT, "start": 0, "duration": 0.5},
        {"action": ActionMapping.SPRINT, "start": 1, "duration": 0.5}],
        f"[ACTION] {ActionPromptMapping[ActionMapping.SPRINT]}", 
        perform_duration=2, 
        record_video=True)

    dataset_generator.generate(
        ActionMapping[f'SWITCH_CHARACTER_{switch}'], 
        f"[ACTION] {ActionPromptMapping[ActionMapping[f'SWITCH_CHARACTER_{switch}']]}", 
        perform_duration=0.5, 
        record_video=False)

dataset_generator.save_metadata()