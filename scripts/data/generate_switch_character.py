from data.generate_dataset import GenerateDataset
from tools.genshin.mapping import ActionMapping, ActionPromptMapping
from tools.genshin.controller import GenshinImpactController
import os
import pyautogui
import time

controller = GenshinImpactController()
dir = os.path.join("dataset", "switch_character")
dataset_generator = GenerateDataset(controller=controller, dataset_dir=dir, fps=8, overwrite=False)

pyautogui.keyDown('alt')
pyautogui.press('tab')
time.sleep(0.1)
pyautogui.keyUp('alt')

for _ in range(4):
    for switch in [2, 3, 4, 1]:
        dataset_generator.generate(
            ActionMapping[f'SWITCH_CHARACTER_{switch}'], 
            f"[ACTION] {ActionPromptMapping[ActionMapping[f'SWITCH_CHARACTER_{switch}']]}", 
            perform_duration=1
        )


dataset_generator.save_metadata()

