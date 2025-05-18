from data.generate_dataset import GenerateDataset
from tools.genshin.mapping import ActionMapping, ActionPromptMapping
from tools.genshin.controller import GenshinImpactController
import pyautogui
import os
import time
import random

controller = GenshinImpactController()
dir = os.path.join("dataset", "attack")
dataset_generator = GenerateDataset(controller=controller, dataset_dir=dir, fps=8, overwrite=False)

pyautogui.keyDown('alt')
pyautogui.press('tab')
time.sleep(0.1)
pyautogui.keyUp('alt')

for switch in [2, 3, 4, 1]:
    dataset_generator.generate(
        [
            (ActionMapping.NORMAL_ATTACK, 0.2),
            (ActionMapping.NORMAL_ATTACK, 0.2), 
            (ActionMapping.NORMAL_ATTACK, 0.2),
            (ActionMapping.NORMAL_ATTACK, 0.2), 
            (ActionMapping.NORMAL_ATTACK, 0.2),
        ], 
        f"[ACTION] {ActionPromptMapping[ActionMapping.NORMAL_ATTACK]}",
        perform_duration=2,
        record_video=True)

    dataset_generator.generate(
        ActionMapping[f'SWITCH_CHARACTER_{switch}'], 
        f"[ACTION] {ActionPromptMapping[ActionMapping[f'SWITCH_CHARACTER_{switch}']]}", 
        perform_duration=1, 
        record_video=False)

dataset_generator.save_metadata()