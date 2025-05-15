from data.generate_dataset import GenerateDataset
from tools.genshin.mapping import ActionMapping
from tools.genshin.controller import GenshinImpactController
import pyautogui
import os
import time

controller = GenshinImpactController()
dir = os.path.join("dataset", "attack")
dataset_generator = GenerateDataset(controller=controller, dataset_dir=dir, fps=8, overwrite=True)

pyautogui.keyDown('alt')
pyautogui.press('tab')
time.sleep(0.1)
pyautogui.keyUp('alt')

for switch in [(1, 2), (2, 3), (3, 4), (4, 1)]:
    dataset_generator.generate([
        (ActionMapping.NORMAL_ATTACK, 0.5),
        (ActionMapping.NORMAL_ATTACK, 0.5), 
        (ActionMapping.NORMAL_ATTACK, 0.5),
        (ActionMapping.NORMAL_ATTACK, 0.5), 
        (ActionMapping.NORMAL_ATTACK, 0.5),], 
        "Attack", 1, 5, record_video=True)

    dataset_generator.generate(
        ActionMapping[f'SWITCH_CHARACTER_{switch[1]}'],
        f"Switch character from {switch[0]} to {switch[1]}",
        perform_duration=1,
        record_video=False
    )

dataset_generator.save_metadata()