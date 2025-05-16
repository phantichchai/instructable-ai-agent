from data.generate_dataset import GenerateDataset
from tools.genshin.mapping import ActionMapping
from tools.genshin.controller import GenshinImpactController
import pyautogui
import os
import time
import random

attack_variantion = [
    "Assault", "Strike", "Charge", "Assail", "Engage", "Pounce", "Lunge", "Attack"
]

controller = GenshinImpactController()
dir = os.path.join("dataset", "attack")
dataset_generator = GenerateDataset(controller=controller, dataset_dir=dir, fps=8, overwrite=False)

pyautogui.keyDown('alt')
pyautogui.press('tab')
time.sleep(0.1)
pyautogui.keyUp('alt')

for switch in [(1, 2), (2, 3), (3, 4), (4, 1)]:
    attack_label = random.choices(attack_variantion)

    dataset_generator.generate([
        (ActionMapping.NORMAL_ATTACK, 0.5),
        (ActionMapping.NORMAL_ATTACK, 0.5), 
        (ActionMapping.NORMAL_ATTACK, 0.5),
        (ActionMapping.NORMAL_ATTACK, 0.5), 
        (ActionMapping.NORMAL_ATTACK, 0.5),], 
        attack_label, 1, 5, record_video=True)

    dataset_generator.generate(
        ActionMapping[f'SWITCH_CHARACTER_{switch[1]}'],
        f"Switch character from {switch[0]} to {switch[1]}",
        perform_duration=1,
        record_video=False
    )

dataset_generator.save_metadata()