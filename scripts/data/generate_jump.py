from data.generate_dataset import GenerateDataset
from tools.genshin.mapping import ActionMapping
from tools.genshin.controller import GenshinImpactController
import os
import pyautogui
import time
import random

jump_variations = [
    "Jump", "Leap", "Hop", "Bounce", "Spring",
    "Vault", "Bound", "Skip", "Jump Up", "Take Off", "Leap Upwards"
]

controller = GenshinImpactController()
dir = os.path.join("dataset", "jump")
dataset_generator = GenerateDataset(controller=controller, dataset_dir=dir, fps=8, overwrite=False)

pyautogui.keyDown('alt')
pyautogui.press('tab')
time.sleep(0.1)
pyautogui.keyUp('alt')

for switch in [(1, 2), (2, 3), (3, 4), (4, 1)]:
    jump_label = random.choice(jump_variations)
    dataset_generator.generate(ActionMapping.JUMP, jump_label, 1)

    dataset_generator.generate(
        ActionMapping[f'SWITCH_CHARACTER_{switch[1]}'],
        f"Switch character from {switch[0]} to {switch[1]}",
        perform_duration=1,
        record_video=False
    )

dataset_generator.save_metadata()

