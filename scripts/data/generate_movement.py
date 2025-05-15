from data.generate_dataset import GenerateDataset
from tools.genshin.mapping import ActionMapping
from tools.genshin.controller import GenshinImpactController

if __name__ == '__main__':
    controller = GenshinImpactController()
    dataset_generator = GenerateDataset(controller=controller, dataset_dir="dataset/movement", fps=8)

    # Example of generating data for moving forward
    for switch in [(1, 2), (2, 3), (3, 4), (4, 1)]:
        dataset_generator.generate(ActionMapping.MOVE_FORWARD, "Move forward for 5 seconds", perform_duration=5)
        dataset_generator.generate(ActionMapping.MOVE_LEFT, "Move left for 5 seconds", perform_duration=5)
        dataset_generator.generate(ActionMapping.MOVE_BACKWARD, "Move backward for 5 seconds", perform_duration=5)
        dataset_generator.generate(ActionMapping.MOVE_RIGHT, "Move right for 5 seconds", perform_duration=5)
        dataset_generator.generate([ActionMapping.MOVE_FORWARD, ActionMapping.SPRINT], "Sprint forward for 5 seconds", perform_duration=5)
        dataset_generator.generate(ActionMapping[f'SWITCH_CHARACTER_{switch[1]}'], f"Switch character from {switch[0]} to {switch[1]}", perform_duration=1)

    # Save metadata for the dataset
    dataset_generator.save_metadata()
