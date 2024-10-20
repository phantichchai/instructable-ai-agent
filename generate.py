from data.generate_dataset import GenerateDataset
from tools.genshin_impact_controller import GenshinImpactController

if __name__ == '__main__':
    controller = GenshinImpactController()
    dataset_generator = GenerateDataset(controller)

    # Example of generating data for moving forward
    dataset_generator.generate("move_forward", "Move forward for 5 seconds", duration=5)
    dataset_generator.generate("move_left", "Move left for 5 seconds", duration=5)

    # Save metadata for the dataset
    dataset_generator.save_metadata()
