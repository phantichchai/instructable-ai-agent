from torch.utils.data import DataLoader
from data.dataset import CommonGameplayPromptActionDataset

root_dir = "dataset"
image_size = (160, 256)

dataset = CommonGameplayPromptActionDataset(root_dir=root_dir, image_size=image_size)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Example loop
for batch in dataloader:
    print(batch["frames"].shape)         # (B, T, C, H, W)
    print(batch["text_prompt"])          # List of prompts
    print(batch["action"].shape)         # (B, num_actions)