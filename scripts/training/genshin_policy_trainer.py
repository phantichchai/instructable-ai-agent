import torch
from torch import nn
from torch.nn.functional import cosine_similarity
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from data.dataset import GenshinPolicyDataset
from model.genshin.policy import PolicyFromMineCLIP
from model.mineclip.mineclip import MineCLIP
from tools.utils import read_video_tensor

def train_behavior_cloning_with_mineclip(
    model: nn.Module,
    mineclip: MineCLIP,
    dataloader,
    optimizer: Optimizer,
    device="cuda"
):
    model.train()
    mineclip.eval()  # keep MineCLIP frozen

    for data in dataloader:
        video = data['video_tensor']
        text_prompt = data['text_prompt']
        video = video.to(device)
        text_token = mineclip.encode_text(text_prompt).to(device)  # e.g., CLIP tokenizer

        # Get target text embedding from MineCLIP
        with torch.no_grad():
            text_embed = mineclip.encode_text(text_token)
            text_embed = text_embed / text_embed.norm(dim=1, keepdim=True)

        # Predict policy output (action embedding)
        pred_embed = model(video)
        pred_embed = pred_embed / pred_embed.norm(dim=1, keepdim=True)

        # Cosine similarity loss (maximize alignment)
        similarity = cosine_similarity(pred_embed, text_embed, dim=1)
        loss = 1 - similarity.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item():.4f}, Similarity: {similarity.mean().item():.4f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set up parameters
    root_dir = './dataset'  # Path to your dataset
    image_size = (224, 224)  # Image size for the input
    batch_size = 8
    num_epochs = 1
    learning_rate = 1e-3
    save_dir = './saved_models'  # Directory to save models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Dataset
    dataset = GenshinPolicyDataset(root_dir=root_dir, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    cfg = {'arch': 'vit_base_p16_fz.v2.t2', 'hidden_dim': 512, 'image_feature_dim': 512, 'mlp_adapter_spec': 'v0-2.t0', 'pool_type': 'attn.d2.nh8.glusw', 'resolution': [160, 256]}

    mineclip = MineCLIP(**cfg).to(device)
    mineclip.load_state_dict(torch.load("saved_models/mineclip/attn_new.pth"))


    model = PolicyFromMineCLIP(
        mineclip=mineclip,
        embed_dim=512
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_behavior_cloning_with_mineclip(
        model,
        mineclip,
        dataloader,
        optimizer
    )