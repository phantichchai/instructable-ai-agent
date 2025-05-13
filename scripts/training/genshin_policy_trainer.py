import torch
import os
from torch import nn
from torch.nn.functional import cosine_similarity
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data.dataset import CommonGameplayPromptActionDataset
from datetime import datetime
from model.genshin.policy import PolicyFromMineCLIP
from model.mineclip.mineclip import MineCLIP

saved_weights = []

def save_model(save_dir, model: nn.Module, epoch):
    """Saves the model and maintains only 5 most recent weights."""
    model_name = f"{model.__class__.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_epoch_{epoch}.pt"
    model_path = os.path.join(save_dir, model_name)
    torch.save(model.state_dict(), model_path)
    saved_weights.append(model_path)

    # Remove old weights if more than 5 saved
    if len(saved_weights) > 5:
        oldest_model = saved_weights.pop(0)
        if os.path.exists(oldest_model):
            os.remove(oldest_model)

def train_behavior_cloning_with_mineclip(
    model: nn.Module,
    mineclip: MineCLIP,
    dataloader,
    optimizer: Optimizer,
    criterion,
    save_dir: str,
    num_epochs: int,
    device="cuda"
):
    model.train()
    mineclip.eval()  # keep MineCLIP frozen
    writer = SummaryWriter(log_dir=f"runs/{model.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")


    for epoch in range(num_epochs):
        for batch_idx, data in enumerate(dataloader):
            video = data['frames']
            video = video.to(device)
            video_embed = mineclip.encode_video(video).to(device)
            text_prompt = data['text_prompt']
            text_embed = mineclip.encode_text(text_prompt).to(device)
            text_embed = text_embed / text_embed.norm(dim=1, keepdim=True)

            obs_embed = torch.cat([video_embed, text_embed], dim=-1)
            # Predict policy output (action embedding)
            pred_embed = model(obs_embed)

            action = data["action"].to(device)
            loss = criterion(pred_embed, action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}], Loss: {loss.item():.4f}')
            writer.add_scalar('training_loss', loss, epoch * len(dataloader) + batch_idx)
        
        if epoch % 250 == 0:
            save_model(save_dir, model, epoch)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set up parameters
    root_dir = './dataset'  # Path to your dataset
    image_size = (160, 256)  # Image size for the input
    batch_size = 8
    num_epochs = 1000
    learning_rate = 1e-3
    save_dir = './saved_models'  # Directory to save models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Dataset
    dataset = CommonGameplayPromptActionDataset(root_dir=root_dir, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    cfg = {'arch': 'vit_base_p16_fz.v2.t2', 'hidden_dim': 512, 'image_feature_dim': 512, 'mlp_adapter_spec': 'v0-2.t0', 'pool_type': 'attn.d2.nh8.glusw', 'resolution': [160, 256]}

    mineclip = MineCLIP(**cfg).to(device)
    mineclip.load_state_dict(torch.load("saved_models/mineclip/attn_new.pth"))


    policy = PolicyFromMineCLIP(
        mineclip=mineclip,
        input_dim=1024,
        embed_dim=512,
        num_actions=22
    ).to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    train_behavior_cloning_with_mineclip(
        policy,
        mineclip,
        dataloader,
        optimizer,
        criterion,
        save_dir=save_dir,
        num_epochs=num_epochs
    )