import torch
import os
from typing import Optional
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data.dataset import CommonGameplayPromptActionDataset
from datetime import datetime
from model.genshin.policy import PolicyFromMineCLIP
from model.mineclip.mineclip import MineCLIP

saved_weights = []

def save_model(save_dir, model: nn.Module, optimizer: Optimizer, epoch, loss=None):
    """Saves the training checkpoint and maintains only 5 most recent weights."""
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }

    model_name = f"{model.__class__.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_epoch_{epoch}.pt"
    model_path = os.path.join(save_dir, model_name)
    torch.save(checkpoint, model_path)
    saved_weights.append(model_path)

    # Keep only the 5 most recent weights
    if len(saved_weights) > 5:
        oldest = saved_weights.pop(0)
        if os.path.exists(oldest):
            os.remove(oldest)

def load_checkpoint(checkpoint_path, model: nn.Module, optimizer: Optimizer=None, device='cuda'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', None)
    return epoch, loss

def train_behavior_cloning_with_mineclip(
    model: nn.Module,
    mineclip: MineCLIP,
    dataloader: DataLoader,
    optimizer: Optimizer,
    scheduler: Optional[StepLR],
    criterion: nn.Module,
    save_dir: str,
    num_epochs: int,
    device: str = "cuda",
    resume_checkpoint: Optional[str] = None
) -> None:
    model.to(device)
    model.train()
    mineclip.to(device)
    mineclip.eval()

    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=f"runs/{model.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

    start_epoch = 0
    if resume_checkpoint:
        start_epoch, _ = load_checkpoint(resume_checkpoint, model, optimizer, device)
        print(f"Resumed training from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        for batch_idx, data in enumerate(dataloader):
            video = data['frames'].to(device)
            text_prompt = data['text_prompt']
            action = data["action"].to(device)

            with torch.no_grad():
                video_embed = mineclip.encode_video(video).to(device)
                text_embed = mineclip.encode_text(text_prompt).to(device)
                text_embed = text_embed / text_embed.norm(dim=1, keepdim=True)

            obs_embed = torch.cat([video_embed, text_embed], dim=-1)
            pred_embed = model(obs_embed)

            loss = criterion(pred_embed, action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('training_loss', loss.item(), global_step)

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] average loss: {avg_loss:.4f}")

        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"Learning rate after epoch {epoch+1}: {current_lr}")
            writer.add_scalar('learning_rate', current_lr, epoch + 1)

        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            save_model(save_dir, model, optimizer, epoch + 1, avg_loss)

    writer.close()



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root_dir = './dataset'
    image_size = (160, 256)
    batch_size = 8
    num_epochs = 1000
    learning_rate = 1e-3
    save_dir = './saved_models'

    # Dataset and dataloader
    dataset = CommonGameplayPromptActionDataset(root_dir=root_dir, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # MineCLIP config and loading weights
    cfg = {
        'arch': 'vit_base_p16_fz.v2.t2',
        'hidden_dim': 512,
        'image_feature_dim': 512,
        'mlp_adapter_spec': 'v0-2.t0',
        'pool_type': 'attn.d2.nh8.glusw',
        'resolution': [160, 256]
    }

    mineclip = MineCLIP(**cfg)
    mineclip_ckpt = "saved_models/mineclip/attn_new.pth"
    if os.path.exists(mineclip_ckpt):
        mineclip.load_state_dict(torch.load(mineclip_ckpt))
    else:
        raise FileNotFoundError(f"MineCLIP weights not found at {mineclip_ckpt}")

    policy = PolicyFromMineCLIP(
        input_dim=1024,
        embed_dim=512,
        num_actions=22
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    # scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    
    criterion = nn.BCEWithLogitsLoss()

    train_behavior_cloning_with_mineclip(
        model=policy,
        mineclip=mineclip,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=None,
        criterion=criterion,
        save_dir=save_dir,
        num_epochs=num_epochs,
        device=device
    )