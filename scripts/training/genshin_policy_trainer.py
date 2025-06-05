import torch
import os
import numpy as np
import time
from typing import Optional
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tools.utils import generate_experiment_name
from data.dataset import CommonGameplayPromptActionDataset
from model.genshin.policy import PolicyFromMineCLIP
from model.mineclip.mineclip import MineCLIP
from sklearn.metrics import f1_score, precision_score, recall_score


saved_weights = []

def save_model(save_dir, model: nn.Module, optimizer: Optimizer, epoch, loss=None, experiment_name="default_experiment"):
    """Saves the training checkpoint and maintains only 5 most recent weights."""
    os.makedirs(save_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{experiment_name}_epoch{epoch}_{timestamp}.pt"
    model_path = os.path.join(save_dir, filename)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
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
    val_dataloader: DataLoader,  # NEW
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
    model_name = model.__class__.__name__
    experiment_name = generate_experiment_name(
        model_name=model_name,
        experiment="SingleAction_slr1e-3g1e-1",
        dataset="ConsistentV2.3",
        prompt_style="ACTION",
        seed=42,
        include_epoch=False,
        add_timestamp=False
    )
    writer = SummaryWriter(log_dir=f"runs/{experiment_name}")

    start_epoch = 0
    best_val_loss = float("inf")
    if resume_checkpoint:
        start_epoch, _ = load_checkpoint(resume_checkpoint, model, optimizer, device)
        print(f"[RESUME] Training resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, data in enumerate(dataloader):
            video = data['frames'].to(device)
            text_prompt = data['text_prompt']
            action = data["action"].to(device)

            with torch.no_grad():
                video_embed = mineclip.encode_video(video)
                text_embed = mineclip.encode_text(text_prompt)
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

            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            writer.add_scalar("grad_norm", total_norm, global_step)

            predicted_labels = (torch.sigmoid(pred_embed) > 0.5).int()
            for cls in range(model.output_dim):
                count = predicted_labels[:, cls].sum().item()
                writer.add_scalar(f'predicted_class_distribution/class_{cls}', count, global_step)

            confidences = torch.sigmoid(pred_embed)
            mean_max_confidence = confidences.max(dim=1)[0].mean().item()
            writer.add_scalar("prediction_confidence", mean_max_confidence, global_step)

            if global_step % 500 == 0:
                writer.add_text("sample_prompt", text_prompt[0], global_step)
                pred_vector_str = ", ".join(f"{x:.4f}" for x in pred_embed[0].tolist())
                writer.add_text("predicted_output_vector", pred_vector_str, global_step)
                true_vector_str = ", ".join(f"{x:.1f}" for x in action[0].tolist())
                writer.add_text("true_action_vector", true_vector_str, global_step)

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
                print(f"[Epoch {epoch+1}/{num_epochs}] Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(dataloader)
        print(f"[Epoch {epoch+1}/{num_epochs}] Average Training Loss: {avg_loss:.4f}")
        writer.add_scalar('epoch_training_loss', avg_loss, epoch + 1)

        # Validation step
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for val_batch in val_dataloader:
                video = val_batch['frames'].to(device)
                text_prompt = val_batch['text_prompt']
                action = val_batch["action"].to(device)

                video_embed = mineclip.encode_video(video)
                text_embed = mineclip.encode_text(text_prompt)
                text_embed = text_embed / text_embed.norm(dim=1, keepdim=True)

                obs_embed = torch.cat([video_embed, text_embed], dim=-1)
                pred_embed = model(obs_embed)
                loss = criterion(pred_embed, action)
                val_loss += loss.item()

                pred_binary = (torch.sigmoid(pred_embed ) > 0.5).int().cpu().numpy()
                all_preds.append(pred_binary)
                all_labels.append(action.cpu().numpy())

        avg_val_loss = val_loss / len(val_dataloader)
        writer.add_scalar('epoch_validation_loss', avg_val_loss, epoch + 1)
        print(f"[Epoch {epoch+1}] Validation Loss: {avg_val_loss:.4f}")

        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)
        precision = precision_score(all_labels, all_preds, average="micro", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="micro", zero_division=0)

        writer.add_scalar('val_f1_score_micro', f1, epoch + 1)
        writer.add_scalar('val_precision_micro', precision, epoch + 1)
        writer.add_scalar('val_recall_micro', recall, epoch + 1)

        if f1 < 1.0:
            print(f"[Metrics] F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"Learning Rate after Epoch {epoch+1}: {current_lr}")
            writer.add_scalar('learning_rate', current_lr, epoch + 1)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_model(save_dir, model, optimizer, epoch + 1, avg_val_loss, experiment_name)
            print(f"✅ [Checkpoint] Best model saved at epoch {epoch+1} with val_loss={avg_val_loss:.4f}")

    writer.close()



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root_dir = './dataset'
    image_size = (160, 256)
    batch_size = 8
    num_epochs = 800
    learning_rate = 1e-3
    save_dir = './saved_models'

    # Dataset and dataloader
    full_dataset = CommonGameplayPromptActionDataset(root_dir=root_dir, image_size=image_size)
    
    val_ratio = 0.1  # 10% validation
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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
    scheduler = StepLR(optimizer, step_size=200, gamma=0.1)
    
    criterion = nn.BCEWithLogitsLoss()

    train_behavior_cloning_with_mineclip(
        model=policy,
        mineclip=mineclip,
        dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        save_dir=save_dir,
        num_epochs=num_epochs,
        device=device,
    )