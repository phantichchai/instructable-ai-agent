import torch
import time
from data.dataset import GameplayActionPairVideoDataset
from torch import optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model.agent import Agent, AgentConfig, device
from model.action_loss import ActionLoss
from model.cvivit import CvivitConfig
from model.encoder import MultiModelEncoderConfig
from model.decoder import MultiModelDecoderConfig
from tools.utils import custom_collate_fn
from torch.utils.tensorboard import SummaryWriter


# Set up TensorBoard writer
writer = SummaryWriter(log_dir='runs/behavior_cloning')

# Initialize your model, loss, optimizer
config = AgentConfig(
    encoder_config=MultiModelEncoderConfig(
        vit_model_name='google/vit-base-patch16-224-in21k',
        language_model_name='bert-base-uncased',
        cvivit_config=CvivitConfig(
            image_size=224,
            color_channel=3,
            emb_size=768,
            d_model=768,
            patch_size=(2, 8, 8),
            num_layers_spatial=2,
            num_heads_spatial=4,
            dim_feedforward_spatial=512,
            dropout_spatial=0.1,
            num_layers_temporal=2,
            num_heads_temporal=4,
            dim_feedforward_temporal=512,
            dropout_temporal=0.1
        )
    ),
    decoder_config=MultiModelDecoderConfig(
        d_model=768,
        dim_feedforward=512,
        nhead=4,
        num_layers=2
    )
)
agent = Agent(config=config, debug=False).to(device)
criterion = ActionLoss()
optimizer = optim.Adam(agent.parameters(), lr=0.001)

# Load data
root_dir = "output_logs"
dataset = GameplayActionPairVideoDataset(root_dir=root_dir, image_size=(224, 224))
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

epochs = 20000

def train():
    for epoch in range(epochs):
        running_loss = 0.0
        for batch, (instruction, frames, action) in enumerate(dataloader):
            optimizer.zero_grad()
            frames = frames.to(device)
            action = action.to(device)
            _, _, channel, height, width = frames.shape
            images = frames.reshape(-1, channel, height, width).to(device)
            logits = agent(images, frames, instruction)
            loss = criterion(logits, action)
            loss.backward()
            optimizer.step()

            # Accumulate the loss
            running_loss += loss.item()

            # Log gradients and weights
            # for name, param in agent.named_parameters():
            #     if param.grad is not None:
            #         writer.add_histogram(f'{name}.grad', param.grad, epoch)
            #     writer.add_histogram(name, param, epoch)
        
        # Log average loss per epoch
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        writer.add_scalar('training_loss_per_epoch', avg_loss, epoch)

        # Log learning rate
        for param_group in optimizer.param_groups:
            writer.add_scalar('learning_rate', param_group['lr'], epoch)
        
        # Log the time taken per epoch
        writer.add_scalar('time_per_epoch', time.time(), epoch)
            
        # Save the model weights every 2000 epochs
        if (epoch + 1) % 2000 == 0:
            save()

def save():
    torch.save(agent.state_dict(), 'model_weights.pth')
    print("Model weights saved.")

def close_writer():
    writer.close()

if __name__ == "__main__":
    train()
    close_writer()