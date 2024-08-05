import torch
from data.dataset import GameplayActionPairVideoDataset
from torch import optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model.agent import Agent, device
from model.action_loss import ActionLoss
from tools.utils import custom_collate_fn
from torch.utils.tensorboard import SummaryWriter


root_dir = "output_logs"
writer = SummaryWriter()

agent = Agent(debug=False).to(device)
criterion = ActionLoss()
optimizer = optim.Adam(agent.parameters(), lr=0.001)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = GameplayActionPairVideoDataset(root_dir=root_dir, tokenizer=tokenizer)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

num_epochs = 20000

def train():
    for epoch in range(num_epochs):
        for step, (instruction, frames, action) in enumerate(dataloader):
            frames = frames.to(device)
            action = action.to(device)
            _, _, channel, height, width = frames.shape
            images = frames.reshape(-1, channel, height, width).to(device)
            optimizer.zero_grad()
            logits = agent(images, frames, instruction)
            loss = criterion(logits, action)
            loss.backward()
            optimizer.step()
            
            # Log the loss to TensorBoard
            writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + step)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        
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