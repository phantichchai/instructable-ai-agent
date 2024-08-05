import torch
from data.dataset import GameplayActionPairVideoDataset
from torch import optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model.agent import Agent, device
from model.action_loss import ActionLoss
from tools.utils import custom_collate_fn


root_dir = "output_logs"
agent = Agent(debug=False).to(device)
criterion = ActionLoss()
optimizer = optim.Adam(agent.parameters(), lr=0.001)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = GameplayActionPairVideoDataset(root_dir=root_dir, tokenizer=tokenizer)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

num_epochs = 1000

def train():
    for epoch in range(num_epochs):
        for batch, (instruction, frames, action) in enumerate(dataloader):
            frames = frames.to(device)
            action = action.to(device)
            _, _, channel, height, width = frames.shape
            images = frames.reshape(-1, channel, height, width).to(device)
            instruction = tokenizer.decode(*instruction)
            optimizer.zero_grad()
            logits = agent(images, frames, instruction)
            loss = criterion(logits, action)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

def save():
    torch.save(agent.state_dict(), 'model_weights.pth')

if __name__ == "__main__":
    train()
    save()