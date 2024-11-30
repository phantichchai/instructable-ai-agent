import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from data.dataset import CommonGameplayPromptActionDataset
from model.base.multi_model_policy_network import MultiModalModel
from model.base.bert_policy_network import BertPolicyNetwork

class Trainer:
    def __init__(self, model, dataloader, optimizer, criterion, num_epochs, save_dir, device):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.device = device
        self.writer = SummaryWriter(log_dir=f"runs/{self.model.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        self.saved_weights = []

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def save_model(self, epoch):
        """Saves the model and maintains only 5 most recent weights."""
        model_name = f"{self.model.__class__.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_epoch_{epoch}.pt"
        model_path = os.path.join(self.save_dir, model_name)
        torch.save(self.model.state_dict(), model_path)
        self.saved_weights.append(model_path)

        # Remove old weights if more than 5 saved
        if len(self.saved_weights) > 5:
            oldest_model = self.saved_weights.pop(0)
            if os.path.exists(oldest_model):
                os.remove(oldest_model)

    def train(self):
        self.model.to(self.device)
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for batch_idx, data in enumerate(self.dataloader):
                # Move data to device
                frames = data['frames'].to(self.device)
                text_prompt = data['text_prompt']
                action = data['action'].to(self.device)

                # Only take the first frame of the video for this example
                image_input = frames[:, 0, :, :, :] # Assuming we're using the first frame as the image input

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(text_prompt)
                loss = self.criterion(outputs, action)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                # Track loss
                running_loss += loss.item()

                if batch_idx % 10 == 9:  # Log every 10 mini-batches
                    avg_loss = running_loss / 10
                    print(f'Epoch [{epoch+1}/{self.num_epochs}], Batch [{batch_idx+1}], Loss: {avg_loss:.4f}')
                    self.writer.add_scalar('training_loss', avg_loss, epoch * len(self.dataloader) + batch_idx)
                    running_loss = 0.0

            # Save model after each epoch
            if epoch % 250 == 0:
                self.save_model(epoch)

        print("Finished Training")
        self.writer.close()

if __name__ == "__main__":
    # Set up parameters
    root_dir = './dataset'  # Path to your dataset
    image_size = (224, 224)  # Image size for the input
    batch_size = 8
    num_epochs = 2000
    learning_rate = 1e-3
    save_dir = './saved_models'  # Directory to save models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Dataset
    dataset = CommonGameplayPromptActionDataset(root_dir=root_dir, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize Model, Optimizer, and Loss Function
    model = BertPolicyNetwork(num_actions=dataset.action_size, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()  # Assuming multi-label classification

    # Training Loop
    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=num_epochs,
        save_dir=save_dir,
        device=device
    )

    # Run the training process
    trainer.train()