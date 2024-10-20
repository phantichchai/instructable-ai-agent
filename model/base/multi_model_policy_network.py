import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertTokenizer, BertModel

class MultiModalModel(nn.Module):
    def __init__(self, image_output_size=256, text_output_size=256, num_actions=10, device=None, debug=False):
        super(MultiModalModel, self).__init__()
        self.device = device
        self.debug = debug
        
        # Image Encoder: Pre-trained ResNet model
        self.image_encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # Using ResNet-18 as an example
        self.image_encoder.fc = nn.Linear(self.image_encoder.fc.in_features, image_output_size)

        # Text Encoder: Pre-trained BERT model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.text_fc = nn.Linear(self.text_encoder.config.hidden_size, text_output_size)

        # Fusion Layer
        self.fc = nn.Linear(image_output_size + text_output_size, 512)
        self.output_layer = nn.Linear(512, num_actions)

    def forward(self, image, text):
        # Process image through the image encoder
        if self.debug:
            print(f"Image shape: {image.shape}")
        image_embedding = self.image_encoder(image)
        if self.debug:
            print(f"Image embedding shape: {image_embedding.shape}")

        # Process text through the text encoder
        # Tokenize the input text and convert to tensors
        text_inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        text_inputs = text_inputs.to(self.device)
        text_outputs = self.text_encoder(**text_inputs)
        text_embedding = self.text_fc(text_outputs.last_hidden_state[:, 0, :])  # Use [CLS] token representation    

        # Concatenate embeddings
        combined_embedding = torch.cat((image_embedding, text_embedding), dim=1)

        # Pass through the fusion layer
        combined_output = self.fc(combined_embedding)
        action_logits = self.output_layer(combined_output)

        return action_logits

# Example usage
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Create a sample model
    model = MultiModalModel(num_actions=10, device=device).to(device)

    # Sample data (replace with your actual data)
    # Example: random image tensor of shape (batch_size, channels, height, width)
    sample_image = torch.randn(4, 3, 224, 224).to(device)  # Batch of 4 images
    sample_text = ["This is a test input.", "Another sample input.", "Third example text.", "Fourth one here."]  # Example text inputs

    # Forward pass
    output = model(sample_image, sample_text)
    print("Output shape:", output.shape)  # Should be (batch_size, num_actions)
