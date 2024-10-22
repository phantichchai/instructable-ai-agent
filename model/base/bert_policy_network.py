from torch import nn
from transformers import BertTokenizer, BertModel

class BertPolicyNetwork(nn.Module):
    def __init__(self, text_output_size=256, num_actions=10, device=None) -> None:
        super(BertPolicyNetwork, self).__init__()
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.text_fc = nn.Linear(self.text_encoder.config.hidden_size, text_output_size)

        self.fc = nn.Sequential(
            nn.Linear(text_output_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3)  # Dropout with 30% probability
        )
        self.output_layer = nn.Linear(512, num_actions)

    def forward(self, text):
        text_inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        text_inputs = text_inputs.to(self.device)
        text_outputs = self.text_encoder(**text_inputs)
        text_embedding = self.text_fc(text_outputs.last_hidden_state[:, 0, :])

        combined_output = self.fc(text_embedding)
        action_logits = self.output_layer(combined_output)

        return action_logits