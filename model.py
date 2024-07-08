import torch
import torch.nn as nn
from transformers import VivitConfig, VivitModel, ViTModel, ViTConfig, BertModel, BertTokenizer, BertConfig, TransfoXLModel

device = "cuda" if torch.cuda.is_available() else "cpu"

class MultiModelEncoder(nn.Module):
    def __init__(self, vit_model_name='google/vit-base-patch16-224-in21k',
                 vivit_model_name='google/vivit-b-16x2-kinetics400',
                 bert_model_name='bert-base-uncased'):
        super(MultiModelEncoder, self).__init__()

        self.vit = ViTModel.from_pretrained(vit_model_name)
        self.vivit = VivitModel.from_pretrained(vivit_model_name)
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    def forward(self, image, video, text):
        vit_outputs = self.vit(image)
        vit_pooled_output = vit_outputs.pooler_output
        vit_last_hidden_state = vit_outputs.last_hidden_state

        vivit_outputs = self.vivit(video)
        vivit_pooled_output = vivit_outputs.pooler_output
        vivit_last_hidden_state = vivit_outputs.last_hidden_state

        text_encoding = self.bert_tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )
        input_ids = text_encoding['input_ids'].to(device)
        attention_mask = text_encoding['attention_mask'].to(device)
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_pooled_output = bert_outputs.pooler_output
        bert_last_hidden_state = bert_outputs.last_hidden_state

        concatenated_output = torch.cat((vit_pooled_output, vivit_pooled_output, bert_pooled_output), dim=1)
        last_hidden_states = torch.cat((vit_last_hidden_state, vivit_last_hidden_state, bert_last_hidden_state), dim=1)

        return { 
            "concatenated_output": concatenated_output,
            "last_hidden_states": last_hidden_states
        }
    

class MultiModelDecoder(nn.Module):
    def __init__(self, d_model=2304,
                 nhead=8,
                 num_layers=12
                ):
        super(MultiModelDecoder, self).__init__()
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_model = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, encoder_last_state, attension_state):
        transformer_outputs = self.transformer_model(encoder_last_state, attension_state)
        return transformer_outputs
    

class Policy(nn.Module):
    def __init__(self, d_model, hidden_size, num_actions):
        super(Policy, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.Dropout(0.1, False),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.1, False),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_actions)
        )
        
    def forward(self, input):
        return self.policy(input)


class Agent(nn.Module):
    def __init__(self, num_class):
        super(Agent, self).__init__()
        self.encoder = MultiModelEncoder()
        self.decoder = MultiModelDecoder()
        self.policy = Policy(2304, 2048, num_class)

    def forward(self, image, video, text):
        encoder_embed = self.encoder(image, video, text)
        decoder_embed = self.decoder(encoder_embed['concatenated_output'], encoder_embed['concatenated_output'])
        logits = self.policy(decoder_embed)
        return logits

    def get_model_size(self):
        param_size = 0
        buffer_size = 0
        for param in self.parameters():
            param_size += param.numel() * param.element_size()
        for buffer in self.buffers():
            buffer_size += buffer.numel() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return { 
            "size_all_mb" : size_all_mb,
            "parameter_size" : param_size
        }