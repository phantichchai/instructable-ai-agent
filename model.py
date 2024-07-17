import torch
import torch.nn as nn
from transformers import ViTModel, BertModel, BertTokenizer, ViTConfig
from cvivit import VideoTransformerModel

device = "cuda" if torch.cuda.is_available() else "cpu"

class CvivitConfig:
    def __init__(
        self,
        image_size=64,
        color_channel=3,
        num_frames=32,
        emb_size=768,
        d_model=768,
        patch_size=(2, 8, 8),
        num_layers_spatial=4,
        num_heads_spatial=8,
        dim_feedforward_spatial=2048,
        dropout_spatial=0.1,
        num_layers_temporal=4,
        num_heads_temporal=8,
        dim_feedforward_temporal=2048,
        dropout_temporal=0.1,
    ):
        self.image_size = image_size
        self.color_channel = color_channel
        self.num_frames = num_frames
        self.emb_size = emb_size
        self.d_model = d_model
        self.patch_size = patch_size
        self.num_layers_spatial = num_layers_spatial
        self.num_heads_spatial = num_heads_spatial
        self.dim_feedforward_spatial = dim_feedforward_spatial
        self.dropout_spatial = dropout_spatial
        self.num_layers_temporal = num_layers_temporal
        self.num_heads_temporal = num_heads_temporal
        self.dim_feedforward_temporal = dim_feedforward_temporal
        self.dropout_temporal = dropout_temporal

class PolicyConfig:
    def __init__(
        self,
        d_model=768,
        hidden_size=2048,
        num_actions=12,
        seq_len=None
    ):
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.seq_len = seq_len

class MultiModelEncoder(nn.Module):
    def __init__(self, vit_model_name='google/vit-base-patch16-224-in21k',
                 bert_model_name='bert-base-uncased',
                 cvivit_config: CvivitConfig=CvivitConfig(),
                 debug: bool=False):
        super(MultiModelEncoder, self).__init__()

        self.vit = ViTModel(ViTConfig(image_size=64))
        self.cvivit = VideoTransformerModel(
            video_dimension=(cvivit_config.color_channel, cvivit_config.image_size, cvivit_config.image_size),
            emb_size=cvivit_config.emb_size,
            d_model=cvivit_config.d_model,
            patch_size=cvivit_config.patch_size,
            num_layers_spatial=cvivit_config.num_layers_spatial,
            num_heads_spatial=cvivit_config.num_heads_spatial,
            dim_feedforward_spatial=cvivit_config.dim_feedforward_spatial,
            dropout_spatial=cvivit_config.dropout_spatial,
            num_layers_temporal=cvivit_config.num_layers_temporal, 
            num_heads_temporal=cvivit_config.num_heads_temporal,
            dim_feedforward_temporal=cvivit_config.dim_feedforward_temporal,
            dropout_temporal=cvivit_config.dropout_temporal
        )
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.debug = debug

    def forward(self, image, video, text):
        vit_outputs = self.vit(image)
        vit_last_hidden_state = vit_outputs.last_hidden_state
        vit_last_hidden_state = vit_last_hidden_state.flatten(0, 1).unsqueeze(dim=0)

        cvivit = self.cvivit(video)
        
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
        instruction_last_hidden_state = bert_outputs.last_hidden_state
        
        vision_last_hidden_states = torch.cat((vit_last_hidden_state, cvivit), dim=1)

        if self.debug:
            print(f"Vit last_hidden_states shape: {vit_last_hidden_state.shape}")
            print(f"Video last_hidden_states shape: {cvivit.shape}")
            print(f"Text last_hidden_states shape: {instruction_last_hidden_state.shape}")
            print(f"Token last_hidden_states shape: {vision_last_hidden_states.shape}")
        return {
            "instruction_last_hidden_state": instruction_last_hidden_state,
            "vision_last_hidden_states": vision_last_hidden_states
        }
    

class MultiModelDecoder(nn.Module):
    def __init__(
        self,
        d_model=768,
        nhead=8,
        num_layers=4,
        debug:bool=False
    ):
        super(MultiModelDecoder, self).__init__()
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=num_layers)
        self.debug = debug

    def forward(self, instruction, memory):
        x = self.decoder(instruction, memory)

        if self.debug:
            print(f"Decoder outputs shape: {x.shape}")
        return x
    

class Policy(nn.Module):
    def __init__(self, policy_config=PolicyConfig()):
        super(Policy, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(policy_config.d_model, policy_config.hidden_size),
            nn.Dropout(0.1, False),
            nn.LayerNorm(policy_config.hidden_size),
            nn.Linear(policy_config.hidden_size, policy_config.num_actions),
        )
        
    def forward(self, input):
        return self.policy(input)


class Agent(nn.Module):
    def __init__(
        self,
        debug: bool=False
    ):
        super(Agent, self).__init__()
        self.debug = debug
        self.encoder = MultiModelEncoder(debug=self.debug)
        self.decoder = MultiModelDecoder(debug=self.debug)
        self.policy = Policy()

    def forward(self, image, video, text):
        encoder_outputs = self.encoder(image, video, text)
        decoder_outputs = self.decoder(encoder_outputs['instruction_last_hidden_state'], encoder_outputs['vision_last_hidden_states'])
        logits = self.policy(decoder_outputs)
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