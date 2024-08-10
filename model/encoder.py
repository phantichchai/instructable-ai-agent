import torch
import torch.nn as nn
from transformers import ViTModel, BertModel, BertTokenizer
from model.cvivit import CvivitConfig, VideoTransformerModel
    

class MultiModelEncoderConfig:
    def __init__(self,
                 vit_model_name: str='google/vit-base-patch16-224-in21k', 
                 language_model_name: str='bert-base-uncased', 
                 cvivit_config: CvivitConfig=CvivitConfig()
        ) -> None:
        self.vit_model_name = vit_model_name
        self.language_model_name = language_model_name
        self.cvivit_config = cvivit_config


class MultiModelEncoder(nn.Module):
    def __init__(self,
                 config: MultiModelEncoderConfig=None, 
                 device: str=None,
                 debug: bool=False):
        super(MultiModelEncoder, self).__init__()

        self.bert = BertModel.from_pretrained(config.language_model_name)
        self.bert_tokenizer = BertTokenizer.from_pretrained(config.language_model_name)
        self.vit = ViTModel.from_pretrained(config.vit_model_name)
        self.cvivit = VideoTransformerModel(
            video_dimension=(config.cvivit_config.color_channel, config.cvivit_config.image_size, config.cvivit_config.image_size),
            emb_size=config.cvivit_config.emb_size,
            d_model=config.cvivit_config.d_model,
            patch_size=config.cvivit_config.patch_size,
            num_layers_spatial=config.cvivit_config.num_layers_spatial,
            num_heads_spatial=config.cvivit_config.num_heads_spatial,
            dim_feedforward_spatial=config.cvivit_config.dim_feedforward_spatial,
            dropout_spatial=config.cvivit_config.dropout_spatial,
            num_layers_temporal=config.cvivit_config.num_layers_temporal, 
            num_heads_temporal=config.cvivit_config.num_heads_temporal,
            dim_feedforward_temporal=config.cvivit_config.dim_feedforward_temporal,
            dropout_temporal=config.cvivit_config.dropout_temporal
        )
        self.debug = debug
        self.device = device

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
        ).to(self.device)
        input_ids = text_encoding['input_ids']
        attention_mask = text_encoding['attention_mask']
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        instruction_last_hidden_state = bert_outputs.last_hidden_state
        
        vision_last_hidden_states = torch.cat((instruction_last_hidden_state, vit_last_hidden_state, cvivit), dim=1)

        if self.debug:
            print(f"Vit last_hidden_states shape: {vit_last_hidden_state.shape}")
            print(f"Video last_hidden_states shape: {cvivit.shape}")
            print(f"Text last_hidden_states shape: {instruction_last_hidden_state.shape}")
            print(f"Token last_hidden_states shape: {vision_last_hidden_states.shape}")
        return {
            "vision_last_hidden_states": vision_last_hidden_states
        }