import torch
import torch.nn as nn
from model.encoder import MultiModelEncoder
from model.decoder import MultiModelDecoder
from model.policy import Policy

device = "cuda" if torch.cuda.is_available() else "cpu"

class Agent(nn.Module):
    def __init__(
        self,
        debug: bool=False
    ):
        super(Agent, self).__init__()
        self.debug = debug
        self.encoder = MultiModelEncoder(debug=self.debug, device=device)
        self.decoder = MultiModelDecoder(debug=self.debug)
        self.policy = Policy()

    def forward(self, image, video, text):
        batch, frame, _, _, _ = video.shape
        encoder_outputs = self.encoder(image, video, text)
        memory = encoder_outputs['vision_last_hidden_states']
        tgt = torch.zeros(batch, frame, memory.shape[-1]).to(device)
        decoder_outputs = self.decoder(tgt, memory)
        logits = self.policy(decoder_outputs)
        return logits
    
    def predict(self, video, instruction):
        self.eval()
        with torch.inference_mode():
            _, _, channel, height, width = video.shape
            images = video.reshape(-1, channel, height, width)
            images.to(device)
            video.to(device)
            logits = self(images, video, instruction)
        return logits
    
    def get_actions(self, video, instruction):
        actions = self.predict(video, instruction)
        return actions

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