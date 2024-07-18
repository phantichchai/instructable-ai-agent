import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
    
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