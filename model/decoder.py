import torch.nn as nn


class MultiModelDecoderConfig:
    def __init__(self, 
                d_model=768,
                dim_feedforward=2048, 
                nhead=8, 
                num_layers=4
        ) -> None:
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.nhead = nhead
        self.num_layers = num_layers

class MultiModelDecoder(nn.Module):
    def __init__(
        self,
        config: MultiModelDecoderConfig=None,
        debug:bool=False
    ):
        super(MultiModelDecoder, self).__init__()
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward, 
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=config.num_layers)
        self.debug = debug

    def forward(self, target, memory):
        x = self.decoder(target, memory)

        if self.debug:
            print(f"Decoder outputs shape: {x.shape}")
        return x