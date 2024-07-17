import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, video_dimensions, patch_dim, emb_size):
        super(PatchEmbedding, self).__init__()
        self.patch_dim = patch_dim
        self.emb_size = emb_size
        self.channels = video_dimensions[0]
        self.video_height = video_dimensions[1]
        self.video_width = video_dimensions[2]
        
        self.patch_video = nn.Conv3d(in_channels=self.channels,
                                    out_channels=self.emb_size,
                                    kernel_size=self.patch_dim,
                                    stride=self.patch_dim)
        
        self.patch_first_frame = nn.Conv2d(in_channels=self.channels,
                                           out_channels=self.emb_size,
                                           kernel_size=self.patch_dim[1:],
                                           stride=self.patch_dim[1:])
    
    def forward(self, x):
        first_frame, frames = torch.split(x, [1, x.size(1)-1], dim=1)
        first_frame = first_frame.transpose(1, 2).squeeze(dim=2)
        first_frame = self.patch_first_frame(first_frame)
        first_frame = first_frame.unsqueeze(dim=2)
        frames = frames.transpose(1, 2)
        frames = self.patch_video(frames)
        video_tokens = torch.cat([first_frame, frames], dim=2)
        video_tokens = video_tokens.flatten(2).transpose(1, 2)
        return video_tokens

class TransformerEncoder3D(nn.Module):
    def __init__(self, emb_size, num_layers, num_heads, dim_feedforward, dropout):
        super(TransformerEncoder3D, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        x = self.transformer_encoder(x)
        return x
    
class CausalSelfAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super(CausalSelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=emb_size, num_heads=num_heads, dropout=dropout)
        self.mask = None

    def forward(self, x):
        seq_len = x.size(0)
        if self.mask is None or self.mask.size(0) != seq_len:
            self.mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        attn_output, _ = self.attention(x, x, x, attn_mask=self.mask)
        return attn_output

class TemporalTransformerEncoder(nn.Module):
    def __init__(self, emb_size, num_layers, num_heads, dim_feedforward, dropout):
        super(TemporalTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=emb_size, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.causal_attention = CausalSelfAttention(emb_size, num_heads, dropout)
        
    def forward(self, x):
        x = x.transpose(0, 1)  # Transformer expects sequence as first dimension
        x = self.causal_attention(x)
        for layer in self.layers:
            x = layer(x)
        x = x.transpose(0, 1)  # Revert back to original dimensions
        return x
    
class VideoTransformerModel(nn.Module):
    def __init__(self, 
                 video_dimension, 
                 emb_size,
                 d_model,
                 patch_size, 
                 num_layers_spatial, 
                 num_heads_spatial, 
                 dim_feedforward_spatial, 
                 dropout_spatial, 
                 num_layers_temporal, 
                 num_heads_temporal, 
                 dim_feedforward_temporal, 
                 dropout_temporal):
        super(VideoTransformerModel, self).__init__()
        self.patch_embedding = PatchEmbedding(video_dimension, patch_size, emb_size)
        self.spatial_transformer = TransformerEncoder3D(d_model, num_layers_spatial, num_heads_spatial, dim_feedforward_spatial, dropout_spatial)
        self.temporal_transformer = TemporalTransformerEncoder(d_model, num_layers_temporal, num_heads_temporal, dim_feedforward_temporal, dropout_temporal)
        
    def forward(self, x):
        patches = self.patch_embedding(x)
        spatial_dims = patches.shape
        # print(f"Patches: {patches.shape}")
        spatially_encoded_patches = self.spatial_transformer(patches)
        # print(f"Spatial: {spatially_encoded_patches.shape}")
        temporally_encoded_patches = self.temporal_transformer(spatially_encoded_patches)
        # print(f"Temporal: {temporally_encoded_patches.shape}")
        batch_size = temporally_encoded_patches.size(0)
        return temporally_encoded_patches