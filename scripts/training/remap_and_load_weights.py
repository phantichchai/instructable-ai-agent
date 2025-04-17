import torch

from model.mineclip.mineclip import MineCLIP
from torchinfo import summary

def remap_and_load_weights(model: MineCLIP, old_checkpoint_path, key_map: dict, new_checkpoint_path):
    """
    Loads weights from old keys to new keys using a provided key_map.

    Args:
        model: your new model instance
        old_checkpoint_path: path to the old .pth file
        key_map: dict of {new_key: old_key}
    """
    # Load the old checkpoint
    old_state_dict = torch.load(old_checkpoint_path)

    # If the old checkpoint is wrapped in a 'state_dict'
    if 'state_dict' in old_state_dict:
        old_state_dict = old_state_dict['state_dict']

    new_state_dict = model.state_dict()

    for new_key, old_key in key_map.items():
        if old_key in old_state_dict and new_key in new_state_dict:
            new_state_dict[new_key] = old_state_dict[old_key]
            print(f"Mapped: {old_key} --> {new_key}")
        else:
            print(f"❌ Could not map: {old_key} --> {new_key} (missing in source or target)")

    # Load into model
    model.load_state_dict(new_state_dict, strict=False)
    
    # Save the new model weights
    torch.save(model.state_dict(), new_checkpoint_path)
    print(f"\n💾 Saved new checkpoint to: {new_checkpoint_path}")


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = {'arch': 'vit_base_p16_fz.v2.t2', 'hidden_dim': 512, 'image_feature_dim': 512, 'mlp_adapter_spec': 'v0-2.t0', 'pool_type': 'attn.d2.nh8.glusw', 'resolution': [160, 256]}

    model = MineCLIP(**cfg).to(device)

    mapping = {
        "temporal_encoder.attn.model.attn_layers.final_norm.gamma": "model.temporal_encoder.attn.model.norm.weight",
        "temporal_encoder.attn.model.attn_layers.layers.0.0.0.gamma": "model.temporal_encoder.attn.model.attn_layers.layers.0.0.0.weight", 
        "temporal_encoder.attn.model.attn_layers.layers.1.0.0.gamma": "model.temporal_encoder.attn.model.attn_layers.layers.1.0.0.weight", 
        "temporal_encoder.attn.model.attn_layers.layers.1.1.ff.0.proj.weight": "model.temporal_encoder.attn.model.attn_layers.layers.1.1.net.0.proj.weight", 
        "temporal_encoder.attn.model.attn_layers.layers.1.1.ff.0.proj.bias": "model.temporal_encoder.attn.model.attn_layers.layers.1.1.net.0.proj.bias", 
        "temporal_encoder.attn.model.attn_layers.layers.1.1.ff.2.weight": "model.temporal_encoder.attn.model.attn_layers.layers.1.1.net.3.weight", 
        "temporal_encoder.attn.model.attn_layers.layers.1.1.ff.2.bias": "model.temporal_encoder.attn.model.attn_layers.layers.1.1.net.3.bias", 
        "temporal_encoder.attn.model.attn_layers.layers.2.0.0.gamma": "model.temporal_encoder.attn.model.attn_layers.layers.2.0.0.weight", 
        "temporal_encoder.attn.model.attn_layers.layers.3.0.0.gamma": "model.temporal_encoder.attn.model.attn_layers.layers.3.0.0.weight", 
        "temporal_encoder.attn.model.attn_layers.layers.3.1.ff.0.proj.weight": "model.temporal_encoder.attn.model.attn_layers.layers.3.1.net.0.proj.weight", 
        "temporal_encoder.attn.model.attn_layers.layers.3.1.ff.0.proj.bias": "model.temporal_encoder.attn.model.attn_layers.layers.3.1.net.0.proj.bias", 
        "temporal_encoder.attn.model.attn_layers.layers.3.1.ff.2.weight": "model.temporal_encoder.attn.model.attn_layers.layers.3.1.net.3.weight", 
        "temporal_encoder.attn.model.attn_layers.layers.3.1.ff.2.bias": "model.temporal_encoder.attn.model.attn_layers.layers.3.1.net.3.bias"
    }

    remap_and_load_weights(model=model, old_checkpoint_path="saved_models/mineclip/attn.pth", key_map=mapping, new_checkpoint_path="saved_models/mineclip/attn_new.pth")
        
if __name__ == "__main__":
    main()
    