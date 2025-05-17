import torch
from model.genshin.policy import PolicyFromMineCLIP
from model.mineclip.mineclip import MineCLIP
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = {'arch': 'vit_base_p16_fz.v2.t2', 'hidden_dim': 512, 'image_feature_dim': 512, 'mlp_adapter_spec': 'v0-2.t0', 'pool_type': 'attn.d2.nh8.glusw', 'resolution': [160, 256]}
mineclip = MineCLIP(**cfg).to(device)
mineclip.load_state_dict(torch.load("saved_models/mineclip/attn_new.pth"))

agent = PolicyFromMineCLIP(1024, 512, 22).to(device)
checkpoint = torch.load(os.getenv("POLICY_PATH"), map_location=device)
agent.load_state_dict(checkpoint["model_state_dict"])
agent.eval()

