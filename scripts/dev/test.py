import torch

from model.mineclip.mineclip import MineCLIP
from tools.utils import read_video_tensor

@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = {'arch': 'vit_base_p16_fz.v2.t2', 'hidden_dim': 512, 'image_feature_dim': 512, 'mlp_adapter_spec': 'v0-2.t0', 'pool_type': 'attn.d2.nh8.glusw', 'resolution': [160, 256]}

    model = MineCLIP(**cfg).to(device)
    model.load_state_dict(torch.load("saved_models/mineclip/attn_new.pth"))

    video = read_video_tensor("dataset/20241022_205338.mp4", resize_to=(256, 160), max_frames=32).unsqueeze(dim=0).to(device)
    prompts = [
        "Fight against an Ender dragon",
        "Shear sheep to obtain wool",
        "Attack enemy",
        "Move forward"
    ]

    VIDEO_BATCH, TEXT_BATCH = video.size(0), len(prompts)

    image_feats = model.forward_image_features(video)
    video_feats = model.forward_video_features(image_feats)

    print(f"image_feats: {image_feats.shape}")
    print(f"video_feats: {video_feats.shape}")

    text_feats_batch = model.encode_text(prompts)
    logits_per_video, logits_per_text = model.forward_reward_head(
        video_features=video_feats,
        text_tokens=text_feats_batch,
        softmax=True
    )

    print(f"logits_per_video: {logits_per_video}")
    print(f"shape: {logits_per_video.shape}")
    print(f"logits_per_text: {logits_per_text}")
    print(f"shape: {logits_per_text.shape}")


if __name__ == "__main__":
    main()
    


    