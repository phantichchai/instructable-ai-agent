import torch

from model.mineclip.mineclip import MineCLIP
from torchinfo import summary


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = {'arch': 'vit_base_p16_fz.v2.t2', 'hidden_dim': 512, 'image_feature_dim': 512, 'mlp_adapter_spec': 'v0-2.t0', 'pool_type': 'attn.d2.nh8.glusw', 'resolution': [160, 256]}

    model = MineCLIP(**cfg).to(device)
    model.load_state_dict(torch.load("saved_models/mineclip/attn_new.pth"))

    video = torch.randint(0, 255, (6, 16, 3, 160, 256), device=device)
    prompts = [
        "hello, this is MineCLIP",
        "MineCLIP is a VideoCLIP model trained on YouTube dataset from MineDojo's knowledge base",
        "Feel free to also checkout MineDojo at",
        "https://minedojo.org",
    ]

    VIDEO_BATCH, TEXT_BATCH = video.size(0), len(prompts)

    image_feats = model.forward_image_features(video)
    video_feats = model.forward_video_features(image_feats)

    video_feats_2 = model.encode_video(video)
    text_feats_batch = model.encode_text(prompts)
    logits_per_video, logits_per_text = model.forward_reward_head(
        video_feats, text_tokens=text_feats_batch
    )

    print(f"logits_per_video: {logits_per_video}")
    print(f"shape: {logits_per_video.shape}")
    print(f"logits_per_text: {logits_per_text}")
    print(f"shape: {logits_per_text.shape}")


if __name__ == "__main__":
    main()
    


    