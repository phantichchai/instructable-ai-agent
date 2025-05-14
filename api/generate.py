import torch
from api.core import mineclip, agent, device
    
async def process_frames(frames, instruction):
    # List of 16 frames in BGR format
    frame_tensors = [torch.from_numpy(f).permute(2, 0, 1).float() / 255.0 for f in frames]

    # Stack into a batch of shape [16, C, H, W]
    video_tensor = torch.stack(frame_tensors).unsqueeze(0).to(device) # Shape: [16, 3, H, W]

    # Forward through your MineCLIP model
    video_embed = mineclip.encode_video(video_tensor)
    text_embed = mineclip.encode_text(instruction)
    obs_embed = torch.cat([video_embed, text_embed], dim=-1)

    # Predict action
    actions = agent.get_actions(obs_embed)
    return actions.cpu().numpy()