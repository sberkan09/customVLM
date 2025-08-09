# src/models/vision_connector.py
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor

class VisionConnector(nn.Module):
    """
    CLIP ViT-L/14 -> patch tokenleri (SxD=~257x1024) al
    1) CLS'i at, S-1 patch'i kalır
    2) AdaptiveAvgPool1d ile S -> T_v 'ye indir (örn. 64)
    3) MLP ile D(1024) -> H(LLM hidden, örn. 4096) projekte et
    Son: (B, T_v, H) döner
    """
    def __init__(self, tv_tokens=64, llm_hidden=4096, clip_name="openai/clip-vit-large-patch14"):
        super().__init__()
        self.vision = CLIPVisionModel.from_pretrained(clip_name)
        for p in self.vision.parameters():
            p.requires_grad = False  # Frozen
        self.processor = CLIPImageProcessor.from_pretrained(clip_name)
        clip_width = self.vision.config.hidden_size  # 1024 (ViT-L/14)
        self.pool = nn.AdaptiveAvgPool1d(tv_tokens)
        self.proj = nn.Sequential(
            nn.Linear(clip_width, llm_hidden),
            nn.GELU(),
            nn.Linear(llm_hidden, llm_hidden),
        )

    @torch.no_grad()
    def encode_image(self, images):
        # images: list of PIL
        px = self.processor(images=images, return_tensors="pt")
        px = {k: v.to(self.vision.device) for k, v in px.items()}
        out = self.vision(**px).last_hidden_state  # (B, S, 1024), S=CLS+patches
        return out

    def forward(self, images):
        """
        images: list[PIL] -> (B, T_v, H)
        """
        with torch.no_grad():
            hs = self.encode_image(images)  # (B,S,1024)
        hs = hs[:, 1:, :]                  # CLS'i at -> (B,S-1,1024)
        hs = hs.transpose(1, 2)            # (B,1024,S-1)
        hs = self.pool(hs)                 # (B,1024,T_v)
        hs = hs.transpose(1, 2).contiguous()  # (B,T_v,1024)
        hs = self.proj(hs)                 # (B,T_v,H)
        return hs
