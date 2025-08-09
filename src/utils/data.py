# src/utils/data.py
import json, random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

USER_PROMPT = "Aşağıdaki görüntüyü tek cümleyle açıkla."
# İstersen İngilizce: "Describe the image in one concise sentence."

class CaptionJsonl(Dataset):
    def __init__(self, index_path, tokenizer, max_len=512):
        self.recs = [json.loads(l) for l in open(index_path, "r", encoding="utf-8")]
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        img = Image.open(r["image_path"]).convert("RGB")
        cap = r["caption"].strip().replace("\n"," ")
        # Basit chat biçimi:
        # user: <image>\n{prompt}
        # assistant: {caption}
        user = f"<image>\n{USER_PROMPT}"
        assistant = cap
        # LLM giriş metni: user + assistant
        text = f"User: {user}\nAssistant: {assistant}"
        ids = self.tok(text, truncation=True, max_length=self.max_len, return_tensors="pt")["input_ids"][0]
        return {"image": img, "input_ids": ids}

def collate_fn(batch, pad_id):
    # Değişken uzunlukları pad'le
    images = [b["image"] for b in batch]
    ids = [b["input_ids"] for b in batch]
    maxL = max(x.size(0) for x in ids)
    out = torch.full((len(ids), maxL), pad_id, dtype=torch.long)
    for i, x in enumerate(ids): out[i, :x.size(0)] = x
    return {"images": images, "input_ids": out}
