# src/train_mm.py
import os
import json
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator
from accelerate.utils import DistributedType
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

from PIL import Image

# Proje içi importlar
from src.models.vision_connector import VisionConnector
from src.models.mm_model import MiniVLM
from src.utils.tokenizer_utils import get_tokenizer


# -------------------------
# Yardımcı: reproducibility
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# JSONL tabanlı veri kümesi
# Her satır: {"image_path": "...", "caption": "..."}
# -------------------------
class JsonlImageTextDataset(Dataset):
    def __init__(self, index_path: str, tokenizer: AutoTokenizer, max_len: int):
        super().__init__()
        self.recs = []
        with open(index_path, "r", encoding="utf-8") as fr:
            for line in fr:
                line = line.strip()
                if not line:
                    continue
                self.recs.append(json.loads(line))
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        rec = self.recs[i]
        img_path = rec["image_path"]
        cap = rec["caption"]

        # Görüntüyü oku (RGB)
        img = Image.open(img_path).convert("RGB")

        # Basit bir prompt şablonu: <image> sonrası öğrenilecek hedef "Assistant: ..." metni
        # Not: MiniVLM.build_inputs, <image>'e kadar ve görsel tokenları maskeler.
        text = f"User: <image>\nAssistant: {cap}"

        ids = self.tok(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        return {"input_ids": ids, "image": img}


# -------------------------
# Collate: input_ids -> batch tensor, images -> liste
# -------------------------
def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    images = [b["image"] for b in batch]  # VisionConnector kendi içinde handle edecek
    return {"input_ids": input_ids, "images": images}


# -------------------------
# Argümanlar
# -------------------------
@dataclass
class Args:
    base_llm: str
    train_index: str
    val_index: str
    tv_tokens: int = 64
    max_len: int = 512
    lr_connector: float = 1e-3
    lr_lora: float = 5e-5
    epochs: int = 1
    steps_per_epoch: int = 4000
    grad_accum: int = 16
    batch_size: int = 1
    num_workers: int = 2
    out_dir: str = "ckpts/mini-vlm"
    seed: int = 42


def parse_args() -> Args:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--base_llm", type=str, required=True)
    p.add_argument("--train_index", type=str, required=True)
    p.add_argument("--val_index", type=str, required=True)
    p.add_argument("--tv_tokens", type=int, default=64)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--lr_connector", type=float, default=1e-3)
    p.add_argument("--lr_lora", type=float, default=5e-5)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--steps_per_epoch", type=int, default=4000)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--out_dir", type=str, default="ckpts/mini-vlm")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    return Args(**vars(a))


# -------------------------
# Optimizasyon yardımcıları
# -------------------------
def build_optim_groups(model: MiniVLM, lr_connector: float, lr_lora: float):
    # 1) Vision connector parametreleri (tamamen eğitilecek)
    conn_params = [p for p in model.connector.parameters() if p.requires_grad]
    conn_param_ids = {id(p) for p in conn_params}

    # 2) LoRA/LLM tarafındaki trainable parametreler
    #    (PEFT ile sadece LoRA katmanları requires_grad=True olur)
    lora_params = [p for p in model.llm.parameters() if p.requires_grad and id(p) not in conn_param_ids]

    # (Opsiyonel) hızlı sanity-check
    print(f"[optim] connector params: {sum(p.numel() for p in conn_params):,} | "
          f"lora params: {sum(p.numel() for p in lora_params):,}")

    return [
        {"params": conn_params, "lr": lr_connector, "weight_decay": 0.01},
        {"params": lora_params, "lr": lr_lora, "weight_decay": 0.0},
    ]


# -------------------------
# Ana akış
# -------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum, mixed_precision="fp16")
    is_main = accelerator.is_main_process

    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)
        print("Args:", args)

    # 1) Tokenizer
    tok = get_tokenizer(args.base_llm)  # trust_remote_code=True içeride verildi

    # 2) LLM config → gerçek hidden_size'i çek
    cfg = AutoConfig.from_pretrained(args.base_llm, trust_remote_code=True)
    llm_hidden = (
        getattr(cfg, "hidden_size", None)
        or getattr(cfg, "n_embd", None)
        or getattr(cfg, "d_model", None)
    )
    assert llm_hidden is not None, "LLM hidden size bulunamadı"
    if is_main:
        print(f"[debug] LLM hidden_size = {llm_hidden}, tv_tokens = {args.tv_tokens}")

    # 3) VisionConnector (boyutu LLM'e uyumlu)
    connector = VisionConnector(tv_tokens=args.tv_tokens, llm_hidden=llm_hidden)

    # 4) MiniVLM (bnb 4-bit QLoRA, minimal prepare - düşük VRAM)
    #    MiniVLM içinde: AutoModelForCausalLM(..., quantization_config=BitsAndBytesConfig(load_in_4bit=True, ...),
    #    trust_remote_code=True, gradient checkpointing, use_cache=False, LoRA eklenmiş)
    model = MiniVLM(
        base_llm_name=args.base_llm,
        tokenizer=tok,
        connector=connector,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    )

    # 5) Dataset & DataLoader
    train_ds = JsonlImageTextDataset(args.train_index, tok, args.max_len)
    val_ds = JsonlImageTextDataset(args.val_index, tok, args.max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.num_workers // 2),
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # 6) Optimizer & Scheduler (iki parametre grubu)
    optim_groups = build_optim_groups(model, args.lr_connector, args.lr_lora)
    optimizer = torch.optim.AdamW(optim_groups, betas=(0.9, 0.95), eps=1e-8)

    total_train_steps = args.steps_per_epoch * args.epochs
    warmup_steps = max(100, int(0.03 * total_train_steps))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_train_steps
    )

    # 7) Accelerate hazırlığı
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # 8) Eğitim döngüsü
    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        running = 0.0
        for step, batch in enumerate(train_loader):
            # steps_per_epoch sınırı
            if step >= args.steps_per_epoch:
                break

            with accelerator.accumulate(model):
                input_ids = batch["input_ids"].to(accelerator.device, non_blocking=True)
                images = batch["images"]  # VisionConnector içinde to(device) yapılacak

                out = model(input_ids=input_ids, images=images)  # MiniVLM.forward loss döndürür
                loss = out.loss if hasattr(out, "loss") else out[0]

                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                running += loss.item()
                if is_main and (global_step % 20 == 0):
                    avg = running / (step + 1)
                    print(f"epoch {epoch} step {step}/{args.steps_per_epoch}  loss {loss.item():.4f}  avg {avg:.4f}")

            global_step += 1

        # Basit bir val: birkaç batch'in loss'u
        model.eval()
        val_loss, val_count = 0.0, 0
        with torch.no_grad():
            for vstep, vbatch in enumerate(val_loader):
                if vstep >= 50:  # hızlı bir örnekleme
                    break
                vinp = vbatch["input_ids"].to(accelerator.device, non_blocking=True)
                vimgs = vbatch["images"]
                vout = model(input_ids=vinp, images=vimgs)
                vloss = vout.loss if hasattr(vout, "loss") else vout[0]
                val_loss += vloss.item()
                val_count += 1
        if is_main and val_count:
            print(f"[val] epoch {epoch}  loss {val_loss/val_count:.4f}")
        model.train()

        # 9) Checkpoint kaydet (yalnızca ana süreç)
        if is_main:
            ep_dir = os.path.join(args.out_dir, f"epoch{epoch}")
            os.makedirs(ep_dir, exist_ok=True)
            # LoRA + tokenizer + connector kaydı
            try:
                model.llm.save_pretrained(ep_dir)  # PEFT adapter ağırlıkları
            except Exception as e:
                print("[warn] llm.save_pretrained hata:", e)
            try:
                tok.save_pretrained(ep_dir)
            except Exception as e:
                print("[warn] tokenizer save_pretrained hata:", e)
            try:
                torch.save(model.connector.state_dict(), os.path.join(ep_dir, "vision_connector.pt"))
            except Exception as e:
                print("[warn] connector save hata:", e)

    if is_main:
        print("Eğitim bitti.")


if __name__ == "__main__":
    main()
