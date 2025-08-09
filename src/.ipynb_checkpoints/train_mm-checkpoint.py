# src/train_mm.py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # proje kökünü path'e ekle

import math
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from src.utils.tokenizer_utils import get_tokenizer
from src.utils.data import CaptionJsonl, collate_fn
from src.models.vision_connector import VisionConnector
from src.models.mm_model import MiniVLM

def main():
    llm_name = "mistralai/Mistral-7B-Instruct-v0.3"
    train_index = "data/coco-mini/train.jsonl"
    val_index = "data/coco-mini/validation.jsonl"
    output_dir = "ckpts/mini-vlm"

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device

    tok = get_tokenizer(llm_name)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    train_ds = CaptionJsonl(train_index, tok, max_len=512)
    val_ds   = CaptionJsonl(val_index, tok, max_len=512)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, pad_id), num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False,
                              collate_fn=lambda b: collate_fn(b, pad_id), num_workers=2)

    connector = VisionConnector(tv_tokens=64, llm_hidden=4096)  # Mistral hidden 4096
    model = MiniVLM(llm_name, tokenizer=tok, connector=connector, lora_r=16, lora_alpha=32, lora_dropout=0.05)

    # Optimizasyon: connector ve LoRA parametrelerini eğit
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=0.01)

    num_epochs = 1
    steps_per_epoch = min(4000, len(train_loader))  # hızlı deneme
    max_steps = num_epochs * steps_per_epoch
    warmup = int(0.03 * max_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, max_steps)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    model.train()
    global_step = 0
    best_val = 1e9
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_loader):
            if step >= steps_per_epoch: break
            out = model(input_ids=batch["input_ids"].to(device), images=batch["images"])
            loss = out.loss
            accelerator.backward(loss)

            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

            if accelerator.is_local_main_process and global_step % 50 == 0:
                print(f"step {global_step} loss {loss.item():.4f}")
            global_step += 1

            if accelerator.is_local_main_process and global_step % 500 == 0:
                val_loss = evaluate(model, val_loader, device, accelerator)
                print(f"[val] step {global_step} loss {val_loss:.4f}")
                if val_loss < best_val:
                    best_val = val_loss
                    save_path = os.path.join(output_dir, f"step{global_step}_loss{val_loss:.3f}")
                    accelerator.unwrap_model(model).llm.save_pretrained(save_path)  # LoRA dahil LLM state
                    tok.save_pretrained(save_path)

def evaluate(model, val_loader, device, accelerator):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in val_loader:
            out = model(input_ids=batch["input_ids"].to(device), images=batch["images"])
            losses.append(accelerator.gather_for_metrics(out.loss.detach()).mean().item())
    model.train()
    return sum(losses)/len(losses)

if __name__ == "__main__":
    main()
