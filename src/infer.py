# src/infer.py
import os
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from src.models.vision_connector import VisionConnector

# Ortam değişkeniyle override edilebilir
BASE_LLM = os.environ.get("BASE_LLM", "lmsys/gpt-oss-20b-bf16")
CKPT_DIR = "ckpts/mini-vlm/epoch0"          # train_mm.py'nin kaydettiği klasör
CONNECTOR_PATH = f"{CKPT_DIR}/vision_connector.pt"

def load_model():
    # 1) Tokenizer'ı CKPT'ten yükle (vocab eşleşmesi için KRİTİK)
    tok = AutoTokenizer.from_pretrained(CKPT_DIR, use_fast=True, trust_remote_code=True)

    # 2) Base LLM'i 4-bit ile getir
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    llm = AutoModelForCausalLM.from_pretrained(
        BASE_LLM,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=bnb_cfg,
        torch_dtype=None,
    )

    # 3) Embedding'i tokenizer uzunluğuna göre yeniden boyutlandır
    llm.resize_token_embeddings(len(tok))

    # 4) LoRA adapter'ı yükle
    llm = PeftModel.from_pretrained(llm, CKPT_DIR)
    llm.eval()

    # 5) VisionConnector'ı kur ve ağırlıkları yükle
    hidden_size = llm.config.hidden_size
    connector = VisionConnector(tv_tokens=32, llm_hidden=hidden_size)
    sd = torch.load(CONNECTOR_PATH, map_location="cpu", weights_only=True)
    connector.load_state_dict(sd)
    connector = connector.to(next(llm.parameters()).device)

    return llm, tok, connector

@torch.inference_mode()
def describe(image_path: str) -> str:
    llm, tok, connector = load_model()
    img = Image.open(image_path).convert("RGB")

    # Basit prompt
    prompt = (
      "System: You are a helpful assistant who gives short and clear answers.\n"
      "User: <image>\n"
      "Describe the image in one concise sentence.\n"
      "Assistant:"
    )
    input_ids = tok(prompt, return_tensors="pt").input_ids.to(next(llm.parameters()).device)

    # Text ve görsel embedding'lerini birleştir
    wte = llm.get_input_embeddings()
    text_embeds = wte(input_ids)  # (1, L, H) - genelde fp16
    img_embeds = connector([img]).to(device=text_embeds.device, dtype=text_embeds.dtype)  # (1, T_v, H)
    inputs_embeds = torch.cat([text_embeds, img_embeds], dim=1)

    attn_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)

    out = llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attn_mask,
        max_new_tokens=48,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.1,
        eos_token_id=tok.eos_token_id,
    )

    text = tok.decode(out[0], skip_special_tokens=True)
    return text.split("Assistant:")[-1].strip()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    print(describe(args.image))
