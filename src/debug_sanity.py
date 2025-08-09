# src/debug_sanity.py (tek seferlik)
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from src.models.vision_connector import VisionConnector

ckpt="ckpts/mini-vlm/epoch0"
base="lmsys/gpt-oss-20b-bf16"

tok = AutoTokenizer.from_pretrained(ckpt, use_fast=True, trust_remote_code=True)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
llm = AutoModelForCausalLM.from_pretrained(base, trust_remote_code=True, device_map="auto",
                                           quantization_config=bnb, torch_dtype=None, low_cpu_mem_usage=True)
llm.resize_token_embeddings(len(tok))
llm = PeftModel.from_pretrained(llm, ckpt)
llm.eval()

hidden=llm.config.hidden_size
conn=VisionConnector(tv_tokens=32, llm_hidden=hidden)
conn.load_state_dict(torch.load(f"{ckpt}/vision_connector.pt", map_location="cpu", weights_only=True))
conn=conn.to(next(llm.parameters()).device)
img=Image.open("inference_data/img001.jpg").convert("RGB")

with torch.inference_mode():
    wte = llm.get_input_embeddings()
    ids = tok("User: <image>\nAssistant:", return_tensors="pt").input_ids.to(next(llm.parameters()).device)
    te = wte(ids)
    ie = conn([img]).to(device=te.device, dtype=te.dtype)

print("text_embeds norm:", te.norm().item())
print("img_embeds  norm:", ie.norm().item(), "shape:", tuple(ie.shape))
