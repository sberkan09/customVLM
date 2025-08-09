import os, torch, gradio as gr
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from src.models.vision_connector import VisionConnector
from src.utils.tokenizer_utils import get_tokenizer

BASE_LLM = os.environ.get("BASE_LLM", "lmsys/gpt-oss-20b-bf16")
CKPT_DIR = "ckpts/mini-vlm/epoch0"
CONNECTOR_PATH = f"{CKPT_DIR}/vision_connector.pt"

def load():
    tok = get_tokenizer(BASE_LLM)
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16
    )
    llm = AutoModelForCausalLM.from_pretrained(
        BASE_LLM, trust_remote_code=True, device_map="auto",
        low_cpu_mem_usage=True, quantization_config=bnb_cfg, torch_dtype=None
    )
    hidden = llm.config.hidden_size
    connector = VisionConnector(tv_tokens=32, llm_hidden=hidden)
    connector.load_state_dict(torch.load(CONNECTOR_PATH, map_location="cpu"))
    connector = connector.to(next(llm.parameters()).device)
    llm.eval()
    return llm, tok, connector

llm, tok, connector = load()

@torch.inference_mode()
def infer(img: Image.Image):
    prompt = "User: <image>\nAssistant:"
    input_ids = tok(prompt, return_tensors="pt").input_ids.to(next(llm.parameters()).device)
    wte = llm.get_input_embeddings()
    text_embeds = wte(input_ids)
    img_embeds = connector([img])
    inputs_embeds = torch.cat([text_embeds, img_embeds], dim=1)
    out = llm.generate(
        inputs_embeds=inputs_embeds, max_new_tokens=64,
        do_sample=True, top_p=0.9, temperature=0.7,
        eos_token_id=tok.eos_token_id
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    return text.split("Assistant:")[-1].strip()

demo = gr.Interface(fn=infer, inputs=gr.Image(type="pil"), outputs="text", title="Mini VLM")
if __name__ == "__main__":
    demo.launch(share=True)
