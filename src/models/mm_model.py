import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

class MiniVLM(nn.Module):
    def __init__(self, base_llm_name, tokenizer, connector,
                 lora_r=16, lora_alpha=32, lora_dropout=0.05, dtype=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.connector = connector

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,  # A40 için güvenli
        )

        self.llm = AutoModelForCausalLM.from_pretrained(
            base_llm_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=None,               # 4-bit yüklemede None bırak
            low_cpu_mem_usage=True,
            quantization_config=bnb_cfg,
        )

        # ---- Minimal QLoRA hazırlığı (PEFT prepare() YOK) ----
        # Gradient checkpointing (reentrant=False → bellek piki düşer)
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.llm.config.use_cache = False

        # Bazı HF modellerinde mevcut (embedding grad’ı için):
        if hasattr(self.llm, "enable_input_require_grads"):
            self.llm.enable_input_require_grads()

        # <image> özel token eklendiği için embedding’i yeniden boyutla
        self.llm.resize_token_embeddings(len(self.tokenizer))

        # LoRA sadece attention projeksiyonlarına
        lconf = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        self.llm = get_peft_model(self.llm, lconf)

        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
        self.hidden_size = self.llm.config.hidden_size

    def build_inputs(self, input_ids, images):
        B, L = input_ids.shape
        device = input_ids.device
        image_embeds = self.connector(images).to(device)  # (B,T_v,H)
        T_v = image_embeds.size(1)
        wte = self.llm.get_input_embeddings()
        text_embeds = wte(input_ids)
        image_embeds = image_embeds.to(dtype=text_embeds.dtype)

        new_embeds, new_labels = [], []
        for b in range(B):
            ids = input_ids[b]
            lbl = ids.clone()
            where = (ids == self.image_token_id).nonzero(as_tuple=True)[0]
            assert len(where) == 1, "Her örnekte tam 1 <image> olmalı"
            idx = where.item()
            new_embeds.append(text_embeds[b, :idx, :])
            new_labels.append(torch.full((idx,), -100, dtype=torch.long, device=device))
            new_embeds.append(image_embeds[b:b+1, :, :].squeeze(0))
            new_labels.append(torch.full((T_v,), -100, dtype=torch.long, device=device))
            new_embeds.append(text_embeds[b, idx+1:, :])
            new_labels.append(lbl[idx+1:])
        inputs_embeds = torch.cat(new_embeds, dim=0).view(B, -1, self.hidden_size)
        labels = torch.cat(new_labels, dim=0).view(B, -1)
        return inputs_embeds, labels

    def forward(self, input_ids, images, labels=None):
        inputs_embeds, labels_masked = self.build_inputs(input_ids, images)
        out = self.llm(inputs_embeds=inputs_embeds, labels=labels_masked)
        return out
