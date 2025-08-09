# src/models/mm_model.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model

class MiniVLM(nn.Module):
    """
    LLM (CausalLM) + VisionConnector
    - Tokenizer'ın "<image>" özel tokenını arar.
    - inputs_embeds oluştururken "<image>" yerine T_v görsel embeddingleri koyar.
    - Loss: sadece asistan cevabı üzerinde (label=-100 maskesi ile) hesaplanır.
    """
    def __init__(self, llm_name, tokenizer, connector, lora_r=16, lora_alpha=32, lora_dropout=0.05):
        super().__init__()
        self.tokenizer = tokenizer
        self.connector = connector
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, device_map="auto"
        )
        # tokenizer'a eklediğimiz özel tokenları LLM embedding boyutuna genişlet
        self.llm.resize_token_embeddings(len(self.tokenizer))
        # LoRA
        lconf = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=["q_proj","k_proj","v_proj","o_proj"]
        )
        self.llm = get_peft_model(self.llm, lconf)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
        self.hidden_size = self.llm.config.hidden_size

    def build_inputs(self, input_ids, images):
        """
        input_ids: (B, L) -> metin içinde "<image>" tek tokenı var varsayımı
        images: list[PIL]
        Dönen:
          inputs_embeds: (B, L - 1 + T_v, H)
          labels: (B, L - 1 + T_v)
        """
        B, L = input_ids.shape
        device = input_ids.device
        # Görsel embedleri hazırla
        image_embeds = self.connector(images).to(device)  # (B,T_v,H)
        T_v = image_embeds.size(1)
        # Text embeddingleri çek
        wte = self.llm.get_input_embeddings()
        text_embeds = wte(input_ids)  # (B,L,H)

        # "<image>" pozisyonunu bul, yerine T_v görsel embed koy
        new_embeds = []
        new_labels = []
        for b in range(B):
            ids = input_ids[b]
            lbl = ids.clone()
            # "<image>" index
            where = (ids == self.image_token_id).nonzero(as_tuple=True)[0]
            assert len(where) == 1, "Her örnekte tam 1 <image> olmalı"
            idx = where.item()
            # Öncesi
            new_embeds.append(text_embeds[b, :idx, :])
            new_labels.append(torch.full((idx,), -100, dtype=torch.long, device=device))  # prompt'a loss yazma
            # Görsel
            new_embeds.append(image_embeds[b:b+1, :, :].squeeze(0))
            new_labels.append(torch.full((T_v,), -100, dtype=torch.long, device=device))  # görsel tokenlara loss yok
            # Sonrası
            new_embeds.append(text_embeds[b, idx+1:, :])
            new_labels.append(lbl[idx+1:])
        inputs_embeds = torch.cat(new_embeds, dim=0).view(B, -1, self.hidden_size)
        labels = torch.cat(new_labels, dim=0).view(B, -1)
        return inputs_embeds, labels

    def forward(self, input_ids, images, labels=None):
        inputs_embeds, labels_masked = self.build_inputs(input_ids, images)
        out = self.llm(inputs_embeds=inputs_embeds, labels=labels_masked)
        return out
