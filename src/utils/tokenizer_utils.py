# src/utils/tokenizer_utils.py
from transformers import AutoTokenizer

SPECIAL_TOKENS = {"additional_special_tokens": ["<image>"]}

def get_tokenizer(name="mistralai/Mistral-7B-Instruct-v0.3"):
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    tok.add_special_tokens(SPECIAL_TOKENS)
    return tok
