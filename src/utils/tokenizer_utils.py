# src/utils/tokenizer_utils.py
from transformers import AutoTokenizer
SPECIAL_TOKENS = {"additional_special_tokens": ["<image>"]}

def get_tokenizer(name):
    tok = AutoTokenizer.from_pretrained(name, use_fast=True, trust_remote_code=True)
    tok.add_special_tokens(SPECIAL_TOKENS)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok
