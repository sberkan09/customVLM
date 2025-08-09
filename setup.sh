#!/bin/bash
set -e

# ================================
# CONFIG
# ================================
PYTHON_VERSION=3.11
VENV_NAME=.venv
BASE_LLM="lmsys/gpt-oss-20b-bf16"
HF_HOME_DIR="$PWD/.cache/huggingface"

# ================================
# 1) Python ortamı kur
# ================================
echo "[INFO] Python venv oluşturuluyor..."
python$PYTHON_VERSION -m venv $VENV_NAME
source $VENV_NAME/bin/activate

# ================================
# 2) Gerekli paketler
# ================================
echo "[INFO] Paketler yükleniyor..."
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# ================================
# 3) Hugging Face ayarları
# ================================
echo "[INFO] Hugging Face cache dizini ayarlanıyor..."
mkdir -p "$HF_HOME_DIR"
export HF_HOME="$HF_HOME_DIR"

# ================================
# 4) Gerekli dizinleri oluştur
# ================================
echo "[INFO] Gerekli klasörler oluşturuluyor..."
mkdir -p data/coco-mini
mkdir -p ckpts/mini-vlm
mkdir -p inference_data

# ================================
# 5) Model önceden indir (opsiyonel)
# ================================
echo "[INFO] Model indiriliyor: $BASE_LLM"
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('$BASE_LLM', trust_remote_code=True)"

# ================================
# 6) Örnek kullanım bilgisi
# ================================
echo ""
echo "Kurulum tamamlandı!"
echo ""
echo "[Eğitim başlatmak için]:"
echo "env TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=$HF_HOME_DIR accelerate launch --mixed_precision fp16 src/train_mm.py \\"
echo "  --base_llm \"$BASE_LLM\" \\"
echo "  --train_index data/coco-mini/train.jsonl \\"
echo "  --val_index data/coco-mini/validation.jsonl \\"
echo "  --tv_tokens 32 --max_len 256 \\"
echo "  --lr_connector 1e-3 --lr_lora 5e-5 \\"
echo "  --steps_per_epoch 4000 --epochs 1"
echo ""
echo "[Inference çalıştırmak için]:"
echo "export BASE_LLM=\"$BASE_LLM\""
echo "python -m src.infer --image inference_data/img001.jpg"
