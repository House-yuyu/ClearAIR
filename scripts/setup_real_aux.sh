#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
HF_CLI="${HF_CLI:-hf}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
DACLIP_COMMIT="c2cc14146faac85680b4cb39ece4ef9e2a7c7c24"
SAM2_COMMIT="2b90b9f5ceec907a1c18123530e92e794ad901a4"

cd "$ROOT"

"$PYTHON" -m pip install -e '.[real-aux]'
"$PYTHON" - <<'PY'
from packaging.version import Version
import torch

if Version(torch.__version__.split("+")[0]) < Version("2.5.1"):
    raise SystemExit(f"PyTorch >=2.5.1 is required for the real stack; found {torch.__version__}")
print(f"Using torch {torch.__version__}; CUDA available: {torch.cuda.is_available()}")
PY

if [ ! -d third_party/daclip-uir/.git ]; then
    mkdir -p third_party
    git clone --depth 1 https://github.com/Algolzw/daclip-uir.git third_party/daclip-uir
fi
git -C third_party/daclip-uir fetch --depth 1 origin "$DACLIP_COMMIT"
git -C third_party/daclip-uir checkout --detach "$DACLIP_COMMIT"
if grep -Fq 'for _ in range(self.layers)]).cuda()' \
    third_party/daclip-uir/da-clip/src/open_clip/transformer.py; then
    patch -d third_party/daclip-uir -p1 < scripts/patches/daclip-device-agnostic.patch
fi
"$PYTHON" -m pip install --no-deps -e third_party/daclip-uir/da-clip

if [ ! -d third_party/sam2/.git ]; then
    git clone --depth 1 https://github.com/facebookresearch/sam2.git third_party/sam2
fi
git -C third_party/sam2 fetch --depth 1 origin "$SAM2_COMMIT"
git -C third_party/sam2 checkout --detach "$SAM2_COMMIT"
SAM2_BUILD_CUDA=0 "$PYTHON" -m pip install --no-deps --no-build-isolation -e third_party/sam2

HF_ENDPOINT="$HF_ENDPOINT" "$HF_CLI" download facebook/sam2.1-hiera-tiny \
    sam2.1_hiera_tiny.pt --local-dir pretrained/sam2 --max-workers 1
HF_ENDPOINT="$HF_ENDPOINT" "$HF_CLI" download weblzw/daclip-uir-ViT-B-32-irsde \
    daclip_ViT-B-32.pt --local-dir pretrained/daclip --max-workers 1
HF_ENDPOINT="$HF_ENDPOINT" "$HF_CLI" download zhiyuanyou/DeQA-Score-Mix3 \
    --include '*.safetensors' '*.json' '*.py' 'tokenizer.model' \
    --exclude '*.bin' --local-dir pretrained/DeQA-Score-Mix3 --max-workers 2
if grep -Fq 'self.rotary_emb(value_states, seq_len=kv_seq_len)' \
    pretrained/DeQA-Score-Mix3/modeling_llama2.py; then
    patch -d pretrained/DeQA-Score-Mix3 -p1 < scripts/patches/deqa-transformers-446.patch
fi

echo "Real auxiliary dependencies and checkpoints are ready under $ROOT/pretrained."
