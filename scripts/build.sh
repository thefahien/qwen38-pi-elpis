#!/bin/bash
# Build the pi-adapted Qwen3.8-27B as an EXL3 model for exllamav3 / elpis-style speculative decoding.
#   1. download Qwen/Qwen3.8-27B (BF16) and the harness-hub pi LoRA at pinned revisions
#   2. merge the LoRA into the weights, stored as FP16 (see README: BF16 loses most of the delta)
#   3. verify the merge
#   4. convert to EXL3 3.5 bpw with exllamav3's converter (needs one CUDA GPU, ~64 GB system RAM)
#
# usage: bash build.sh <work_dir> [bits=3.5]
# needs: python with torch + safetensors + exllamav3 (pip install exllamav3), huggingface_hub (`hf` CLI), ~180 GB disk
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
W="${1:?work dir}"; BITS="${2:-3.5}"
PY="${PYTHON:-python3}"
BASE_REV=37561816e66b5b9153b1b64c10968c18c1e6f4a0      # Qwen/Qwen3.8-27B
LORA_REV=d99af7e949ea027b44a2889a47b044d7bf1af44e      # harness-hub/Qwen3.8-27B-pi-adapted
mkdir -p "$W"/{base,adapter,merged,exl3,tmp}

[ -f "$W/base/config.json" ] || hf download Qwen/Qwen3.8-27B --revision "$BASE_REV" --local-dir "$W/base" \
    --exclude "README.md" "LICENSE" ".gitattributes"
[ -f "$W/adapter/adapter_model.safetensors" ] || hf download harness-hub/Qwen3.8-27B-pi-adapted --revision "$LORA_REV" \
    --local-dir "$W/adapter" --include "adapter_model.safetensors" "adapter_config.json"
(cd "$W" && sha256sum -c "$HERE/../config/inputs.sha256")

if [ ! -f "$W/merged/merge.done" ]; then
  "$PY" "$HERE/merge_lora.py" "$W/base" "$W/adapter/adapter_model.safetensors" "$W/adapter/adapter_config.json" "$W/merged"
  "$PY" "$HERE/verify_merge.py" "$W/base" "$W/merged" "$W/adapter/adapter_model.safetensors" "$W/adapter/adapter_config.json"
  touch "$W/merged/merge.done"
fi

OUT="$W/exl3/qwen38-pi-exl3-$BITS"
if [ ! -f "$OUT/quantization_config.json" ]; then
  if [ -f "$W/tmp/args.json" ]; then
    "$PY" -m exllamav3.conversion.convert_model -w "$W/tmp" -r
  else
    "$PY" -m exllamav3.conversion.convert_model -i "$W/merged" -o "$OUT" -w "$W/tmp" -b "$BITS" -hb 6 -cb mul1 -d 0
  fi
fi
for f in chat_template.jinja tokenizer.json tokenizer_config.json vocab.json merges.txt generation_config.json \
         preprocessor_config.json video_preprocessor_config.json; do
  [ -f "$OUT/$f" ] || cp "$W/merged/$f" "$OUT/" 2>/dev/null || true
done
echo "BUILD DONE: $OUT"
