#!/usr/bin/env python3
"""Merge harness-hub/Qwen3.8-27B-pi-adapted (PEFT LoRA, r=32, alpha=64) into Qwen/Qwen3.8-27B BF16, shard by shard.

W_merged = W + (alpha / r) * (B @ A), computed in fp32, stored as fp16 (see the note at the store). Every tensor the adapter does not touch
(vision tower, MTP head, norms, linear-attention projections, embeddings) is copied unchanged, so the output has the
exact base layout and the DFlash2 draft / exllamav3 arch loader see the same model.

usage: merge_lora.py BASE_DIR ADAPTER.safetensors ADAPTER_CONFIG.json OUT_DIR
"""
import json, os, shutil, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

base, adp_path, cfg_path, out = sys.argv[1:5]
cfg = json.load(open(cfg_path))
r, alpha = cfg["r"], cfg["lora_alpha"]
assert not cfg.get("use_rslora") and not cfg.get("use_dora"), "plain LoRA only"
scale = alpha / r
os.makedirs(out, exist_ok=True)

# adapter key: base_model.model.<base key without .weight>.lora_{A,B}.weight
lora = {}
with safe_open(adp_path, "pt") as f:
    for k in f.keys():
        assert k.startswith("base_model.model.") and k.endswith(".weight"), k
        mod, ab = k[len("base_model.model."):-len(".weight")].rsplit(".", 1)
        lora.setdefault(mod + ".weight", {})[ab] = f.get_tensor(k).float()
print(f"adapter: {len(lora)} target tensors, r={r} alpha={alpha} scale={scale}", flush=True)

idx = json.load(open(os.path.join(base, "model.safetensors.index.json")))
missing = [k for k in lora if k not in idx["weight_map"]]
assert not missing, f"adapter targets not in base: {missing[:5]}"

done = set()
shards = sorted(set(idx["weight_map"].values()))
for sh in shards:
    t0 = time.time()
    tensors, meta = {}, None
    with safe_open(os.path.join(base, sh), "pt") as f:
        meta = f.metadata()
        for k in f.keys():
            w = f.get_tensor(k)
            if k in lora:
                A, B = lora[k]["lora_A"], lora[k]["lora_B"]
                assert B.shape[0] == w.shape[0] and A.shape[1] == w.shape[1], (k, w.shape, A.shape, B.shape)
                merged = w.float() + scale * (B @ A)
                # FP16, not the base's BF16: the LoRA delta is ~1e-3 of |W|, near bf16's rounding step, and a bf16
                # store kept only 58-85% of it (measured, delta_retention.py). fp16 keeps ~100%; |W| < 1 so no
                # overflow, and exllamav3 converts bf16 -> fp16 on load anyway.
                assert merged.abs().max().item() < 60000, k
                tensors[k] = merged.to(torch.float16).contiguous()
                done.add(k)
            else:
                tensors[k] = w
    save_file(tensors, os.path.join(out, sh), metadata=meta or {"format": "pt"})
    print(f"{sh}: {len(tensors)} tensors, {sum(1 for k in tensors if k in lora)} merged, {time.time()-t0:.0f}s", flush=True)
    del tensors

assert done == set(lora), f"not merged: {sorted(set(lora) - done)[:5]}"
for fn in os.listdir(base):
    if fn.endswith(".safetensors") or fn.endswith(".part") or fn in ("dl.done", "dl.fail"):
        continue
    shutil.copy2(os.path.join(base, fn), os.path.join(out, fn))
print(f"MERGE OK: {len(done)} tensors merged into {len(shards)} shards", flush=True)
