#!/usr/bin/env python3
"""How much of the LoRA delta survives rounding into the base dtype? Compares ideal fp32 delta with the stored one,
and what an fp16 merge would keep. usage: delta_retention.py BASE MERGED ADAPTER CFG"""
import json, os, sys
import torch
from safetensors import safe_open
base, merged, adp, cfgp = sys.argv[1:5]
cfg = json.load(open(cfgp)); scale = cfg["lora_alpha"] / cfg["r"]
wm = json.load(open(os.path.join(base, "model.safetensors.index.json")))["weight_map"]
def get(d, k):
    with safe_open(os.path.join(d, wm[k]), "pt") as f:
        return f.get_tensor(k)
picks = ["model.language_model.layers.%d.%s.weight" % (l, m) for l in (3, 31, 63)
         for m in ("mlp.down_proj", "mlp.up_proj", "self_attn.o_proj")]
with safe_open(adp, "pt") as f:
    for t in picks:
        A = f.get_tensor("base_model.model." + t[:-7] + ".lora_A.weight").float()
        B = f.get_tensor("base_model.model." + t[:-7] + ".lora_B.weight").float()
        ideal = scale * (B @ A)
        w0 = get(base, t).float()
        bf = get(merged, t).float() - w0
        fp16 = (w0 + ideal).half().float() - w0
        def ret(x):
            return (x * ideal).sum().item() / (ideal * ideal).sum().item(), torch.nn.functional.cosine_similarity(x.flatten(), ideal.flatten(), dim=0).item(), (x != 0).float().mean().item()
        rb, rf = ret(bf), ret(fp16)
        print(f"{t.split('model.language_model.')[1]:28s} bf16: proj {rb[0]:.3f} cos {rb[1]:.3f} nonzero {rb[2]:.3f} | fp16: proj {rf[0]:.3f} cos {rf[1]:.3f} | |W| {w0.abs().mean():.2e} |d| {ideal.abs().mean():.2e}")
