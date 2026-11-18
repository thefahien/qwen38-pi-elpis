#!/usr/bin/env python3
"""Spot-check the merge: for a few adapter targets, merged == base + scale * B @ A (within bf16 rounding),
and every untouched tensor is bit-identical. usage: verify_merge.py BASE MERGED ADAPTER CFG"""
import json, os, random, sys
import torch
from safetensors import safe_open
base, merged, adp, cfgp = sys.argv[1:5]
cfg = json.load(open(cfgp)); scale = cfg["lora_alpha"] / cfg["r"]
wm = json.load(open(os.path.join(base, "model.safetensors.index.json")))["weight_map"]
wm2 = json.load(open(os.path.join(merged, "model.safetensors.index.json")))["weight_map"]
assert wm == wm2, "index differs"
with safe_open(adp, "pt") as f:
    keys = sorted(f.keys())
    targets = sorted({k[len("base_model.model."):-len(".weight")].rsplit(".", 1)[0] + ".weight" for k in keys})
    random.seed(7)
    pick = random.sample(targets, 6) + [t for t in targets if "layers.63.mlp.down_proj" in t]
    lora = {t: (f.get_tensor("base_model.model." + t[:-7] + ".lora_A.weight").float(),
                f.get_tensor("base_model.model." + t[:-7] + ".lora_B.weight").float()) for t in pick}
def get(d, k):
    with safe_open(os.path.join(d, wm[k]), "pt") as f:
        return f.get_tensor(k)
for t in pick:
    A, B = lora[t]
    w0, w1 = get(base, t), get(merged, t)
    expect = (w0.float() + scale * (B @ A)).to(torch.float16)
    delta = (w1.float() - w0.float()).abs().mean().item()
    print(f"{t}: identical_to_expected={torch.equal(w1, expect)} mean|delta|={delta:.2e}")
    assert torch.equal(w1, expect), t
others = [k for k in wm if k not in targets]
random.seed(3)
for k in random.sample(others, 8) + [k for k in others if "visual" in k][:2] + [k for k in others if "mtp" in k][:2]:
    assert torch.equal(get(base, k), get(merged, k)), k
print(f"VERIFY OK: {len(pick)} merged tensors exact, 12 untouched tensors bit-identical, {len(targets)} targets total")
