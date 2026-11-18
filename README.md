# qwen38-pi-elpis

Run [harness-hub/Qwen3.8-27B-pi-adapted](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted), a LoRA that RL-tunes
Qwen3.8-27B for the [pi](https://github.com/earendil-works/pi) coding agent, at **~150-170 tok/s on code** on one 24 GB
GPU, NVIDIA or AMD. The LoRA is merged into the weights, converted to EXL3 3.5 bpw, and served by exllamav3 / TabbyAPI with a
DFlash2 speculative-decoding draft.

On an RTX 3090 this is **~5x the speed of the adapter's own llama.cpp route** (GGUF + `--lora`, ~34 tok/s on the same card),
for a llmbench score within about one item of it.

## Results

Greedy, thinking off, 512-token replies, one request at a time, tok/s = completion tokens / whole-request wall time
(`bench/bench_api.py`, median of 3). Draft acceptance = accepted / (accepted + rejected) draft tokens.

| | RTX 3090 (exllamav3 1.6.7, CUDA) | RX 7900 XTX (exllamav3-rocm, ROCm 7.2.4) |
|---|---|---|
| code | **168.5 tok/s** (78% accepted) | **149.7 tok/s** (76%) |
| math | 152.8 (70%) | 121.3 (59%) |
| prose | 67.3 (21%) | 64.5 (23%) |
| context, cache | 262144, Q4 | 196608, Q8 |
| needle at ~97% depth | found at 251,842 tokens | found at 188,408 tokens |
| peak VRAM in that request | 19.05 of 24 GB | 24.2 of 24.6 GB |
| prefill at that depth | 532 tok/s | 777 tok/s |

Quality (llmbench quick suites, temperature 0, thinking off; 34 items in code/reasoning/web/3D/agent, plus 6 SVG drawings):

| | avg of 34 | SVG (6) | time for all 40 |
|---|---|---|---|
| RTX 3090, llama.cpp GGUF + `--lora` (UD-Q4_K_XL base) | 0.934 | 1.00 | 1298 s |
| RTX 3090, this recipe | 0.908 | 0.979 | 270 s |
| RX 7900 XTX, this recipe | 0.882 | 0.979 | 273 s |

The suites are small (3-12 items each), so a gap under ~0.03 is one item. The 3090 and 7900 runs of this recipe differ
by exactly one item (a 3D scene that throws on the 7900). Per-suite numbers and the item-level diffs are in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Not like-for-like

- The two cards use different engines (stock exllamav3 on CUDA, the [exllamav3-rocm](https://github.com/phoenixhaxor/exllamav3-rocm)
  port on ROCm) and different cache types, because the ROCm port has no Q4 cache kernels. That is why the 7900 window is
  196608 and its VRAM is nearly full.
- The llama.cpp row runs the adapter unmerged on a Q4_K_XL GGUF base with no draft; it is the route the adapter's model
  card describes, not a tuned llama.cpp setup.
- No result here re-runs the adapter author's agent benchmarks (Terminal-Bench, SWE-bench Pro). llmbench is single-turn.

## Why merge in FP16

The LoRA delta is about 1/1000 of each weight's magnitude, close to BF16's rounding step. Storing the merged weights
in the base's BF16 kept only **58-85%** of the delta (projection of the stored change onto the ideal one,
`scripts/delta_retention.py`). FP16 keeps ~100%, the merged weights stay far inside FP16's range, and exllamav3 loads BF16 as
FP16 anyway. `merge_lora.py` therefore writes FP16.

## Build

Needs ~180 GB of disk, one CUDA GPU for the conversion, and **more than 32 GB of system RAM** (64 GB was used): on the
output-head step of this 248k-vocabulary model the converter reached 31 GB resident and was OOM-killed under a 32 GB
limit, after 2.5 h of layers. The conversion checkpoints, so `build.sh` resumes from where it stopped. Timings here:
download ~15 min, merge 13 min, conversion ~2 h 40 min on an RTX 3090.

```sh
pip install exllamav3 huggingface_hub      # exllamav3 1.6.7 was used
bash scripts/build.sh /data/qwen38-pi      # download, merge, verify, convert
bash scripts/download_draft.sh /models     # Mia-AiLab DFlash2 5.0bpw draft
cp -r /data/qwen38-pi/exl3/qwen38-pi-exl3-3.5 /models/
```

Inputs are pinned by revision and checked against `config/inputs.sha256`. `config/exl3-output.sha256` holds the hashes
of the EXL3 files this recipe produced here. The conversion uses random calibration rows, so a rebuild will not be
bit-identical.

## Serve

- **NVIDIA:** TabbyAPI on stock exllamav3, `config/rtx3090/config.yml` + `config/rtx3090/tabbyapi.service`.
- **AMD 7900 XTX:** build the ROCm port with [elpis-on-7900xtx](https://github.com/thefahien/elpis-on-7900xtx)'s
  `scripts/setup.sh` (ROCm 7.2.4, exllamav3-rocm @360c902, its two patches), then use `config/7900xtx/config.yml` and
  `config/7900xtx/tabbyapi.service`. If 24.2 GB peak is too tight for your desktop, lower `max_seq_len`/`cache_size` to
  163840.

Both configs force thinking off (`template_vars_force: {enable_thinking: false}`), as the adapter was trained.

Point pi at it with an OpenAI-completions provider in `~/.pi/agent/models.json`:

```json
{ "providers": { "local": { "baseUrl": "http://localhost:8096/v1", "api": "openai-completions",
  "apiKey": "<tabbyapi api_key>",
  "models": [ { "id": "qwen38-pi-exl3-3.5", "reasoning": false, "contextWindow": 245760, "maxTokens": 16384 } ] } } }
```

Use `contextWindow: 180000` on the 7900 config. Port 8098 there.

## Limitations

- The DFlash2 draft was trained on the base model, not the adapted one. How much acceptance that costs was not
  measured (no same-protocol run of the base model is included).
- One request at a time (`max_batch_size: 1`).
- Measured on one card of each kind. Prose gains little from the draft (21-23% acceptance).

## Credits

- Adapter: [harness-hub/Qwen3.8-27B-pi-adapted](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted) (Apache-2.0)
- Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0)
- Draft: [Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw](https://huggingface.co/Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw)
- Engines: [exllamav3](https://github.com/turboderp-org/exllamav3), [exllamav3-rocm](https://github.com/phoenixhaxor/exllamav3-rocm),
  [TabbyAPI](https://github.com/theroyallab/tabbyAPI)

MIT licence for the scripts and configs in this repo.
