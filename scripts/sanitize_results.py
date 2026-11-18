#!/usr/bin/env python3
"""Copy raw run files into results/ with private metadata removed (internal URLs, host/CT names, scratch paths,
artifact folders). Scores, prompts, outputs, timings and token counts are untouched.
usage: sanitize_results.py <src.json> <dst.json> <label> <gpu>"""
import json, re, sys
src, dst, label, gpu = sys.argv[1:5]
d = json.load(open(src))
run_label = label


def clean_str(s):
    s = re.sub(r"https?://\d+\.\d+\.\d+\.\d+:\d+", "http://localhost:8096", s)
    s = re.sub(r'/home/[^/\s"\\]+/[^\s"\\]*?/(tmp[\w]+\.py)', r"/tmp/\1", s)
    s = re.sub(r"\b\d{8}-\d{6}-[\w.\-]+", run_label, s)
    return s


def walk(o):
    if isinstance(o, dict):
        return {k: walk(v) for k, v in o.items()}
    if isinstance(o, list):
        return [walk(v) for v in o]
    if isinstance(o, str):
        return clean_str(o)
    return o


d = walk(d)
if "target" in d:
    d["target"] = {"label": label, "base_url": "http://localhost:8096", "model": d["target"].get("model"), "gpu": gpu}
if "run_id" in d:
    d["run_id"] = label
if "base" in d:
    d["base"] = "http://localhost:8096"
if "label" in d and isinstance(d["label"], str):
    d["label"] = label
if isinstance(d.get("params"), dict) and "note" in d["params"]:
    d["params"]["note"] = label
for r in d.get("results", []):
    if isinstance(r.get("detail"), dict):
        r["detail"].pop("artifact", None)
json.dump(d, open(dst, "w"), indent=1)
print("wrote", dst)
