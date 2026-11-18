#!/usr/bin/env python3
"""Privacy scan for a tree about to go public: private IPs, tailnet, .internal names, home paths, fleet host/CT names,
private emails, token-like strings, ELF binaries. Extra literal patterns come from a git-ignored .privacy-patterns file.
usage: privacy_scan.py <dir>   (exit 1 on any hit)"""
import os, re, sys
root = sys.argv[1]
pats = [r"\b10\.\d+\.\d+\.\d+\b", r"\b192\.168\.\d+\.\d+\b", r"\b172\.(1[6-9]|2\d|3[01])\.\d+\.\d+\b",
        r"\b100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d+\.\d+\b", r"\.internal\b", r"/home/[a-z]", r"/Users/[a-z]",
        r"\bsk-[A-Za-z0-9]{8,}", r"\bhf_[A-Za-z0-9]{20,}", r"ghp_[A-Za-z0-9]{20,}",
        r"\b[A-Fa-f0-9]{40,48}\b(?<!sha)"]
extra = os.path.join(root, ".privacy-patterns")
if os.path.exists(extra):
    pats += [re.escape(l.strip()) for l in open(extra) if l.strip() and not l.startswith("#")]
rx = re.compile("|".join(f"(?:{p})" for p in pats), re.I)
allow = re.compile(r"^[0-9a-f]{64}  |revision [0-9a-f]{40}|_REV=[0-9a-f]{40}|@360c902|360c9025188810a0ac9d434c837dc3cb6b8e39cd")
hits = 0
for d, dirs, files in os.walk(root):
    dirs[:] = [x for x in dirs if x not in (".git", "__pycache__")]
    for f in files:
        if f == ".privacy-patterns" or os.path.abspath(os.path.join(d, f)) == os.path.abspath(__file__):
            continue
        p = os.path.join(d, f)
        b = open(p, "rb").read()
        if b[:4] == b"\x7fELF":
            print(f"BINARY {p}"); hits += 1; continue
        for i, line in enumerate(b.decode("utf-8", "replace").splitlines(), 1):
            if allow.search(line):
                continue
            for m in rx.finditer(line):
                print(f"{os.path.relpath(p, root)}:{i}: {m.group(0)!r}  ...{line[max(0,m.start()-40):m.end()+40]!r}"[:260])
                hits += 1
print(f"{hits} hits")
sys.exit(1 if hits else 0)
