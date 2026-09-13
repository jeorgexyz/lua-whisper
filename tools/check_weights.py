#!/usr/bin/env python3
"""Check that weights.lua reads the same numbers PyTorch holds.

    python tools/check_weights.py

weights.lua already verifies that it consumes the file exactly, which
catches a layout that is the wrong SIZE. It cannot catch a layout that is
the wrong ORDER -- swap two tensors of equal shape and the byte count is
still perfect.

So this compares actual values, and picks the probes deliberately:

  enc.conv1.weight      the first tensor; catches a header-size error
  enc.0.k_proj.weight   immediately after the bias that does not exist --
                        the first tensor a wrong bias layout would shift
  enc.pos               large, and before the layer stack
  dec.tok_emb           19.9M floats; catches encoder/decoder swaps
  dec.3.final_ln.bias   the LAST tensor in the file. If this matches,
                        everything before it is aligned.
"""

import os
import struct
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LUA = os.environ.get("LUA", "lua54")
CKPT = os.environ.get("CKPT", "whisper-tiny.en.lwb")

DRIVER = r'''
package.path = "./?.lua;" .. package.path
local weights = require("weights")
local w, err = weights.load(...)
if not w then io.stderr:write(err .. "\n") os.exit(1) end

local function probe(name, t)
  local parts = {}
  for _, i in ipairs({1, 2, 3, #t - 2, #t - 1, #t}) do
    parts[#parts + 1] = string.format("%.9g", t[i])
  end
  io.write(name, "\t", #t, "\t", table.concat(parts, " "), "\n")
end

probe("enc.conv1.weight", w.enc.conv1_w)
probe("enc.0.k_proj.weight", w.enc.layers[1].attn.kw)
probe("enc.pos", w.enc.pos)
probe("dec.tok_emb", w.dec.tok_emb)
probe("dec.last.final_ln.bias", w.dec.layers[#w.dec.layers].final_ln.b)
io.write("kb_absent\t", tostring(w.enc.layers[1].attn.kb == nil), "\n")
io.write("cross_kb_absent\t", tostring(w.dec.layers[1].cross.kb == nil), "\n")
'''


def main():
    ckpt = os.path.join(ROOT, CKPT)
    if not os.path.exists(ckpt):
        sys.exit("%s missing -- run: python tools/export_whisper.py" % CKPT)

    from transformers import WhisperForConditionalGeneration
    m = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
    enc, dec = m.model.encoder, m.model.decoder

    want = {
        "enc.conv1.weight": enc.conv1.weight,
        "enc.0.k_proj.weight": enc.layers[0].self_attn.k_proj.weight,
        "enc.pos": enc.embed_positions.weight,
        "dec.tok_emb": dec.embed_tokens.weight,
        "dec.last.final_ln.bias": dec.layers[-1].final_layer_norm.bias,
    }

    driver = os.path.join(ROOT, "tools", "_w_dump.lua")
    with open(driver, "w", encoding="utf-8") as fh:
        fh.write(DRIVER)
    try:
        proc = subprocess.run([LUA, "tools/_w_dump.lua", CKPT], cwd=ROOT,
                              capture_output=True, text=True)
    finally:
        os.remove(driver)
    if proc.returncode != 0:
        sys.exit("lua failed:\n" + proc.stderr[:800])

    flags, got = {}, {}
    for line in proc.stdout.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) == 2:
            flags[parts[0]] = parts[1]
        else:
            got[parts[0]] = (int(parts[1]), [float(x) for x in parts[2].split()])

    failed = 0
    print("%-24s %10s %12s" % ("tensor", "elements", "max |diff|"))
    print("%-24s %10s %12s" % ("-" * 24, "-" * 10, "-" * 12))
    for name, tensor in want.items():
        a = tensor.detach().float().cpu().numpy().ravel()
        n, probes = got[name]
        if n != a.size:
            print("%-24s  SIZE %d vs %d   FAIL" % (name, n, a.size))
            failed += 1
            continue
        ref = [a[0], a[1], a[2], a[-3], a[-2], a[-1]]
        diff = max(abs(x - y) for x, y in zip(probes, ref))
        bad = diff > 1e-6
        failed += bad
        print("%-24s %10d %12.3e%s" % (name, n, diff, "   FAIL" if bad else ""))

    print()
    for key, label in (("kb_absent", "encoder self-attn k_proj has no bias"),
                       ("cross_kb_absent", "decoder cross-attn k_proj has no bias")):
        ok = flags.get(key) == "true"
        failed += not ok
        print("%-40s %s" % (label, "yes" if ok else "NO   FAIL"))

    print()
    if failed:
        sys.exit("%d check(s) failed" % failed)
    print("weights.lua reads what PyTorch holds")


if __name__ == "__main__":
    main()
