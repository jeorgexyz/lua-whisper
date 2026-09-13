#!/usr/bin/env python3
"""Check fft.lua against numpy.fft.rfft.

    python tools/check_fft.py

Everything downstream -- mel filterbank, encoder, transcript -- is garbage
if the spectrogram is wrong, and a wrong spectrogram is the hardest thing in
the pipeline to diagnose from the far end: the output is just worse text.
So this is the first thing written and the first thing checked.

Signals are chosen to catch specific mistakes rather than to look thorough:

  impulse       flat spectrum; catches twiddle-table indexing
  DC            all energy in bin 0; catches an off-by-one in k
  sine at a bin exact bin, no leakage; catches a sign error in the
                imaginary part or a wrong 2*pi/N
  sine off-bin  leakage across bins; catches a window applied at the
                wrong point
  noise         everything at once, with a fixed seed
"""

import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LUA = os.environ.get("LUA", "lua54")

N_FFT = 400
N_BINS = N_FFT // 2 + 1


def signals():
    rng = np.random.default_rng(0)
    impulse = np.zeros(N_FFT); impulse[0] = 1.0
    late = np.zeros(N_FFT); late[7] = 1.0
    dc = np.ones(N_FFT)
    t = np.arange(N_FFT)
    on_bin = np.sin(2 * np.pi * 10 * t / N_FFT)          # exactly bin 10
    off_bin = np.sin(2 * np.pi * 10.5 * t / N_FFT)       # between bins
    noise = rng.standard_normal(N_FFT)
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * t / N_FFT)     # periodic
    return {
        "impulse": impulse,
        "shifted impulse": late,
        "dc": dc,
        "sine on bin": on_bin,
        "sine off bin": off_bin,
        "noise": noise,
        "noise * hann": noise * hann,
    }


def main():
    sigs = signals()
    payload = {k: list(map(float, v)) for k, v in sigs.items()}

    script = os.path.join(ROOT, "tools", "_fft_dump.lua")
    with open(script, "w", encoding="utf-8") as fh:
        fh.write(
            'package.path = "./?.lua;" .. package.path\n'
            'local fft = require("fft")\n'
            'local input = io.read("a")\n'
            '-- keys arrive one per line: name, then n_fft floats\n'
            'local plan = fft.plan(%d, %d)\n' % (N_FFT, N_BINS) +
            'local out = {}\n'
            'for line in input:gmatch("[^\\n]+") do\n'
            '  local name, rest = line:match("^([^\\t]+)\\t(.*)$")\n'
            '  local frame = {}\n'
            '  for v in rest:gmatch("[^ ]+") do frame[#frame+1] = tonumber(v) end\n'
            '  local re, im = fft.rfft(plan, frame)\n'
            '  local parts = {}\n'
            '  for i = 1, #re do parts[#parts+1] = string.format("%.17g %.17g", re[i], im[i]) end\n'
            '  out[#out+1] = name .. "\\t" .. table.concat(parts, " ")\n'
            'end\n'
            'io.write(table.concat(out, "\\n"))\n')

    stdin = "\n".join(
        "%s\t%s" % (k, " ".join("%.17g" % x for x in v))
        for k, v in payload.items())

    proc = subprocess.run([LUA, "tools/_fft_dump.lua"], cwd=ROOT, input=stdin,
                          capture_output=True, text=True)
    os.remove(script)
    if proc.returncode != 0:
        sys.exit("lua failed:\n" + proc.stderr[:800])

    got = {}
    for line in proc.stdout.strip().split("\n"):
        name, rest = line.split("\t", 1)
        nums = [float(x) for x in rest.split()]
        got[name] = np.array(nums[0::2]) + 1j * np.array(nums[1::2])

    worst, worst_name, failed = 0.0, "", 0
    print("%-18s %12s %12s" % ("signal", "max |diff|", "rel"))
    print("%-18s %12s %12s" % ("-" * 18, "-" * 12, "-" * 12))
    for name, sig in sigs.items():
        want = np.fft.rfft(sig)
        have = got[name]
        if have.shape != want.shape:
            print("%-18s  SHAPE %s vs %s" % (name, have.shape, want.shape))
            failed += 1
            continue
        diff = np.abs(have - want).max()
        scale = max(np.abs(want).max(), 1e-12)
        rel = diff / scale
        if rel > worst:
            worst, worst_name = rel, name
        bad = rel > 1e-10
        failed += bad
        print("%-18s %12.3e %12.3e%s" % (name, diff, rel, "   FAIL" if bad else ""))

    print()
    print("worst relative error %.3e (%s)" % (worst, worst_name))
    if failed:
        sys.exit("%d signal(s) disagree with numpy.fft.rfft" % failed)
    print("fft.lua matches numpy.fft.rfft")


if __name__ == "__main__":
    main()
