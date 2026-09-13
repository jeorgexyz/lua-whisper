#!/usr/bin/env python3
"""Check audio.lua's log-mel against WhisperFeatureExtractor.

    python tools/check_mel.py

The model was trained on HF's exact numbers, so this is not a "close
enough" comparison -- a subtly wrong spectrogram gives subtly worse
transcripts and nothing downstream points at the cause.

Signals are synthetic and seeded, so this needs no audio file and is
reproducible anywhere. They are chosen to exercise the parts of the
pipeline that are easy to get wrong:

  silence        the mel floor and the (max - 8) clamp, where every value
                 is the clamp and any offset shows up immediately
  dc             energy only in bin 0
  sine 440       a single mel band lit, easy to eyeball
  chirp          sweeps every band, catches a frequency-axis flip
  noise          everything at once
  short clip     shorter than 30s, so the pad-to-window path runs

Samples cross to Lua as raw float64 rather than text: 480,000 numbers per
signal is 10 MB of ASCII and the formatting alone would dominate the run.
"""

import os
import struct
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LUA = os.environ.get("LUA", "lua54")
SR = 16000
FULL = SR * 30

DRIVER = r'''
package.path = "./?.lua;" .. package.path
local audio = require("audio")

local inp, outp, mel_path, pad = ...
local fh = assert(io.open(inp, "rb"))
local blob = fh:read("a"); fh:close()

local n = #blob // 8
local samples = {}
for i = 1, n do samples[i] = string.unpack("<d", blob, (i - 1) * 8 + 1) end
if pad == "1" then samples = audio.pad_or_trim(samples) end

local filters = assert(audio.load_mel_filters(mel_path))
local mel, frames = audio.log_mel(samples, filters)

local out = assert(io.open(outp, "wb"))
out:write(string.pack("<i4i4", filters.n_mels, frames))
for m = 1, filters.n_mels do
  for t = 1, frames do out:write(string.pack("<f", mel[m][t])) end
end
out:close()
'''


def signals():
    rng = np.random.default_rng(0)
    t = np.arange(FULL) / SR
    short_n = SR * 3
    ts = np.arange(short_n) / SR
    return {
        "silence":      np.zeros(FULL),
        "dc":           np.full(FULL, 0.5),
        "sine 440":     0.5 * np.sin(2 * np.pi * 440 * t),
        "chirp":        0.5 * np.sin(2 * np.pi * (50 + 3000 * t / 30) * t),
        "noise":        0.1 * rng.standard_normal(FULL),
        "short 3s":     0.4 * np.sin(2 * np.pi * 220 * ts),
    }


def main():
    from transformers import WhisperFeatureExtractor
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny.en")

    mel_bin = os.path.join(ROOT, "mel80.bin")
    if not os.path.exists(mel_bin):
        sys.exit("mel80.bin missing -- run: python tools/export_mel.py")

    driver = os.path.join(ROOT, "tools", "_mel_dump.lua")
    with open(driver, "w", encoding="utf-8") as fh:
        fh.write(DRIVER)

    inp = os.path.join(ROOT, "tools", "_mel_in.bin")
    outp = os.path.join(ROOT, "tools", "_mel_out.bin")

    worst, worst_name, failed = 0.0, "", 0
    print("%-14s %10s %12s %12s" % ("signal", "frames", "max |diff|", "mean |diff|"))
    print("%-14s %10s %12s %12s" % ("-" * 14, "-" * 10, "-" * 12, "-" * 12))

    try:
        for name, sig in signals().items():
            want = fe(sig, sampling_rate=SR,
                      return_tensors="np").input_features[0]      # (80, 3000)

            with open(inp, "wb") as fh:
                fh.write(struct.pack("<%dd" % len(sig), *sig.astype(np.float64)))

            proc = subprocess.run(
                [LUA, "tools/_mel_dump.lua", "tools/_mel_in.bin",
                 "tools/_mel_out.bin", "mel80.bin", "1"],
                cwd=ROOT, capture_output=True, text=True)
            if proc.returncode != 0:
                print("%-14s  LUA FAILED: %s" % (name, proc.stderr.strip()[:160]))
                failed += 1
                continue

            with open(outp, "rb") as fh:
                blob = fh.read()
            n_mels, frames = struct.unpack("<2i", blob[:8])
            have = np.frombuffer(blob, dtype="<f4", count=n_mels * frames,
                                 offset=8).reshape(n_mels, frames)

            if have.shape != want.shape:
                print("%-14s  SHAPE %s vs %s" % (name, have.shape, want.shape))
                failed += 1
                continue

            diff = np.abs(have - want)
            mx, mean = diff.max(), diff.mean()
            if mx > worst:
                worst, worst_name = mx, name
            # float32 storage plus a log10; 1e-4 is generous for noise and
            # far tighter than anything that would change a transcript.
            bad = mx > 1e-4
            failed += bad
            print("%-14s %10d %12.3e %12.3e%s"
                  % (name, frames, mx, mean, "   FAIL" if bad else ""))
    finally:
        for p in (driver, inp, outp):
            if os.path.exists(p):
                os.remove(p)

    print()
    print("worst max |diff| %.3e (%s)" % (worst, worst_name))
    if failed:
        sys.exit("%d signal(s) disagree with WhisperFeatureExtractor" % failed)
    print("audio.lua matches WhisperFeatureExtractor")


if __name__ == "__main__":
    main()
