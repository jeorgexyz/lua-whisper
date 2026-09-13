#!/usr/bin/env python3
"""Export Whisper's mel filterbank for audio.lua.

    python tools/export_mel.py            # writes mel80.bin

The filterbank is a fixed 201x80 matrix: 201 FFT power bins in, 80 mel
bands out. It depends only on the sample rate and n_fft, never on the audio,
so it is a constant and it gets exported as one.

The alternative is reimplementing the mel scale in Lua -- hertz-to-mel,
triangular filter construction, and the Slaney-style area normalisation HF
uses. That is roughly eighty lines whose only job is to reproduce a table
that already exists, and every one of those lines is a place to introduce a
silent half-bin offset. Exporting the constant is both shorter and not
wrong.

Format (same shape as every other file in this project line):
    int32 n_bins, int32 n_mels, then n_bins*n_mels float32, row-major
"""

import argparse
import os
import struct
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-tiny.en")
    ap.add_argument("--output", default="mel80.bin")
    args = ap.parse_args()

    from transformers import WhisperFeatureExtractor

    fe = WhisperFeatureExtractor.from_pretrained(args.model)
    mel = np.asarray(fe.mel_filters, dtype=np.float32)

    n_bins, n_mels = mel.shape
    expected_bins = fe.n_fft // 2 + 1
    if n_bins != expected_bins or n_mels != fe.feature_size:
        sys.exit("unexpected filterbank shape %s (want %d x %d)"
                 % (mel.shape, expected_bins, fe.feature_size))

    with open(args.output, "wb") as fh:
        fh.write(struct.pack("<2i", n_bins, n_mels))
        fh.write(np.ascontiguousarray(mel).tobytes())

    print("wrote %s  (%d bins x %d mels, %.1f KB)"
          % (args.output, n_bins, n_mels,
             os.path.getsize(args.output) / 1024))
    print("front end: sr=%d n_fft=%d hop=%d mels=%d chunk=%ds"
          % (fe.sampling_rate, fe.n_fft, fe.hop_length,
             fe.feature_size, fe.chunk_length))
    # Dither would make the reference non-deterministic and the parity check
    # meaningless, so it is worth confirming rather than assuming.
    print("dither: %r  (must be 0.0 for a reproducible reference)"
          % getattr(fe, "dither", 0.0))


if __name__ == "__main__":
    main()
