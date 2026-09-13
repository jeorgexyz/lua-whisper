#!/usr/bin/env python3
"""Dump encoder states, a token sequence, and per-position decoder logits.

    python tools/reference_decode.py

Teacher-forced rather than free-running: HF generates greedily first, and
those exact tokens are then fed back as decoder inputs so both sides score
the same sequence. A free-running comparison would diverge the moment one
argmax differed and tell you nothing about where.

The encoder states are dumped too, and validate_decoder.lua feeds the Lua
decoder those rather than its own. encoder.lua has its own parity check;
mixing the two here would leave a failure ambiguous between them.

    magic "LWD1", version, T, d, n_tokens, vocab
    encoder states   T x d      float32, time-major
    tokens           n_tokens   int32
    logits           n_tokens x vocab   float32
"""

import argparse
import struct

import numpy as np

MAGIC = b"LWD1"
SR = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-tiny.en")
    ap.add_argument("--output", default="reference_decode.ref")
    ap.add_argument("--max-new", type=int, default=12)
    args = ap.parse_args()

    import torch
    from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor

    # Same synthetic signal as tools/reference.py, so the two dumps describe
    # the same audio and can be reasoned about together.
    import importlib.util, os
    spec = importlib.util.spec_from_file_location(
        "ref", os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference.py"))
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    audio = ref.synthetic()

    fe = WhisperFeatureExtractor.from_pretrained(args.model)
    mel = fe(audio, sampling_rate=SR, return_tensors="pt").input_features

    model = WhisperForConditionalGeneration.from_pretrained(args.model).eval()

    # Teacher-force a FIXED sequence rather than whatever the model would
    # generate. Parity only needs both sides to score the same tokens, and
    # decoupling from generate() buys two things:
    #
    #   * the synthetic audio is not speech, so greedy decoding emits a
    #     single token and there is nothing to compare
    #   * generate()'s first positional argument is input_ids in
    #     transformers 5.x, so passing the mel positionally feeds a
    #     spectrogram into a token embedding
    #
    # The logits will not spell anything. They do not need to.
    from transformers import WhisperTokenizer
    tk = WhisperTokenizer.from_pretrained(args.model)
    body = tk.encode("The quick brown fox jumps over the lazy dog",
                     add_special_tokens=False)
    tokens = [50257, 50362] + body                    # sot, notimestamps
    inp = torch.tensor([tokens], dtype=torch.long)

    with torch.no_grad():
        enc = model.model.encoder(mel).last_hidden_state
        out = model(input_features=mel, decoder_input_ids=inp)
    logits = out.logits[0].float().cpu().numpy()                 # [n-1, vocab]

    enc_np = enc[0].float().cpu().numpy()
    T, d = enc_np.shape
    n_tok, vocab = logits.shape

    with open(args.output, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<5i", 1, T, d, n_tok, vocab))
        fh.write(np.ascontiguousarray(enc_np, dtype=np.float32).tobytes())
        fh.write(np.ascontiguousarray(inp[0].numpy(), dtype=np.int32).tobytes())
        fh.write(np.ascontiguousarray(logits, dtype=np.float32).tobytes())

    print("wrote %s" % args.output)
    print("  encoder %d x %d" % (T, d))
    print("  input tokens : %s" % inp[0].tolist())
    print("  logits       : %d x %d" % (n_tok, vocab))
    print("  argmax chain : %s" % logits.argmax(-1).tolist())


if __name__ == "__main__":
    main()
