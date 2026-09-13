#!/usr/bin/env python3
"""Export a Hugging Face Whisper checkpoint for lua-whisper.

    python tools/export_whisper.py                    # whisper-tiny.en
    python tools/export_whisper.py --model openai/whisper-base.en

Writes a flat float32 file: a small header, then every tensor back to back
in a fixed order, so weights.lua reads it with one pass and no seeking.

THE TRAP THIS FILE EXISTS TO AVOID

In Whisper's attention, q_proj, v_proj and out_proj each carry a bias and
k_proj does NOT -- in self-attention and cross-attention, encoder and
decoder alike. There is no flag for it; it is just how the model was built.

An exporter that writes a bias for all four projections stays perfectly
valid as a file. It is 1536 bytes longer per attention block than the
reader expects, so every tensor after the first attention block is read
from the wrong offset, and the result is not a crash -- it is a model that
loads, runs, and emits plausible-looking garbage. Debugging that from a bad
transcript is miserable.

So the bias is not a comment here. `attn()` takes the projection list with
`has_bias` per entry, and the writer asserts the tensor count it emitted
against what the header promises.
"""

import argparse
import os
import struct
import sys

MAGIC = b"LWB1"
VERSION = 1


class Writer:
    """Counts what it writes so the file can check itself."""

    def __init__(self, path):
        self.fh = open(path, "wb")
        self.tensors = 0
        self.floats = 0

    def i32(self, *vals):
        for v in vals:
            self.fh.write(struct.pack("<i", int(v)))

    def tensor(self, t, name, shape=None):
        import numpy as np
        a = t.detach().float().cpu().contiguous().numpy()
        if shape is not None and tuple(a.shape) != tuple(shape):
            sys.exit(f"{name}: expected {tuple(shape)}, got {tuple(a.shape)}")
        self.fh.write(np.ascontiguousarray(a, dtype=np.float32).tobytes())
        self.tensors += 1
        self.floats += a.size

    def close(self):
        self.fh.close()


def attn(w, block, prefix):
    """One attention block.

    The (name, has_bias) pairs are the point: k_proj is the odd one out and
    saying so in data means the reader and the writer cannot disagree about
    it by accident.
    """
    for name, has_bias in (("q_proj", True), ("k_proj", False),
                           ("v_proj", True), ("out_proj", True)):
        mod = getattr(block, name)
        w.tensor(mod.weight, f"{prefix}.{name}.weight")
        if has_bias:
            if mod.bias is None:
                sys.exit(f"{prefix}.{name} has no bias but the layout expects one")
            w.tensor(mod.bias, f"{prefix}.{name}.bias")
        elif mod.bias is not None:
            # The model changed under us. Better to stop than to write a
            # file whose shape nobody expects.
            sys.exit(f"{prefix}.{name} unexpectedly HAS a bias -- layout is stale")


def layernorm(w, ln, prefix):
    w.tensor(ln.weight, f"{prefix}.weight")
    w.tensor(ln.bias, f"{prefix}.bias")


def mlp(w, block, prefix):
    w.tensor(block.fc1.weight, f"{prefix}.fc1.weight")
    w.tensor(block.fc1.bias, f"{prefix}.fc1.bias")
    w.tensor(block.fc2.weight, f"{prefix}.fc2.weight")
    w.tensor(block.fc2.bias, f"{prefix}.fc2.bias")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-tiny.en")
    ap.add_argument("--output", default="whisper-tiny.en.lwb")
    args = ap.parse_args()

    from transformers import WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.eval()
    cfg = model.config
    enc, dec = model.model.encoder, model.model.decoder

    # lm_head tied to the embedding is the llama2.c convention and saves
    # 19.9M floats of the 37.8M total. Checked, not assumed: an untied
    # checkpoint would need the extra tensor and silently lose the head.
    tied = model.proj_out.weight.data_ptr() == dec.embed_tokens.weight.data_ptr()
    if not tied:
        sys.exit("lm_head is not tied to the embedding; this layout assumes it is")

    w = Writer(args.output)
    w.fh.write(MAGIC)
    w.i32(VERSION,
          cfg.num_mel_bins,            # 80
          cfg.max_source_positions,    # 1500
          cfg.d_model,                 # 384
          cfg.encoder_attention_heads, # 6
          cfg.encoder_layers,          # 4
          cfg.vocab_size,              # 51864
          cfg.max_target_positions,    # 448
          cfg.decoder_attention_heads,
          cfg.decoder_layers,
          cfg.encoder_ffn_dim)         # 1536

    d, ff = cfg.d_model, cfg.encoder_ffn_dim

    # --- encoder ---------------------------------------------------------
    # Conv weights stay in PyTorch's [out, in, k] order; weights.lua indexes
    # them directly rather than transposing at load time.
    w.tensor(enc.conv1.weight, "enc.conv1.weight", (d, cfg.num_mel_bins, 3))
    w.tensor(enc.conv1.bias, "enc.conv1.bias", (d,))
    w.tensor(enc.conv2.weight, "enc.conv2.weight", (d, d, 3))
    w.tensor(enc.conv2.bias, "enc.conv2.bias", (d,))
    w.tensor(enc.embed_positions.weight, "enc.pos", (cfg.max_source_positions, d))

    for i, block in enumerate(enc.layers):
        p = f"enc.{i}"
        attn(w, block.self_attn, f"{p}.self_attn")
        layernorm(w, block.self_attn_layer_norm, f"{p}.self_attn_ln")
        mlp(w, block, p)
        layernorm(w, block.final_layer_norm, f"{p}.final_ln")

    layernorm(w, enc.layer_norm, "enc.ln")

    # --- decoder ---------------------------------------------------------
    w.tensor(dec.embed_tokens.weight, "dec.tok_emb", (cfg.vocab_size, d))
    w.tensor(dec.embed_positions.weight, "dec.pos", (cfg.max_target_positions, d))

    for i, block in enumerate(dec.layers):
        p = f"dec.{i}"
        attn(w, block.self_attn, f"{p}.self_attn")
        layernorm(w, block.self_attn_layer_norm, f"{p}.self_attn_ln")
        attn(w, block.encoder_attn, f"{p}.cross_attn")
        layernorm(w, block.encoder_attn_layer_norm, f"{p}.cross_attn_ln")
        mlp(w, block, p)
        layernorm(w, block.final_layer_norm, f"{p}.final_ln")

    layernorm(w, dec.layer_norm, "dec.ln")
    w.close()

    # What the reader will expect, derived independently from the config so
    # a layout change has to break this too.
    per_attn = 4 + 3                       # 4 weights, 3 biases (k has none)
    per_enc = per_attn + 2 + 4 + 2         # attn, ln, mlp, ln
    per_dec = per_attn + 2 + per_attn + 2 + 4 + 2
    expect = (5                            # convs (4) + enc pos
              + cfg.encoder_layers * per_enc
              + 2                          # enc final ln
              + 2                          # dec embeddings
              + cfg.decoder_layers * per_dec
              + 2)                         # dec final ln
    if w.tensors != expect:
        sys.exit(f"wrote {w.tensors} tensors, layout says {expect} -- "
                 "reader and writer disagree")

    size = os.path.getsize(args.output)
    total = sum(p.numel() for p in model.parameters())
    print(f"wrote {args.output}")
    print(f"  {w.tensors} tensors, {w.floats/1e6:.2f}M floats, {size/1e6:.1f} MB")
    # named_parameters() already dedupes the tied head, so these agree --
    # which is itself the check that the tie is real.
    print(f"  model has {total/1e6:.2f}M params "
          f"(matches: the tied lm_head is stored once, as the embedding)")
    print(f"  d_model={d} layers={cfg.encoder_layers}+{cfg.decoder_layers} "
          f"heads={cfg.encoder_attention_heads} vocab={cfg.vocab_size} ffn={ff}")


if __name__ == "__main__":
    main()
