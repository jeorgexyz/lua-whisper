#!/usr/bin/env python3
"""Dump Hugging Face encoder activations for validate_encoder.lua.

    python tools/reference.py                       # synthetic signal
    python tools/reference.py --wav clip.wav

Writes the mel the reference actually consumed, then the activations after
every stage, so the Lua side is compared on identical input and a
divergence points at one stage instead of at "the transcript is worse".

A full encoder pass is minutes in pure Lua, so the stages matter: catching
a conv layout error after 18 seconds beats catching it after six minutes,
and catching it at layer 1 beats staring at layer 4.

Layouts are chosen to match what encoder.lua produces internally, so the
comparison needs no reshaping on either side:

    mel        n_mels x n_frames      (channel-major, as audio.lua returns)
    conv       d x T                  (channel-major, conv1d output order)
    layer i    T x d                  (time-major, after the residual add)
    final      T x d                  (after the encoder's final LayerNorm)
"""

import argparse
import struct
import sys

import numpy as np

MAGIC = b"LWR1"
SR = 16000


def synthetic(seconds=30):
    """Deterministic, speech-shaped enough to light most mel bands.

    Real speech would be better for a transcript test; for parity the only
    requirements are that both sides see identical samples and that the
    signal is not so simple it hides a bug (silence would pass almost any
    broken encoder).
    """
    rng = np.random.default_rng(0)
    t = np.arange(int(SR * seconds)) / SR
    sig = np.zeros_like(t)
    for f0, amp in ((120, 0.5), (240, 0.25), (480, 0.15), (900, 0.1)):
        env = 0.5 + 0.5 * np.sin(2 * np.pi * 2.7 * t + f0)
        sig += amp * env * np.sin(2 * np.pi * f0 * t)
    sig += 0.02 * rng.standard_normal(t.size)
    return (sig / np.abs(sig).max() * 0.8).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-tiny.en")
    ap.add_argument("--wav")
    ap.add_argument("--output", default="reference.ref")
    args = ap.parse_args()

    import torch
    from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor

    if args.wav:
        import wave
        with wave.open(args.wav, "rb") as wf:
            if wf.getsampwidth() != 2:
                sys.exit("only 16-bit PCM wav is supported")
            raw = wf.readframes(wf.getnframes())
            audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            if wf.getnchannels() > 1:
                audio = audio.reshape(-1, wf.getnchannels()).mean(axis=1)
            if wf.getframerate() != SR:
                sys.exit("wav must be 16 kHz; convert with ffmpeg -ar 16000")
    else:
        audio = synthetic()

    fe = WhisperFeatureExtractor.from_pretrained(args.model)
    mel = fe(audio, sampling_rate=SR, return_tensors="np").input_features[0]

    model = WhisperForConditionalGeneration.from_pretrained(args.model).eval()
    enc = model.model.encoder

    stages = {}

    # conv stem, before the permute to time-major
    def conv_hook(mod, inp, out):
        stages["conv"] = out.detach().float().cpu().numpy()[0]
    h = enc.conv2.register_forward_hook(conv_hook)

    layer_out = []
    hooks = [h]
    for layer in enc.layers:
        def mk(store):
            def hook(mod, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                store.append(o.detach().float().cpu().numpy()[0])
            return hook
        hooks.append(layer.register_forward_hook(mk(layer_out)))

    with torch.no_grad():
        final = enc(torch.from_numpy(mel).unsqueeze(0)).last_hidden_state
    for h in hooks:
        h.remove()

    final = final.detach().float().cpu().numpy()[0]

    # conv output is pre-GELU on the hook; apply it so the dump matches
    # what encoder.lua holds at the same point.
    conv = stages["conv"]
    conv = (0.5 * conv * (1.0 + torch.erf(torch.from_numpy(conv)
                                          / np.sqrt(2.0)).numpy()))

    n_mels, n_frames = mel.shape
    T, d = final.shape
    n_layers = len(layer_out)

    with open(args.output, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<6i", 1, n_mels, n_frames, T, d, n_layers))
        for arr, name in ([(mel, "mel"), (conv, "conv")]
                          + [(a, "layer%d" % i) for i, a in enumerate(layer_out)]
                          + [(final, "final")]):
            a = np.ascontiguousarray(arr, dtype=np.float32)
            fh.write(a.tobytes())

    print("wrote %s" % args.output)
    print("  mel    %d x %d" % (n_mels, n_frames))
    print("  conv   %s  (channel-major, post-gelu)" % (conv.shape,))
    print("  layers %d x %s  (time-major)" % (n_layers, layer_out[0].shape))
    print("  final  %s" % (final.shape,))


if __name__ == "__main__":
    main()
