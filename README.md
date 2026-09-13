# Lua Whisper

Speech recognition in pure Lua 5.3+. CPU only, no Torch, no C extensions, no
runtime dependencies.

The fourth project in the same line as
[lua-llama](https://github.com/jeorgexyz/lua-llama),
[lua-agent](https://github.com/jeorgexyz/lua-agent) and
[lua-mamba](https://github.com/jeorgexyz/lua-mamba): keep the mechanism small
enough to read, then give it a falsifiable correctness test.

> **Status: in progress.** The spectrogram front end is written and checked
> against numpy. The encoder, decoder and tokenizer are not written yet.

## What this is not

[whisper.cpp](https://github.com/ggml-org/whisper.cpp) is the real runtime.
It is faster than real time, runs on Metal and CUDA, quantizes, does beam
search, VAD, streaming and word timestamps. **If you want to transcribe
something, use it.**

This is the other half of that pair, the same way `llama2.c` sits beside
`llama.cpp`:

| | whisper.cpp | lua-whisper |
|---|---|---|
| for | using | reading |
| speed | real time or better | ~6 min per 30s clip |
| size | thousands of lines over a GGML backend | target ~1200 lines, no dependencies |
| has | SIMD, GPU, quantization, beam search, VAD, streaming | one greedy decode path |

What you get in exchange is a mel front end and a cross-attention decoder
you can read end to end in an afternoon, and a parity check against PyTorch
that says the reading was accurate.

## The model

`openai/whisper-tiny.en`, 37.8M parameters.

| | |
|---|---|
| d_model / layers / heads | 384 / 4 encoder + 4 decoder / 6 |
| vocab | 51864, lm_head tied to the embedding |
| encoder / decoder params | 8.2M / 29.6M (19.9M of it the tied embedding) |
| conv frontend | `[384,80,3]` then `[384,384,3]` stride 2 |
| positions | learned, 1500 encoder / 448 decoder |
| front end | 16 kHz, n_fft 400, hop 160, 80 mels, fixed 30s window |

## The spectrogram

`fft.lua` is written and checked. Seven signals against `numpy.fft.rfft`,
each chosen to catch a specific mistake rather than to look thorough — an
impulse for twiddle indexing, DC for an off-by-one in the bin index, an
on-bin sine for a sign error in the imaginary part, an off-bin sine for
window placement:

```
signal               max |diff|          rel
impulse               0.000e+00    0.000e+00
shifted impulse       4.017e-15    4.017e-15
dc                    1.643e-12    4.109e-15
sine on bin           1.489e-12    7.447e-15
sine off bin          1.381e-12    1.059e-14
noise                 3.035e-12    6.115e-14
noise * hann          1.453e-12    4.476e-14

worst relative error 6.115e-14 (noise)
```

```bash
python tools/check_fft.py     # needs numpy
lua54 bench_fft.lua
```

### Why a direct DFT and not an FFT

Whisper uses `n_fft = 400`, which is not a power of two, so radix-2 does not
apply — and zero-padding to 512 is a *different transform*, with bins at
different frequencies, not a workaround. The real options are mixed-radix
(400 = 2⁴·5²), Bluestein, or the definition with precomputed twiddles.

This uses the definition, because of the measurement rather than taste:

```
per frame      1.500 ms
30s clip       4.5 s  (3000 frames)
spectrogram is 1.2% of the pipeline
```

The encoder is ~37 GFLOP, roughly six minutes in pure Lua. Bluestein could
save at most four of the 4.5 seconds. Several hundred lines of index
arithmetic to move a 1.2% number is a bad trade in a repo built to be read.
`bench_fft.lua` prints these numbers so the trade can be checked rather than
believed.

## Runtime, honestly

Measured pure-Lua throughput on the development machine is 164 MFLOP/s peak
matmul, ~104 MFLOP/s end to end across lua-llama and lua-mamba. From that:

| | |
|---|---|
| spectrogram | 4.5 s (measured) |
| encoder, 30s clip | ~6 min (~37 GFLOP, fixed cost) |
| decoder | ~0.6 s/token, two thirds of it the 51864-wide lm_head |

Whisper always encodes a 30-second window regardless of clip length. The
positional embedding can be sliced for shorter audio — a 5-second clip needs
250 positions instead of 1500 — which is roughly a 6x saving and what
whisper.cpp does. Parity testing has to use the padded path, since that is
what Hugging Face computes.

## Planned

```text
main.lua        wav in, text out
audio.lua       WAV parsing, framing, mel projection
fft.lua         the transform                          [done]
encoder.lua     conv frontend, 4 transformer layers
decoder.lua     self-attention, cross-attention, tied lm_head
weights.lua     flat float32 loader
tokenizer.lua   GPT-2 byte-level BPE
validate.lua    per-layer parity against a PyTorch dump
tools/export_whisper.py
tools/reference.py
tools/check_fft.py                                     [done]
```

Two claims, once it runs: per-layer parity against Hugging Face, and an
exact greedy transcript match on a known clip.

## One trap worth writing down

In Whisper's attention, `k_proj` has **no bias** while `q_proj`, `v_proj`
and `out_proj` all do — in both self- and cross-attention. An exporter that
assumes uniform bias will either crash or, worse, silently misalign the
whole weight stream and leave you debugging it from garbage transcripts.

## License

MIT
