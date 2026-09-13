# Lua Whisper

Speech recognition in pure Lua 5.3+. CPU only, no Torch, C extensions, or
runtime dependencies.

The fourth project in the same line as
[lua-llama](https://github.com/jeorgexyz/lua-llama),
[lua-agent](https://github.com/jeorgexyz/lua-agent) and
[lua-mamba](https://github.com/jeorgexyz/lua-mamba)

> **Status: in progress.** Front end and weight loading are done and
> checked against Hugging Face. The encoder, decoder and tokenizer are not
> written yet.

## Design Philosophy

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

## The front end

`audio.lua` turns a WAV into the 80x3000 log-mel the encoder expects, and is
checked against `WhisperFeatureExtractor` itself:

```
signal             frames   max |diff|  mean |diff|
silence              3000    0.000e+00    0.000e+00
dc                   3000    5.960e-08    5.811e-08
sine 440             3000    1.162e-05    5.721e-08
chirp                3000    6.089e-05    2.776e-07
noise                3000    1.516e-06    2.193e-08
short 3s             3000    1.812e-05    7.404e-08

worst max |diff| 6.089e-05 (chirp)
```

That is float32 storage plus a log10, not a difference in behaviour. The
signals are synthetic and seeded, so the check needs no audio file:
`silence` exercises the mel floor and the clamp, `chirp` sweeps every band
and would catch a flipped frequency axis, `short 3s` runs the pad-to-window
path.

```bash
python tools/export_mel.py    # writes mel80.bin, 63 KB
python tools/check_mel.py
```

Whole front end: **5.1 s** for a 30-second clip, 1.4% of the estimated
encoder time.

### Four conventions that are easy to get wrong

Each of these is invisible when wrong -- the spectrogram still looks like a
spectrogram, and the transcript is just quietly worse:

- **Periodic** Hann (divide by `n`, not `n-1`). `torch.hann_window` defaults
  to periodic; the symmetric variant is a different window.
- **Reflect** padding by 200, not zeros. `[a,b,c,d]` becomes
  `[c,b,a,b,c,d,c,b]` -- the edge sample is not repeated.
- The last frame is dropped **after** the log. Framing gives 3001 frames for
  30 seconds; the model wants 3000.
- The clamp is relative to **this clip's** maximum, so it is not a fixed
  floor and two clips normalise differently. That is intended.

The mel filterbank is exported as a constant rather than reconstructed in
Lua. It is a fixed 201x80 matrix that depends only on sample rate and
`n_fft`, and reimplementing hertz-to-mel plus Slaney normalisation would be
eighty lines whose only job is to reproduce a table that already exists --
every one of them a place to put a silent half-bin offset.

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

## Weights

`tools/export_whisper.py` writes a flat float32 checkpoint; `weights.lua`
reads it in one pass. 167 tensors, 37.76M floats, 151 MB, loading in 3.8s.

```bash
python tools/export_whisper.py     # writes whisper-tiny.en.lwb
python tools/check_weights.py
```

```
tensor                     elements   max |diff|
enc.conv1.weight              92160    0.000e+00
enc.0.k_proj.weight          147456    0.000e+00
enc.pos                      576000    0.000e+00
dec.tok_emb                19915776    0.000e+00
dec.last.final_ln.bias          384    0.000e+00

encoder self-attn k_proj has no bias     yes
decoder cross-attn k_proj has no bias    yes
```

Exact, not approximate — it is the same float32 bits, just relocated.

### The bias trap, and why it is checked three ways

In Whisper's attention, `q_proj`, `v_proj` and `out_proj` each carry a bias
and **`k_proj` does not** — self-attention and cross-attention, encoder and
decoder alike. There is no flag for it.

An exporter that writes a bias for all four produces a perfectly valid
file, 384 floats longer per attention block than the reader expects. Every
tensor after the first block is then read from the wrong offset, and the
result is not a crash: the model loads, runs, and transcribes plausible
nonsense. That is a genuinely miserable thing to debug backwards from bad
text.

So it is not left to a comment:

1. Both sides list `(name, has_bias)` as data, in the same order, so they
   cannot drift apart by accident.
2. The exporter counts the tensors it wrote and compares against a count
   derived independently from the config.
3. The reader checks it consumed the file to the byte, and
   `check_weights.py` probes actual values — including the *last* tensor in
   the file, which can only match if everything before it is aligned.

Size checks alone would not catch a wrong *order*: swap two tensors of
equal shape and the byte count stays perfect. Hence the value probes.

## Runtime

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
audio.lua       WAV parsing, framing, mel projection    [done]
fft.lua         the transform                          [done]
encoder.lua     conv frontend, 4 transformer layers
decoder.lua     self-attention, cross-attention, tied lm_head
weights.lua     flat float32 loader                   [done]
tokenizer.lua   GPT-2 byte-level BPE
validate.lua    per-layer parity against a PyTorch dump
tools/export_whisper.py                                [done]
tools/reference.py
tools/check_fft.py                                     [done]
tools/check_mel.py                                     [done]
tools/export_mel.py                                    [done]
tools/check_weights.py                                 [done]
```

Two claims, once it runs: per-layer parity against Hugging Face, and an
exact greedy transcript match on a known clip.

## Notes

In Whisper's attention, `k_proj` has **no bias** while `q_proj`, `v_proj`
and `out_proj` all do — in both self- and cross-attention. An exporter that
assumes uniform bias will either crash or, worse, silently misalign the
whole weight stream and leave you debugging it from garbage transcripts.

## License

MIT
