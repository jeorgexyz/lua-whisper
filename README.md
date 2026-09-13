# Lua Whisper

Speech recognition in pure Lua 5.3+. CPU only, no Torch, C extensions, or
runtime dependencies.

The fourth project in the same line as
[lua-llama](https://github.com/jeorgexyz/lua-llama),
[lua-agent](https://github.com/jeorgexyz/lua-agent) and
[lua-mamba](https://github.com/jeorgexyz/lua-mamba)

> **Status: working.** WAV in, text out, matching a completely different
> runtime character for character. See [End to end](#end-to-end).

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

## End to end

```bash
lua54 main.lua examples/speech.wav
```

```
audio      3.7s, 16000 Hz, 1 channel(s)
mel        5.2s
encoder    312s, 1500 positions
cross-attn 25s (once, reused by every token)

 The quick brown fox jumps over the lazy dog.

tokens     10 in 8s (0.80s each)
total      353s
```

The clip is Windows TTS speaking a known sentence, so there is a ground
truth independent of any model. Three runtimes on the same file:

| | transcript |
|---|---|
| ground truth (what was spoken) | The quick brown fox jumps over the lazy dog. |
| faster-whisper, CTranslate2 int8 | The quick brown fox jumps over the lazy dog. |
| **lua-whisper, pure Lua float32** | **The quick brown fox jumps over the lazy dog.** |

Character-exact against a different language, a different weight format and
a different numeric precision. Agreement across that gap says more than any
per-layer epsilon can: the per-stage checks prove the arithmetic matches,
this proves the whole thing *behaves* the same.

The faster-whisper row is [polymimus](https://github.com/jeorgexyz/polymimus)'
engine -- the practical counterpart to this repo, and a useful way to
generate reference clips for it.

### Where the time goes

| stage | time | share |
|---|---|---|
| spectrogram | 5.2s | 1.5% |
| encoder | 312s | 88% |
| cross-attention K/V | 25s | 7% |
| 10 tokens | 8s | 2% |

353s total against the 356s predicted from measured pure-Lua throughput,
which is closer than it had any right to be. The encoder is a fixed cost:
Whisper always encodes a 30-second window, so a 3.7-second clip pays the
same as a full one.

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

## Encoder

`encoder.lua` is the conv stem plus four pre-norm transformer layers,
checked stage by stage against a Hugging Face dump:

```
  conv         max 1.753e-05   mean 2.578e-07   rel 9.517e-06
  layer 1      max 1.392e-05   mean 5.201e-07   rel 8.623e-06
  layer 2      max 1.505e-05   mean 7.682e-07   rel 1.266e-05
  layer 3      max 1.893e-05   mean 1.176e-06   rel 1.418e-05
  layer 4      max 2.173e-02   mean 3.296e-06   rel 7.085e-04
  final        max 9.217e-04   mean 1.840e-06   rel 5.526e-04

elements outside |a-b| <= 5e-3 + 1e-3|b|: 0
encoder parity passed
```

```bash
python tools/reference.py                          # dumps every stage
lua54 validate_encoder.lua whisper-tiny.en.lwb reference.ref
lua54 validate_encoder.lua whisper-tiny.en.lwb reference.ref 1   # conv + layer 1
```

306 seconds for a 30-second clip. The third argument stops early, which
matters: a conv layout error surfaces 25 seconds in rather than after the
full pass.

### Whisper needs a relative tolerance, and lua-mamba did not

The first full run failed. Layer 4 reported a max absolute error of 2.2e-2
against a 1e-3 threshold.

It was not a bug, and the tell was that **the max jumped a thousandfold
while the mean barely moved** — 1.18e-6 to 3.3e-6. A real error moves both.

The worst element sits at a reference value of **-320.65**, a relative error
of 6.8e-5. Layer 4 of tiny.en peaks at **|564|**, with 52 of its 576,000
values above 50 — the massive activations transformers are known to carry.
At that magnitude, ordinary float32 noise *is* 2e-2 in absolute terms.

So the check now gates the way `numpy.allclose` does, on
`|a-b| <= atol + rtol*|b|`, and reports relative error beside absolute.
`lua-mamba` got away with a plain absolute tolerance because its activations
are O(1); Whisper's are not, and that is an architectural difference worth
naming rather than tuning around.

### The GELU has to be the exact one

`F.gelu` defaults to the erf form and Whisper was trained that way. The erf
here (Abramowitz & Stegun 7.1.26) is accurate to **2.08e-7**; the tanh
approximation is off by **4.7e-4**, which is above tolerance and would fail
the check outright. Measured, not assumed.

### Other differences from the llama-family siblings

All simplifications except the GELU:

| | lua-llama / lua-mamba | here |
|---|---|---|
| norm | RMSNorm | LayerNorm with bias, eps 1e-5 |
| positions | RoPE | learned, added once |
| activation | SwiGLU | exact GELU |
| masking | causal | none — audio attention is bidirectional |
| cache | KV cache | none; the encoder runs once over fixed 1500 positions |

## Decoder

`decoder.lua` is self-attention over generated tokens, cross-attention over
the audio, and the tied lm_head. Teacher-forced against a PyTorch dump:

```
pos  token      max |diff|          rel        lua        ref
1    50257      1.507e-05    2.699e-06      50361      50361
...
11   3290       6.444e-06    4.147e-06        338        338

worst |diff| 1.507e-05 at position 1
argmax agreement: 11/11
decoder parity passed
```

```bash
python tools/reference_decode.py
lua54 validate_decoder.lua whisper-tiny.en.lwb reference_decode.ref
```

0.7s per token. **Argmax agreement is the headline, not the logit
difference** -- logit drift at 1e-5 changes nothing, while a changed argmax
is a changed word. The check fails on either, but only one of them maps onto
"the transcript is the same".

### Two caches, doing different jobs

The self-attention cache is the familiar one: keys and values for tokens
already emitted, appended one per step.

The cross-attention cache is what makes decoding cheap. Its keys and values
come from the **encoder** output, which does not change while decoding, so K
and V for all 1500 audio positions are computed once before the first token
and reused forever after. That is 26 seconds here. Recomputing it per token
would add those 26 seconds to every step and turn a 50-token transcript from
about thirty seconds into fifteen minutes.

### What dominates a step

Not attention -- the **lm_head**. 384 x 51864 is 19.9M multiply-accumulates,
roughly 40 of the ~60 MFLOP a step costs. Everything else together is a
third of the work.

### Teacher-forced, and not via generate()

The reference scores a fixed token sequence rather than whatever the model
would produce. Parity only needs both sides to see the same tokens, and
decoupling from `generate()` avoided two real problems: the synthetic test
signal is not speech, so greedy decoding emits a single token and there is
nothing to compare; and `generate()`'s first positional argument is
`input_ids` in transformers 5.x, so `generate(mel, ...)` feeds a
spectrogram into a token embedding and fails with an indices dtype error
that reads like a decoder bug and is not one.

The reference also supplies the encoder states. `encoder.lua` has its own
parity check, and feeding its output in here would leave a failure ambiguous
between the two -- as well as adding five minutes to a check that otherwise
takes seconds per token.

## Tokenizer

Decode only, and that is the whole point. Transcription runs
audio -> tokens -> text; nothing in that chain turns text into tokens, so
there is no merge table, no BPE priority loop, and no pre-tokenizer regex
with Unicode category classes. That is the error-prone half of a GPT-2
tokenizer and none of it is needed to read the model's output.

```bash
python tools/export_tokenizer.py     # writes tokenizer.lwt, 768 KB
python tools/check_tokenizer.py
```

```
cases        : 53871
  singles    : 51864 (every id)
  sequences  : 2000 random + 7 hand-picked
mismatches   : 0 (byte-exact)
text check   : 0 mismatched (vs HF decode)
tokenizer.lua matches WhisperTokenizer
```

### The byte-level mapping, and the two ways checking it goes wrong

GPT-2 does not store raw bytes: byte 32 is a space, which cannot appear in a
whitespace-delimited vocabulary, so every byte is mapped to a printable
codepoint first. The token for `" the"` is U+0120 followed by `the`.
Decoding reverses that and reads the result as UTF-8. Get it wrong and the
transcript comes out with missing spaces or a stray `G`-with-dot mid-word --
quiet, not loud.

Two things made the *check* wrong before the tokenizer was:

- **A byte-level vocabulary contains tokens that decode to a literal newline
  and a carriage return.** Shipping decoded text over a line-delimited
  protocol cannot frame itself. All 53,871 cases "failed" identically, which
  is the signature of a broken harness -- a real bug does not fail
  everything. Results now cross as hex.
- **HF's `decode()` applies `errors="replace"`**, so a token holding a lone
  UTF-8 continuation byte comes back as U+FFFD. `tokenizer.lua` returns the
  raw byte, which is the behaviour that *composes*: join two such tokens and
  you get a valid character; replace each in isolation and you have
  destroyed it. Comparing against `decode()` marked the correct answer wrong
  1,923 times. Byte-exactness is now checked against the mapping directly,
  with the text cases still going through `decode()` so a wrong table would
  still surface.

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
main.lua        wav in, text out                       [done]
audio.lua       WAV parsing, framing, mel projection    [done]
fft.lua         the transform                          [done]
encoder.lua     conv frontend, 4 transformer layers    [done]
decoder.lua     self-attn, cross-attn, tied lm_head     [done]
weights.lua     flat float32 loader                   [done]
tokenizer.lua   GPT-2 byte-level BPE, decode only     [done]
validate_encoder.lua  per-stage parity vs a PyTorch dump [done]
tools/export_whisper.py                                [done]
tools/reference.py                                     [done]
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
