#!/usr/bin/env python3
"""Export Whisper's vocabulary for tokenizer.lua.

    python tools/export_tokenizer.py         # writes tokenizer.lwt

DECODE ONLY, AND THAT IS THE WHOLE POINT

Transcription runs audio -> tokens -> text. Nothing in that chain turns text
into tokens, so the BPE merge table and the merge algorithm are not needed
at all -- only a table from id to token string, and the byte-level mapping
that turns those strings back into bytes.

That removes the single most error-prone part of a GPT-2 tokenizer. An
encoder would need the ~50k merge ranks, the priority loop, and the
pre-tokenizer regex with its Unicode categories, all of which have to match
exactly or words tokenize differently. None of it is required to read the
model's output.

(A prompt or an initial-text conditioning feature would need encoding. If
that is ever added, the merges go in a second file rather than complicating
this one.)

THE BYTE-LEVEL MAPPING

GPT-2 does not store raw bytes in its vocabulary. Byte 32 (space) would be
an actual space in a whitespace-delimited format, so every byte is mapped to
a printable codepoint first: space becomes U+0120 ("G with dot"), and the
token for " the" is literally "Gthe" with that character. Decoding reverses
it, then interprets the resulting bytes as UTF-8.

The 256-entry mapping is exported rather than reconstructed in Lua, for the
same reason as the mel filterbank: it is a constant, and reimplementing the
function that generates it only adds a place to be subtly wrong.

Format:
    magic "LWT1", int32 version
    int32 n_tokens, int32 sot, int32 notimestamps, int32 eot
    int32 timestamp_begin        first <|x.xx|> id, or -1
    int32 n_suppress, then that many int32
    256 x int32                  codepoint for byte 0..255
    n_tokens x (int32 len, len bytes utf-8, int32 is_special)
"""

import argparse
import os
import struct
import sys

MAGIC = b"LWT1"


def bytes_to_unicode():
    """GPT-2's byte -> codepoint map, copied in behaviour from the original.

    Printable ASCII and two Latin-1 runs map to themselves; everything else
    is pushed above 256 so no token string contains whitespace or control
    characters.
    """
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, cs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-tiny.en")
    ap.add_argument("--output", default="tokenizer.lwt")
    args = ap.parse_args()

    from transformers import WhisperTokenizer, WhisperForConditionalGeneration

    tk = WhisperTokenizer.from_pretrained(args.model)
    gc = WhisperForConditionalGeneration.from_pretrained(args.model).generation_config

    n = len(tk)
    strings = []
    specials = set(tk.all_special_ids) | set(tk.added_tokens_encoder.values())

    for i in range(n):
        t = tk.convert_ids_to_tokens(i)
        if t is None:
            sys.exit("vocabulary has a hole at id %d" % i)
        strings.append(t)

    # The timestamp tokens are a contiguous run at the end: <|0.00|> upward.
    ts_begin = -1
    for i, t in enumerate(strings):
        if t == "<|0.00|>":
            ts_begin = i
            break

    suppress = list(getattr(gc, "suppress_tokens", None) or [])

    b2u = bytes_to_unicode()
    codepoints = [b2u[b] for b in range(256)]

    with open(args.output, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<i", 1))
        fh.write(struct.pack("<4i", n,
                             gc.decoder_start_token_id,
                             getattr(gc, "no_timestamps_token_id", -1),
                             gc.eos_token_id))
        fh.write(struct.pack("<i", ts_begin))
        fh.write(struct.pack("<i", len(suppress)))
        for s in suppress:
            fh.write(struct.pack("<i", int(s)))
        for cp in codepoints:
            fh.write(struct.pack("<i", cp))
        for i, t in enumerate(strings):
            raw = t.encode("utf-8")
            fh.write(struct.pack("<i", len(raw)))
            fh.write(raw)
            fh.write(struct.pack("<i", 1 if i in specials else 0))

    print("wrote %s  (%.0f KB)" % (args.output, os.path.getsize(args.output) / 1024))
    print("  tokens         : %d  (%d special)" % (n, len(specials)))
    print("  sot / nots / eot: %d / %d / %d"
          % (gc.decoder_start_token_id,
             getattr(gc, "no_timestamps_token_id", -1), gc.eos_token_id))
    print("  timestamps from: %d  (%r)"
          % (ts_begin, strings[ts_begin] if ts_begin >= 0 else None))
    print("  suppressed     : %d tokens" % len(suppress))
    print("  sample         : 464 -> %r, 23748 -> %r"
          % (strings[464], strings[23748]))


if __name__ == "__main__":
    main()
