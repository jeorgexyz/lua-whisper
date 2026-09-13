#!/usr/bin/env python3
"""Check tokenizer.lua's decode against WhisperTokenizer.

    python tools/check_tokenizer.py

Decoding is where a byte-level vocabulary goes quietly wrong: the failure is
not a crash, it is missing spaces, or a stray "G with dot" in the middle of
a word. So this decodes real id sequences and compares the exact strings.

Coverage is chosen for the byte-level mapping specifically:

  every single id      all 51,864, one at a time -- catches any hole in the
                       vocabulary or the codepoint table
  random sequences     multi-token, seeded
  leading spaces       the U+0120 case that makes or breaks word spacing
  punctuation, digits  short tokens that often merge wrongly
  non-ascii            multi-byte UTF-8 split across tokens, which only
                       decodes correctly if bytes are joined before being
                       interpreted rather than after

RESULTS COME BACK AS HEX, AND THEY HAVE TO

A byte-level vocabulary contains tokens that decode to a literal newline
and to a carriage return. Shipping raw decoded text over a line-delimited
protocol therefore cannot frame itself -- and Windows text-mode stdout
rewrites newlines on top of that. The first version of this check did
exactly that and reported all 53,871 cases as failures, which is the
signature of a broken harness rather than a broken tokenizer: a real bug
does not fail literally everything identically.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LUA = os.environ.get("LUA", "lua54")

DRIVER = "\n".join([
    'package.path = "./?.lua;" .. package.path',
    'local tokenizer = require("tokenizer")',
    'local tk = assert(tokenizer.load("tokenizer.lwt"))',
    'for line in io.lines() do',
    '  local ids = {}',
    '  for v in line:gmatch("%-?%d+") do ids[#ids+1] = tonumber(v) end',
    '  local s = tk:decode(ids, true)',
    '  local hex = s:gsub(".", function(ch)',
    '    return string.format("%02x", string.byte(ch))',
    '  end)',
    '  io.write(hex, "\\n")',
    'end',
    '',
])


def main():
    from transformers import WhisperTokenizer
    tk = WhisperTokenizer.from_pretrained("openai/whisper-tiny.en")

    if not os.path.exists(os.path.join(ROOT, "tokenizer.lwt")):
        sys.exit("tokenizer.lwt missing -- run: python tools/export_tokenizer.py")

    import random
    rng = random.Random(0)
    n = len(tk)

    cases = []
    cases += [[i] for i in range(n)]
    cases += [[rng.randrange(0, 50257) for _ in range(rng.randrange(2, 12))]
              for _ in range(2000)]
    for text in ["The quick brown fox", "hello, world!", "  leading spaces",
                 "1234567890", "café naïve 你好",
                 "don't -- it's \"quoted\"", "\U0001f600 emoji"]:
        cases.append(tk.encode(text, add_special_tokens=False))

    driver = os.path.join(ROOT, "tools", "_tok_dump.lua")
    with open(driver, "w", encoding="utf-8") as fh:
        fh.write(DRIVER)
    try:
        stdin = "\n".join(" ".join(str(i) for i in ids)
                          for ids in cases).encode("ascii")
        proc = subprocess.run([LUA, "tools/_tok_dump.lua"], cwd=ROOT,
                              input=stdin, capture_output=True)
    finally:
        os.remove(driver)
    if proc.returncode != 0:
        sys.exit("lua failed:\n" + proc.stderr.decode("utf-8", "replace")[:800])

    # Hex, whitespace-separated: immune to whatever the payload contains and
    # to newline translation.
    got = [bytes.fromhex(h) for h in proc.stdout.decode("ascii").split()]

    if len(got) != len(cases):
        sys.exit("got %d results for %d cases" % (len(got), len(cases)))

    # HF's decode() runs bytes through errors="replace", so a token holding
    # a lone UTF-8 continuation byte comes back as U+FFFD. tokenizer.lua
    # returns the raw byte, which is the behaviour that COMPOSES: join two
    # such tokens and you get a valid character, whereas replacing each in
    # isolation destroys it. Comparing against decode() would therefore be
    # comparing against a lossy reference and marking the correct answer
    # wrong -- 1,923 of them.
    #
    # So the expected bytes are built from HF's own byte_decoder, which is
    # the same table export_tokenizer.py wrote out, applied without the
    # lossy step.
    # This version of WhisperTokenizer exposes no byte_decoder, so the
    # mapping is rebuilt from the same function export_tokenizer.py uses --
    # one source of truth for the table itself.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "expt", os.path.join(HERE, "export_tokenizer.py"))
    expt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(expt)
    bd = {chr(cp): b for b, cp in expt.bytes_to_unicode().items()}

    def want_bytes(ids):
        out = bytearray()
        for i in ids:
            t = tk.convert_ids_to_tokens(i)
            if t in tk.added_tokens_encoder:
                out += t.encode("utf-8")          # specials are literal text
            else:
                out += bytes(bd[ch] for ch in t)
        return bytes(out)

    bad, shown = 0, 0
    for i, ids in enumerate(cases):
        want = want_bytes(ids)
        if got[i] != want:
            bad += 1
            if shown < 8:
                shown += 1
                print("MISMATCH ids=%s\n  lua %r\n  ref %r"
                      % (ids[:8], got[i], want))

    # Sharing the mapping with the exporter means the check above proves the
    # Lua reads it correctly, not that the mapping itself is right. The text
    # cases close that gap: they go through HF's own decode(), so a wrong
    # byte table shows up as mangled words here even though every byte-level
    # comparison passed.
    text_bad = 0
    for text in ["The quick brown fox", "hello, world!", "  leading spaces",
                 "café naïve 你好", "\U0001f600 emoji"]:
        ids = tk.encode(text, add_special_tokens=False)
        idx = cases.index(ids)
        have = got[idx].decode("utf-8", "replace")
        want = tk.decode(ids, skip_special_tokens=False,
                         clean_up_tokenization_spaces=False)
        if have != want:
            text_bad += 1
            print("TEXT MISMATCH\n  lua %r\n  ref %r" % (have, want))

    print()
    print("cases        : %d" % len(cases))
    print("  singles    : %d (every id)" % n)
    print("  sequences  : 2000 random + 7 hand-picked")
    print("mismatches   : %d (byte-exact)" % bad)
    print("text check   : %d mismatched (vs HF decode)" % text_bad)
    if bad or text_bad:
        sys.exit("tokenizer.lua disagrees with WhisperTokenizer")
    print("tokenizer.lua matches WhisperTokenizer")


if __name__ == "__main__":
    main()
