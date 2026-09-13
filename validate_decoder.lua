-- validate_decoder.lua - compare decoder.lua against a PyTorch dump.
--
--   lua54 validate_decoder.lua whisper-tiny.en.lwb reference_decode.ref
--
-- Teacher-forced: the reference supplies both the token sequence and the
-- encoder states, and the Lua decoder is stepped over the same tokens. A
-- free-running comparison would diverge at the first differing argmax and
-- say nothing about where the fault is.
--
-- Using the reference's encoder states rather than running encoder.lua is
-- deliberate. The encoder has its own parity check; feeding its output in
-- here would leave a failure ambiguous between the two, and would add five
-- minutes to a check that otherwise takes seconds per token.
--
-- Two things are reported per position:
--
--   logits    how close the raw scores are
--   argmax    whether the chosen token matches, which is what a transcript
--             actually depends on. Logits can drift a little with no effect
--             on the output; a changed argmax is a changed word.

package.path = "./?.lua;" .. package.path

local weights = require('weights')
local decoder = require('decoder')

local ckpt = arg[1] or "whisper-tiny.en.lwb"
local refpath = arg[2] or "reference_decode.ref"

local fh = assert(io.open(refpath, "rb"), "cannot open " .. refpath)
local blob = fh:read("a")
fh:close()
assert(blob:sub(1, 4) == "LWD1", refpath .. " is not a decoder reference")

local version, T, d, n_tok, vocab = string.unpack("<i4i4i4i4i4", blob, 5)
assert(version == 1, "unsupported reference version " .. version)

local pos = 5 + 20
local function floats(n)
    local t = {}
    for i = 1, n do t[i] = string.unpack("<f", blob, pos) pos = pos + 4 end
    return t
end
local function ints(n)
    local t = {}
    for i = 1, n do t[i] = string.unpack("<i4", blob, pos) pos = pos + 4 end
    return t
end

io.write(string.format("reference: encoder %dx%d, %d tokens, vocab %d\n",
    T, d, n_tok, vocab))

local enc_states = floats(T * d)
local tokens = ints(n_tok)
local ref_logits = floats(n_tok * vocab)

local w = assert(weights.load(ckpt))
assert(w.config.d_model == d, "checkpoint/reference d_model mismatch")
assert(w.config.vocab_size == vocab, "checkpoint/reference vocab mismatch")

io.write("precomputing cross-attention keys and values...\n")
io.flush()
local t0 = os.clock()
local cross = decoder.prepare_cross(w, enc_states, T)
io.write(string.format("  done in %.0fs (once, reused by every step)\n\n",
    os.clock() - t0))

local st = decoder.new_state(w, n_tok)

local worst, worst_at = 0.0, 0
local argmax_ok, checked = 0, 0

io.write(string.format("%-4s %-8s %12s %12s %10s %10s\n",
    "pos", "token", "max |diff|", "rel", "lua", "ref"))
io.write(string.format("%-4s %-8s %12s %12s %10s %10s\n",
    "----", "--------", "------------", "------------", "----------", "----------"))

for i = 1, n_tok do
    local step_t = os.clock()
    local logits = decoder.step(w, cross, st, tokens[i], i)

    local base = (i - 1) * vocab
    local mx, rel = 0.0, 0.0
    local lua_best, lua_v = 0, -math.huge
    local ref_best, ref_v = 0, -math.huge

    for v = 1, vocab do
        local a, b = logits[v], ref_logits[base + v]
        local e = math.abs(a - b)
        if e > mx then mx = e end
        local mag = math.abs(b)
        if mag > 1.0 then
            local r = e / mag
            if r > rel then rel = r end
        end
        if a > lua_v then lua_v, lua_best = a, v - 1 end
        if b > ref_v then ref_v, ref_best = b, v - 1 end
    end

    if mx > worst then worst, worst_at = mx, i end
    checked = checked + 1
    if lua_best == ref_best then argmax_ok = argmax_ok + 1 end

    io.write(string.format("%-4d %-8d %12.3e %12.3e %10d %10d%s   %.1fs\n",
        i, tokens[i], mx, rel, lua_best, ref_best,
        lua_best == ref_best and "" or "  MISMATCH", os.clock() - step_t))
    io.flush()
end

io.write(string.format("\nworst |diff| %.3e at position %d\n", worst, worst_at))
io.write(string.format("argmax agreement: %d/%d\n", argmax_ok, checked))

-- The argmax is the claim that matters: it is what a transcript is made of.
-- Logit drift at 1e-2 on values of this scale is float32 accumulation.
if argmax_ok ~= checked then
    io.write("FAIL: the decoder would emit different tokens\n")
    os.exit(1)
end
if worst > 0.1 then
    io.write(string.format("FAIL: logit drift %.3e is too large\n", worst))
    os.exit(1)
end
io.write("decoder parity passed\n")
