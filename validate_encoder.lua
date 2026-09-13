-- validate_encoder.lua - compare encoder.lua against a PyTorch dump.
--
--   lua54 validate_encoder.lua whisper-tiny.en.lwb reference.ref [max_layers]
--
-- A full encoder pass is minutes, so this reports every stage as it
-- reaches it rather than one number at the end. A conv layout error shows
-- up about twenty seconds in; without staging you would wait six minutes
-- and then only know that the final states were wrong.
--
-- The third argument stops early. `... reference.ref 1` checks the conv
-- stem and the first layer and exits, which is the loop you actually want
-- while the code is still wrong.
--
-- The mel comes from the reference file, not from audio.lua. That is
-- deliberate: audio.lua has its own check against WhisperFeatureExtractor,
-- and feeding the reference's own mel here means a failure is an encoder
-- failure rather than an ambiguity about which stage drifted.

package.path = "./?.lua;" .. package.path

local weights = require('weights')
local encoder = require('encoder')

local ckpt = arg[1] or "whisper-tiny.en.lwb"
local refpath = arg[2] or "reference.ref"
local max_layers = tonumber(arg[3])

local fh = assert(io.open(refpath, "rb"), "cannot open " .. refpath)
local blob = fh:read("a")
fh:close()
assert(blob:sub(1, 4) == "LWR1", refpath .. " is not a lua-whisper reference")

-- Lua's string.unpack has no repeat count; "<6i4" is Python struct syntax
-- and raises "invalid format option '6'" here.
local version, n_mels, n_frames, T, d, n_layers =
    string.unpack("<i4i4i4i4i4i4", blob, 5)
assert(version == 1, "unsupported reference version " .. version)

local pos = 5 + 24
local function take(n)
    local t = {}
    for i = 1, n do
        t[i] = string.unpack("<f", blob, pos)
        pos = pos + 4
    end
    return t
end

io.write(string.format("reference: mel %dx%d, T=%d, d=%d, %d layers\n\n",
    n_mels, n_frames, T, d, n_layers))

local ref_mel  = take(n_mels * n_frames)
local ref_conv = take(d * T)
local ref_layer = {}
for i = 1, n_layers do ref_layer[i] = take(T * d) end
local ref_final = take(T * d)

-- mel[m][t], the shape encoder.encode expects
local mel = {}
for m = 1, n_mels do
    local row, base = {}, (m - 1) * n_frames
    for t = 1, n_frames do row[t] = ref_mel[base + t] end
    mel[m] = row
end

local w = assert(weights.load(ckpt))
assert(w.config.d_model == d, "checkpoint/reference d_model mismatch")

-- WHY THIS IS NOT A PLAIN ABSOLUTE TOLERANCE
--
-- Whisper's encoder carries massive activations: layer 4 of tiny.en peaks
-- at |564|, with 52 of its 576,000 values above 50. At a value of -320 a
-- float32-level relative error of 7e-5 is an absolute error of 2e-2, so an
-- absolute threshold flags arithmetic noise as a failure -- and the giveaway
-- is that the MEAN error barely moves while the max jumps a thousandfold.
--
-- So this reports both, and gates the way numpy.allclose does:
--
--     |a - b|  <=  atol + rtol * |b|
--
-- Relative error is reported only for |ref| > 1, because relative error on
-- a value of 1e-6 is meaningless and would dominate the number.
local ATOL, RTOL = 5e-3, 1e-3

local worst, where = 0.0, ""
local failed_elems = 0

local function compare(label, got, want, n)
    local mx, sum, rel, rel_at = 0.0, 0.0, 0.0, 0.0
    local bad = 0
    for i = 1, n do
        local ref = want[i]
        local e = math.abs(got[i] - ref)
        if e > mx then mx = e end
        sum = sum + e
        local a = math.abs(ref)
        if a > 1.0 then
            local r = e / a
            if r > rel then rel, rel_at = r, ref end
        end
        if e > ATOL + RTOL * a then bad = bad + 1 end
    end
    if rel > worst then worst, where = rel, label end
    failed_elems = failed_elems + bad
    io.write(string.format(
        "  %-12s max %.3e   mean %.3e   rel %.3e (at %.1f)%s\n",
        label, mx, sum / n, rel, rel_at,
        bad > 0 and string.format("   %d over tolerance", bad) or ""))
    io.flush()
    return mx
end

local t0 = os.clock()
local checked_final = false

local x, outT = encoder.encode(w, mel, {
    max_layers = max_layers,
    on_conv = function(h, n)
        assert(n == T, string.format("conv gave T=%d, reference has %d", n, T))
        io.write(string.format("[%.0fs] conv stem\n", os.clock() - t0))
        compare("conv", h, ref_conv, d * T)
    end,
    on_layer = function(i, h)
        io.write(string.format("[%.0fs] layer %d\n", os.clock() - t0, i))
        compare("layer " .. i, h, ref_layer[i], T * d)
    end,
})

if not max_layers or max_layers >= n_layers then
    io.write(string.format("[%.0fs] final layer norm\n", os.clock() - t0))
    compare("final", x, ref_final, T * d)
    checked_final = true
end

io.write(string.format(
    "\nworst relative %.3e at %s   (%.0fs total)\n", worst, where, os.clock() - t0))
io.write(string.format("elements outside |a-b| <= %.0e + %.0e|b|: %d\n",
    ATOL, RTOL, failed_elems))

if failed_elems > 0 then
    io.write("FAIL\n")
    os.exit(1)
end
io.write(checked_final and "encoder parity passed\n"
                        or "stages checked so far agree\n")
