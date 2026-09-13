-- encoder.lua - mel spectrogram in, 1500x384 audio states out.
--
--   conv1(80 -> 384, k3 s1 p1)  -> gelu
--   conv2(384 -> 384, k3 s2 p1) -> gelu        3000 frames become 1500
--   + learned positional embedding (all 1500)
--   4 x pre-norm transformer layer
--   final layer norm
--
-- Differences from the llama-family code in the sibling repos, all of them
-- simplifications except the first:
--
--   LayerNorm with bias, not RMSNorm          (eps 1e-5)
--   learned absolute positions, not RoPE      (just an add)
--   GELU, not SwiGLU                          (see below)
--   no causal mask -- audio attention is bidirectional
--   no KV cache -- the encoder runs once over a fixed 1500 positions
--
-- THE GELU IS THE EXACT ONE
--
-- torch's F.gelu defaults to the erf form, not the tanh approximation, and
-- Whisper was trained that way. They are not interchangeable here:
--
--   x = -3    exact -0.00404987    tanh -0.00363743
--
-- That is a 4e-4 gap, larger than the parity tolerance this project holds
-- itself to, so the tanh shortcut would fail the check outright. Lua has no
-- erf, so one is written below.

local encoder = {}

--------------------------------------------------------------------------
-- Activation
--------------------------------------------------------------------------

-- Abramowitz & Stegun 7.1.26. Maximum absolute error 1.5e-7, which is
-- float32 noise and two to three orders below the tolerance that matters
-- downstream. tools/check_encoder.py measures the resulting GELU error
-- rather than trusting this paragraph.
local A1, A2, A3 = 0.254829592, -0.284496736, 1.421413741
local A4, A5, P = -1.453152027, 1.061405429, 0.3275911

local function erf(x)
    local sign = 1
    if x < 0 then sign, x = -1, -x end
    local t = 1.0 / (1.0 + P * x)
    local y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * math.exp(-x * x)
    return sign * y
end

encoder.erf = erf

local SQRT1_2 = 1.0 / math.sqrt(2.0)

local function gelu(x)
    return 0.5 * x * (1.0 + erf(x * SQRT1_2))
end

encoder.gelu = gelu

--------------------------------------------------------------------------
-- Pieces
--------------------------------------------------------------------------

-- LayerNorm over one d-length slice of a flat array, in place.
-- Whisper uses the biased variance (divide by n), which is what
-- torch.nn.LayerNorm does.
local function layernorm(x, off, d, w, b, eps)
    local mean = 0.0
    for i = 1, d do mean = mean + x[off + i] end
    mean = mean / d

    local var = 0.0
    for i = 1, d do
        local v = x[off + i] - mean
        var = var + v * v
    end
    var = var / d

    local inv = 1.0 / math.sqrt(var + (eps or 1e-5))
    for i = 1, d do
        x[off + i] = (x[off + i] - mean) * inv * w[i] + b[i]
    end
end

encoder.layernorm = layernorm

-- out = W x + b, with W stored row-major [rows x cols].
local function matvec(out, W, b, x, xoff, rows, cols)
    for r = 1, rows do
        local acc = b and b[r] or 0.0
        local base = (r - 1) * cols
        for c = 1, cols do
            acc = acc + W[base + c] * x[xoff + c]
        end
        out[r] = acc
    end
    return out
end

-- Conv1d over channel-major data: input[(c-1)*n_in + t].
--
-- Whisper's two convolutions are the only place the time axis changes
-- length: conv2's stride of 2 is what turns 3000 mel frames into the 1500
-- positions the encoder's positional embedding expects.
local function conv1d(input, W, bias, c_in, c_out, n_in, k, stride, pad)
    local n_out = (n_in + 2 * pad - k) // stride + 1
    local out = {}
    for co = 1, c_out do
        local obase = (co - 1) * n_out
        local wbase = (co - 1) * c_in * k
        for t = 1, n_out do
            local acc = bias[co]
            local start = (t - 1) * stride - pad          -- 0-based input index
            for ci = 1, c_in do
                local ibase = (ci - 1) * n_in
                local wb = wbase + (ci - 1) * k
                for j = 1, k do
                    local ti = start + j                   -- 1-based
                    if ti >= 1 and ti <= n_in then
                        acc = acc + W[wb + j] * input[ibase + ti]
                    end
                end
            end
            out[obase + t] = acc
        end
    end
    return out, n_out
end

encoder.conv1d = conv1d

--------------------------------------------------------------------------
-- Self-attention, bidirectional
--------------------------------------------------------------------------

-- x is time-major flat: token t occupies x[(t-1)*d + 1 .. t*d].
--
-- Q, K and V are computed for every position up front (the encoder sees
-- the whole clip at once), then attention runs head by head with a scores
-- buffer of length T. Materialising the full T x T matrix for all heads
-- would be 1500*1500*6 floats -- 13.5M -- for no benefit.
local function self_attention(x, T, d, a, n_heads, scratch)
    local head_dim = d // n_heads
    local scale = 1.0 / math.sqrt(head_dim)

    local Q, K, V = scratch.Q, scratch.K, scratch.V
    local tmp = scratch.tmp

    for t = 1, T do
        local off = (t - 1) * d
        matvec(tmp, a.qw, a.qb, x, off, d, d)
        for i = 1, d do Q[off + i] = tmp[i] end
        matvec(tmp, a.kw, nil, x, off, d, d)      -- k_proj has no bias
        for i = 1, d do K[off + i] = tmp[i] end
        matvec(tmp, a.vw, a.vb, x, off, d, d)
        for i = 1, d do V[off + i] = tmp[i] end
    end

    local scores, ctx = scratch.scores, scratch.ctx
    for h = 1, n_heads do
        local hoff = (h - 1) * head_dim
        for t = 1, T do
            local qoff = (t - 1) * d + hoff

            local hi = -math.huge
            for s = 1, T do
                local koff = (s - 1) * d + hoff
                local acc = 0.0
                for i = 1, head_dim do
                    acc = acc + Q[qoff + i] * K[koff + i]
                end
                acc = acc * scale
                scores[s] = acc
                if acc > hi then hi = acc end
            end

            local sum = 0.0
            for s = 1, T do
                local e = math.exp(scores[s] - hi)
                scores[s] = e
                sum = sum + e
            end
            local inv = 1.0 / sum

            local cbase = (t - 1) * d + hoff
            for i = 1, head_dim do ctx[cbase + i] = 0.0 end
            for s = 1, T do
                local wgt = scores[s] * inv
                if wgt > 0 then
                    local voff = (s - 1) * d + hoff
                    for i = 1, head_dim do
                        ctx[cbase + i] = ctx[cbase + i] + wgt * V[voff + i]
                    end
                end
            end
        end
    end

    -- out_proj, written back into x
    for t = 1, T do
        local off = (t - 1) * d
        matvec(tmp, a.ow, a.ob, ctx, off, d, d)
        for i = 1, d do x[off + i] = tmp[i] end
    end
end

--------------------------------------------------------------------------
-- Encoder
--------------------------------------------------------------------------

local function new_scratch(T, d, ff)
    local function zeros(n)
        local t = {}
        for i = 1, n do t[i] = 0.0 end
        return t
    end
    return {
        Q = zeros(T * d), K = zeros(T * d), V = zeros(T * d),
        ctx = zeros(T * d), scores = zeros(T),
        tmp = zeros(math.max(d, ff)), h = zeros(ff), resid = zeros(T * d),
    }
end

-- mel is mel[m][t] as audio.lua returns it.
--
-- opts.max_layers stops early, and opts.on_layer(i, x) sees the states
-- after each layer. Both exist because a full pass is minutes long, and
-- validating layer by layer beats waiting for a transcript to look wrong.
function encoder.encode(w, mel, opts)
    opts = opts or {}
    local c = w.config
    local d, ff, n_heads = c.d_model, c.ffn_dim, c.n_heads
    local n_mels = c.n_mels

    -- mel[m][t] -> channel-major flat, which is what conv1d wants
    local n_frames = #mel[1]
    local flat = {}
    for m = 1, n_mels do
        local base = (m - 1) * n_frames
        local row = mel[m]
        for t = 1, n_frames do flat[base + t] = row[t] end
    end

    local h1, n1 = conv1d(flat, w.enc.conv1_w, w.enc.conv1_b,
                          n_mels, d, n_frames, 3, 1, 1)
    for i = 1, #h1 do h1[i] = gelu(h1[i]) end

    local h2, T = conv1d(h1, w.enc.conv2_w, w.enc.conv2_b,
                         d, d, n1, 3, 2, 1)
    for i = 1, #h2 do h2[i] = gelu(h2[i]) end
    h1 = nil

    if opts.on_conv then opts.on_conv(h2, T) end

    -- channel-major [c][t] -> time-major [t][c], and add positions.
    --
    -- The positional embedding is sliced to T rather than assumed to be
    -- 1500: a shorter clip uses the first T rows, which is how whisper.cpp
    -- avoids encoding 30 seconds of silence. HF always pads to 1500, so a
    -- parity check has to feed a full-length mel.
    local x = {}
    for t = 1, T do
        local off = (t - 1) * d
        for ch = 1, d do
            x[off + ch] = h2[(ch - 1) * T + t] + w.enc.pos[off + ch]
        end
    end
    h2 = nil

    local scratch = new_scratch(T, d, ff)
    local resid, hbuf, tmp = scratch.resid, scratch.h, scratch.tmp

    local n_layers = math.min(opts.max_layers or c.n_enc_layer, c.n_enc_layer)
    for li = 1, n_layers do
        local L = w.enc.layers[li]

        -- pre-norm self-attention
        for i = 1, T * d do resid[i] = x[i] end
        for t = 1, T do
            layernorm(x, (t - 1) * d, d, L.attn_ln.w, L.attn_ln.b)
        end
        self_attention(x, T, d, L.attn, n_heads, scratch)
        for i = 1, T * d do x[i] = resid[i] + x[i] end

        -- pre-norm feed-forward
        for i = 1, T * d do resid[i] = x[i] end
        for t = 1, T do
            local off = (t - 1) * d
            layernorm(x, off, d, L.final_ln.w, L.final_ln.b)
            matvec(hbuf, L.mlp.w1, L.mlp.b1, x, off, ff, d)
            for i = 1, ff do hbuf[i] = gelu(hbuf[i]) end
            matvec(tmp, L.mlp.w2, L.mlp.b2, hbuf, 0, d, ff)
            for i = 1, d do x[off + i] = tmp[i] end
        end
        for i = 1, T * d do x[i] = resid[i] + x[i] end

        if opts.on_layer then opts.on_layer(li, x, T) end
    end

    if n_layers == c.n_enc_layer then
        for t = 1, T do
            layernorm(x, (t - 1) * d, d, w.enc.ln.w, w.enc.ln.b)
        end
    end

    return x, T
end

return encoder
