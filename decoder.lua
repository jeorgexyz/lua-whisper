-- decoder.lua - audio states in, one token's logits out.
--
-- Per layer, pre-norm throughout:
--
--   self-attention   causal, over the tokens generated so far
--   cross-attention  over the encoder's 1500 audio states
--   feed-forward
--
-- then a final LayerNorm and the tied lm_head.
--
-- TWO CACHES, DOING DIFFERENT JOBS
--
-- The self-attention cache is the familiar one from lua-llama: keys and
-- values for tokens already generated, appended one per step.
--
-- The cross-attention cache is the interesting one, and it is what makes
-- the decoder cheap. Its keys and values come from the ENCODER output,
-- which does not change while decoding -- so K and V for all 1500 audio
-- positions are computed once, before the first token, and reused for
-- every step after.
--
-- That is 1500 x 384 x 384 x 2 projections per layer: about 1.8 GFLOP
-- across four layers, roughly 17 seconds here. Recomputing it per token
-- would add that to every step and make a 50-token transcript take fifteen
-- minutes instead of thirty seconds.
--
-- WHAT DOMINATES A STEP
--
-- Not attention -- the lm_head. 384 x 51864 is 19.9M multiply-accumulates,
-- about 40 of the ~60 MFLOP a decode step costs. Everything else together
-- is a third of the work.

local encoder = require('encoder')

local decoder = {}

local gelu = encoder.gelu
local layernorm = encoder.layernorm

-- out = W x + b, W row-major [rows x cols], x read from xoff.
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

--------------------------------------------------------------------------
-- Cross-attention keys and values: computed once from the encoder output
--------------------------------------------------------------------------

-- enc_states is the encoder's time-major output, T x d.
function decoder.prepare_cross(w, enc_states, T)
    local c = w.config
    local d = c.d_model
    local out = {}

    local tmp = {}
    for li = 1, c.n_dec_layer do
        local a = w.dec.layers[li].cross
        local K, V = {}, {}
        for t = 1, T do
            local off = (t - 1) * d
            matvec(tmp, a.kw, nil, enc_states, off, d, d)   -- k_proj: no bias
            for i = 1, d do K[off + i] = tmp[i] end
            matvec(tmp, a.vw, a.vb, enc_states, off, d, d)
            for i = 1, d do V[off + i] = tmp[i] end
        end
        out[li] = { K = K, V = V }
    end
    out.T = T
    return out
end

--------------------------------------------------------------------------
-- Decode state
--------------------------------------------------------------------------

function decoder.new_state(w, max_tokens)
    local c = w.config
    local d = c.d_model
    max_tokens = max_tokens or c.n_text_ctx

    local layers = {}
    for li = 1, c.n_dec_layer do
        layers[li] = { K = {}, V = {} }     -- grown one token at a time
    end
    return {
        layers = layers,
        n = 0,                               -- tokens consumed so far
        max = max_tokens,
        x = {}, resid = {}, tmp = {}, h = {},
        scores = {}, ctx = {},
    }
end

--------------------------------------------------------------------------
-- One token
--------------------------------------------------------------------------

-- Causal self-attention over the cache, then append this token's K and V.
local function self_attention(w, L, st, li, d, n_heads, pos)
    local head_dim = d // n_heads
    local scale = 1.0 / math.sqrt(head_dim)
    local cache = st.layers[li]
    local x, tmp = st.x, st.tmp

    local q = {}
    matvec(q, L.attn.qw, L.attn.qb, x, 0, d, d)
    matvec(tmp, L.attn.kw, nil, x, 0, d, d)          -- k_proj: no bias
    local koff = (pos - 1) * d
    for i = 1, d do cache.K[koff + i] = tmp[i] end
    matvec(tmp, L.attn.vw, L.attn.vb, x, 0, d, d)
    for i = 1, d do cache.V[koff + i] = tmp[i] end

    local scores, ctx = st.scores, st.ctx
    for h = 1, n_heads do
        local hoff = (h - 1) * head_dim

        -- Causal by construction: only positions 1..pos exist in the cache.
        local hi = -math.huge
        for s = 1, pos do
            local ko = (s - 1) * d + hoff
            local acc = 0.0
            for i = 1, head_dim do acc = acc + q[hoff + i] * cache.K[ko + i] end
            acc = acc * scale
            scores[s] = acc
            if acc > hi then hi = acc end
        end

        local sum = 0.0
        for s = 1, pos do
            local e = math.exp(scores[s] - hi)
            scores[s] = e
            sum = sum + e
        end
        local inv = 1.0 / sum

        for i = 1, head_dim do ctx[hoff + i] = 0.0 end
        for s = 1, pos do
            local wgt = scores[s] * inv
            local vo = (s - 1) * d + hoff
            for i = 1, head_dim do
                ctx[hoff + i] = ctx[hoff + i] + wgt * cache.V[vo + i]
            end
        end
    end

    matvec(tmp, L.attn.ow, L.attn.ob, ctx, 0, d, d)
    for i = 1, d do x[i] = tmp[i] end
end

-- Cross-attention: queries from the token, keys and values from the audio.
local function cross_attention(w, L, st, cross, li, d, n_heads)
    local head_dim = d // n_heads
    local scale = 1.0 / math.sqrt(head_dim)
    local T = cross.T
    local K, V = cross[li].K, cross[li].V
    local x, tmp = st.x, st.tmp

    local q = {}
    matvec(q, L.cross.qw, L.cross.qb, x, 0, d, d)

    local scores, ctx = st.scores, st.ctx
    for h = 1, n_heads do
        local hoff = (h - 1) * head_dim

        local hi = -math.huge
        for s = 1, T do
            local ko = (s - 1) * d + hoff
            local acc = 0.0
            for i = 1, head_dim do acc = acc + q[hoff + i] * K[ko + i] end
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

        for i = 1, head_dim do ctx[hoff + i] = 0.0 end
        for s = 1, T do
            local wgt = scores[s] * inv
            local vo = (s - 1) * d + hoff
            for i = 1, head_dim do
                ctx[hoff + i] = ctx[hoff + i] + wgt * V[vo + i]
            end
        end
    end

    matvec(tmp, L.cross.ow, L.cross.ob, ctx, 0, d, d)
    for i = 1, d do x[i] = tmp[i] end
end

-- One decode step. `pos` is 1-based: the first token is pos 1.
--
-- Returns the logits table, reused across calls -- copy it if you need to
-- keep one.
function decoder.step(w, cross, st, token_id, pos, opts)
    opts = opts or {}
    local c = w.config
    local d, ff, n_heads = c.d_model, c.ffn_dim, c.n_dec_heads
    local x, resid, tmp, hbuf = st.x, st.resid, st.tmp, st.h

    -- token embedding + learned position
    local emb = token_id * d                    -- token ids are 0-based
    local poff = (pos - 1) * d
    for i = 1, d do
        x[i] = w.dec.tok_emb[emb + i] + w.dec.pos[poff + i]
    end

    for li = 1, c.n_dec_layer do
        local L = w.dec.layers[li]

        for i = 1, d do resid[i] = x[i] end
        layernorm(x, 0, d, L.attn_ln.w, L.attn_ln.b)
        self_attention(w, L, st, li, d, n_heads, pos)
        for i = 1, d do x[i] = resid[i] + x[i] end

        for i = 1, d do resid[i] = x[i] end
        layernorm(x, 0, d, L.cross_ln.w, L.cross_ln.b)
        cross_attention(w, L, st, cross, li, d, n_heads)
        for i = 1, d do x[i] = resid[i] + x[i] end

        for i = 1, d do resid[i] = x[i] end
        layernorm(x, 0, d, L.final_ln.w, L.final_ln.b)
        matvec(hbuf, L.mlp.w1, L.mlp.b1, x, 0, ff, d)
        for i = 1, ff do hbuf[i] = gelu(hbuf[i]) end
        matvec(tmp, L.mlp.w2, L.mlp.b2, hbuf, 0, d, ff)
        for i = 1, d do x[i] = resid[i] + tmp[i] end
    end

    layernorm(x, 0, d, w.dec.ln.w, w.dec.ln.b)

    if opts.hidden_only then
        st.n = pos
        return nil
    end

    -- Tied lm_head: the token embedding used as the output projection.
    -- Two thirds of a decode step is spent right here.
    local logits = st.logits
    if not logits then logits = {} st.logits = logits end
    local V = c.vocab_size
    for v = 1, V do
        local base = (v - 1) * d
        local acc = 0.0
        for i = 1, d do acc = acc + w.dec.lm_head[base + i] * x[i] end
        logits[v] = acc
    end

    st.n = pos
    return logits
end

-- Greedy argmax, with optional suppression.
--
-- `suppress` is a set of 0-based token ids that may never be emitted.
-- Whisper suppresses 90 of them always -- mostly punctuation-only and
-- special tokens -- and a couple more at the first generated position.
-- Without it the model will happily emit control tokens mid-sentence.
function decoder.argmax(logits, suppress)
    local best, best_v = 0, -math.huge
    for v = 1, #logits do
        if not (suppress and suppress[v - 1]) then
            local s = logits[v]
            if s > best_v then best_v, best = s, v - 1 end
        end
    end
    return best, best_v
end

return decoder
