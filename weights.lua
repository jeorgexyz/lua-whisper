-- weights.lua - read the flat checkpoint tools/export_whisper.py writes.
--
-- One sequential pass, no seeking: the file is a small header followed by
-- every tensor back to back in a fixed order. The order here mirrors the
-- exporter line for line, because the only thing keeping the two in step is
-- that they are easy to read side by side.
--
-- THE BIAS LAYOUT
--
-- q_proj, v_proj and out_proj have a bias; k_proj does not. That is true in
-- self-attention and cross-attention, encoder and decoder. Reading a bias
-- that was never written shifts every following tensor by 384 floats, and
-- the failure is not an error -- the model loads and transcribes nonsense.
--
-- So `attention()` below lists (name, has_bias) exactly as the exporter
-- does, and load() checks the bytes consumed against the file size. A
-- layout drift becomes a startup error instead of a bad transcript.

local weights = {}

local MAGIC = "LWB1"

--------------------------------------------------------------------------
-- Bulk reading
--------------------------------------------------------------------------

-- The whole file is read once into a string and then unpacked by offset.
-- 37.76M floats is 151 MB; reading it in 4-byte pieces through io.read
-- spends all its time in the io layer, and Lua strings handle a 151 MB
-- buffer without complaint.
local Reader = {}
Reader.__index = Reader

local function reader(path)
    local fh, err = io.open(path, "rb")
    if not fh then return nil, "cannot open " .. path .. ": " .. tostring(err) end
    local blob = fh:read("a")
    fh:close()
    return setmetatable({ blob = blob, pos = 1, floats = 0 }, Reader)
end

function Reader:i32()
    local v = string.unpack("<i4", self.blob, self.pos)
    self.pos = self.pos + 4
    return v
end

-- n floats as a flat 1-indexed table.
--
-- string.unpack is called per element rather than with a repeated format:
-- unpacking 147,456 values in one call returns 147,456 results on the Lua
-- stack, which overflows it. Per-element keeps memory flat and predictable.
function Reader:floats_n(n)
    local t = {}
    local blob, pos = self.blob, self.pos
    for i = 1, n do
        t[i] = string.unpack("<f", blob, pos)
        pos = pos + 4
    end
    self.pos = pos
    self.floats = self.floats + n
    return t
end

--------------------------------------------------------------------------
-- Layout, mirroring tools/export_whisper.py
--------------------------------------------------------------------------

-- Order and bias presence must match the exporter's attn(). Listing them as
-- data rather than four hand-written reads is what stops the two drifting.
local ATTN_PROJ = {
    { "q", true }, { "k", false }, { "v", true }, { "o", true },
}

local function attention(r, d)
    local a = {}
    for _, spec in ipairs(ATTN_PROJ) do
        local name, has_bias = spec[1], spec[2]
        a[name .. "w"] = r:floats_n(d * d)
        if has_bias then a[name .. "b"] = r:floats_n(d) end
    end
    return a
end

local function layernorm(r, d)
    return { w = r:floats_n(d), b = r:floats_n(d) }
end

local function mlp(r, d, ff)
    return {
        w1 = r:floats_n(ff * d), b1 = r:floats_n(ff),
        w2 = r:floats_n(d * ff), b2 = r:floats_n(d),
    }
end

function weights.load(path)
    local r, err = reader(path)
    if not r then return nil, err end

    if r.blob:sub(1, 4) ~= MAGIC then
        return nil, path .. " is not a lua-whisper checkpoint (bad magic)"
    end
    r.pos = 5

    local version = r:i32()
    if version ~= 1 then
        return nil, string.format("checkpoint version %d, expected 1", version)
    end

    local c = {
        n_mels      = r:i32(),
        n_audio_ctx = r:i32(),
        d_model     = r:i32(),
        n_heads     = r:i32(),
        n_enc_layer = r:i32(),
        vocab_size  = r:i32(),
        n_text_ctx  = r:i32(),
        n_dec_heads = r:i32(),
        n_dec_layer = r:i32(),
        ffn_dim     = r:i32(),
    }
    c.head_dim = c.d_model // c.n_heads

    local d, ff = c.d_model, c.ffn_dim
    local w = { config = c }

    -- encoder ------------------------------------------------------------
    -- Conv weights arrive as PyTorch [out, in, k], read flat and indexed
    -- directly rather than transposed here.
    w.enc = {
        conv1_w = r:floats_n(d * c.n_mels * 3),
        conv1_b = r:floats_n(d),
        conv2_w = r:floats_n(d * d * 3),
        conv2_b = r:floats_n(d),
        pos     = r:floats_n(c.n_audio_ctx * d),
        layers  = {},
    }
    for i = 1, c.n_enc_layer do
        w.enc.layers[i] = {
            attn     = attention(r, d),
            attn_ln  = layernorm(r, d),
            mlp      = mlp(r, d, ff),
            final_ln = layernorm(r, d),
        }
    end
    w.enc.ln = layernorm(r, d)

    -- decoder ------------------------------------------------------------
    w.dec = {
        tok_emb = r:floats_n(c.vocab_size * d),
        pos     = r:floats_n(c.n_text_ctx * d),
        layers  = {},
    }
    for i = 1, c.n_dec_layer do
        w.dec.layers[i] = {
            attn      = attention(r, d),
            attn_ln   = layernorm(r, d),
            cross     = attention(r, d),
            cross_ln  = layernorm(r, d),
            mlp       = mlp(r, d, ff),
            final_ln  = layernorm(r, d),
        }
    end
    w.dec.ln = layernorm(r, d)

    -- The lm_head is the token embedding; the exporter refuses to run on a
    -- checkpoint where that is not true, so sharing the table is safe.
    w.dec.lm_head = w.dec.tok_emb

    -- Consumed exactly the file? Anything left over means the reader and
    -- the writer disagree about the layout, which is the failure mode this
    -- whole file is arranged to make loud.
    local left = #r.blob - (r.pos - 1)
    if left ~= 0 then
        return nil, string.format(
            "layout mismatch: %d bytes %s after reading %d floats. "
            .. "The reader and tools/export_whisper.py disagree.",
            math.abs(left), left > 0 and "left over" or "short", r.floats)
    end

    w.n_floats = r.floats
    return w
end

return weights
