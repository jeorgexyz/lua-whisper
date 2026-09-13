-- tokenizer.lua - token ids to text.
--
-- DECODE ONLY
--
-- Transcription is audio -> tokens -> text. Nothing in that chain turns
-- text into tokens, so this carries no merge table and no BPE loop -- just
-- id to token string, and the byte-level mapping that turns those strings
-- back into bytes.
--
-- That leaves out the error-prone half of a GPT-2 tokenizer. Encoding would
-- need ~50k merge ranks applied in priority order plus a pre-tokenizer
-- regex with Unicode category classes, all of which must match exactly or
-- words split differently. Reading the model's output needs none of it.
--
-- THE BYTE-LEVEL MAPPING
--
-- GPT-2 does not put raw bytes in its vocabulary, because byte 32 is a
-- space and a space cannot appear in a whitespace-delimited vocab file.
-- Every byte is mapped to a printable codepoint first, so the token for
-- " the" is not " the" -- it is U+0120 followed by "the". Decoding maps
-- each codepoint back to a byte and then reads the result as UTF-8.
--
-- Getting this wrong is quiet rather than loud: the transcript comes out
-- with missing spaces, or with "Ġ" sprinkled through it.

local tokenizer = {}

local Tok = {}
Tok.__index = Tok

function tokenizer.load(path)
    local fh, err = io.open(path, "rb")
    if not fh then return nil, "cannot open " .. path .. ": " .. tostring(err) end
    local blob = fh:read("a")
    fh:close()

    if blob:sub(1, 4) ~= "LWT1" then
        return nil, path .. " is not a lua-whisper tokenizer file"
    end

    local pos = 5
    local function i32()
        local v = string.unpack("<i4", blob, pos)
        pos = pos + 4
        return v
    end

    local version = i32()
    if version ~= 1 then
        return nil, string.format("tokenizer version %d, expected 1", version)
    end

    local n = i32()
    local self = setmetatable({
        n = n,
        sot = i32(),
        no_timestamps = i32(),
        eot = i32(),
        timestamp_begin = i32(),
    }, Tok)

    local n_suppress = i32()
    self.suppress = {}
    for _ = 1, n_suppress do
        self.suppress[i32()] = true
    end

    -- codepoint -> byte, the inverse of GPT-2's byte encoder
    self.byte_of = {}
    for b = 0, 255 do
        self.byte_of[i32()] = b
    end

    self.tokens = {}
    self.special = {}
    for i = 0, n - 1 do
        local len = i32()
        self.tokens[i] = blob:sub(pos, pos + len - 1)
        pos = pos + len
        self.special[i] = i32() == 1
    end

    return self
end

-- One token's raw byte string.
--
-- The stored string is UTF-8 holding the mapped codepoints, so each
-- codepoint is walked and turned back into a single byte.
function Tok:bytes(id)
    local s = self.tokens[id]
    if not s then return "" end
    local out = {}
    for _, cp in utf8.codes(s) do
        local b = self.byte_of[cp]
        -- A codepoint outside the mapping means a special token like
        -- <|endoftext|>, whose text is literal rather than byte-encoded.
        out[#out + 1] = b and string.char(b) or utf8.char(cp)
    end
    return table.concat(out)
end

-- ids -> text.
--
-- Special tokens are dropped by default: the transcript should not contain
-- <|startoftranscript|>. Pass keep_special to see them, which is what you
-- want while debugging a decode loop.
function Tok:decode(ids, keep_special)
    local out = {}
    for _, id in ipairs(ids) do
        if keep_special or not self.special[id] then
            out[#out + 1] = self:bytes(id)
        end
    end
    return table.concat(out)
end

-- Is this id a timestamp token, and what time does it mark?
--
-- Whisper emits <|0.00|> through <|30.00|> in 0.02s steps as a contiguous
-- run at the end of the vocabulary. Useful for segmenting; ignored when
-- decoding with <|notimestamps|>.
function Tok:timestamp(id)
    if self.timestamp_begin < 0 or id < self.timestamp_begin then return nil end
    return (id - self.timestamp_begin) * 0.02
end

return tokenizer
