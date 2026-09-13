-- main.lua - WAV in, text out.
--
--   lua54 main.lua audio.wav
--   lua54 main.lua audio.wav --model whisper-tiny.en.lwb --max-tokens 60
--
-- Options:
--   --model <path>       checkpoint, default whisper-tiny.en.lwb
--   --tokenizer <path>   default tokenizer.lwt
--   --mel <path>         mel filterbank, default mel80.bin
--   --max-tokens <n>     default 64
--   --quiet              only print the transcript
--
-- The whole pipeline is minutes long, so progress is printed as it goes.
-- Roughly: five seconds of spectrogram, five minutes of encoder, then
-- under a second per token.

package.path = "./?.lua;" .. package.path

local audio     = require('audio')
local weights   = require('weights')
local encoder   = require('encoder')
local decoder   = require('decoder')
local tokenizer = require('tokenizer')

local function parse(argv)
    local o, i = { max_tokens = 64 }, 1
    while argv[i] do
        local a = argv[i]
        if a == "--quiet" then
            o.quiet = true; i = i + 1
        elseif a:sub(1, 2) == "--" then
            local k = a:sub(3):gsub("%-", "_")
            o[k] = argv[i + 1]
            if o[k] == nil then
                io.stderr:write("missing value for " .. a .. "\n")
                os.exit(2)
            end
            i = i + 2
        else
            o.wav = o.wav or a
            i = i + 1
        end
    end
    o.max_tokens = tonumber(o.max_tokens) or 64
    return o
end

local o = parse(arg)
if not o.wav then
    io.write("usage: lua54 main.lua audio.wav [--model f] [--tokenizer f] "
          .. "[--mel f] [--max-tokens n] [--quiet]\n")
    os.exit(2)
end

local function log(fmt, ...)
    if o.quiet then return end
    io.write(string.format(fmt, ...))
    io.flush()
end

local t_start = os.clock()

-- audio ------------------------------------------------------------------
local samples, info = audio.read_wav(o.wav)
if not samples then
    io.stderr:write(info .. "\n")
    os.exit(1)
end
if info.sample_rate ~= audio.SAMPLE_RATE then
    io.stderr:write(string.format(
        "%s is %d Hz; Whisper needs %d. Convert with:\n"
        .. "  ffmpeg -i %s -ar 16000 -ac 1 -c:a pcm_s16le out.wav\n",
        o.wav, info.sample_rate, audio.SAMPLE_RATE, o.wav))
    os.exit(1)
end
log("audio      %.1fs, %d Hz, %d channel(s)\n",
    info.seconds, info.sample_rate, info.channels)

local filters = assert(audio.load_mel_filters(o.mel or "mel80.bin"))
local tk = assert(tokenizer.load(o.tokenizer or "tokenizer.lwt"))
local w = assert(weights.load(o.model or "whisper-tiny.en.lwb"))

local t0 = os.clock()
-- Whisper always encodes a fixed 30-second window; shorter clips are
-- zero-padded and longer ones are cut.
local mel = audio.log_mel(audio.pad_or_trim(samples), filters)
log("mel        %.1fs\n", os.clock() - t0)

-- encoder ----------------------------------------------------------------
t0 = os.clock()
log("encoder    running (this is the slow part)...")
local enc_states, T = encoder.encode(w, mel)
log("\rencoder    %.0fs, %d positions\n", os.clock() - t0, T)

-- decoder ----------------------------------------------------------------
t0 = os.clock()
log("cross-attn precomputing keys/values...")
local cross = decoder.prepare_cross(w, enc_states, T)
log("\rcross-attn %.0fs (once, reused by every token)\n", os.clock() - t0)

local st = decoder.new_state(w, o.max_tokens + 4)

-- Whisper's prompt for an English-only model: start, then "no timestamps".
-- Multilingual checkpoints put language and task tokens in between.
local ids = { tk.sot, tk.no_timestamps }
local out = {}

t0 = os.clock()
local pos = 0
for i = 1, #ids do
    pos = pos + 1
    decoder.step(w, cross, st, ids[i], pos, { hidden_only = i < #ids })
end

-- The last prompt token's logits predict the first real token, so the loop
-- reads the result of the step already taken, then feeds it back.
local logits = decoder.step(w, cross, st, ids[#ids], pos)

for n = 1, o.max_tokens do
    -- begin_suppress_tokens: at the very first generated position Whisper
    -- also blocks a bare space and an immediate end-of-text, which would
    -- otherwise produce an empty transcript.
    local suppress = tk.suppress
    if n == 1 then
        suppress = setmetatable({ [220] = true, [tk.eot] = true },
                                { __index = tk.suppress })
    end

    local id = decoder.argmax(logits, suppress)
    if id == tk.eot then break end

    out[#out + 1] = id
    if not o.quiet then
        io.write(tk:bytes(id))
        io.flush()
    end

    pos = pos + 1
    if pos >= st.max then break end
    logits = decoder.step(w, cross, st, id, pos)
end

local text = tk:decode(out)
if o.quiet then
    io.write(text, "\n")
else
    log("\n\ntokens     %d in %.0fs (%.2fs each)\n",
        #out, os.clock() - t0, (os.clock() - t0) / math.max(1, #out))
    log("total      %.0fs\n", os.clock() - t_start)
end
