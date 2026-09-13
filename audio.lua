-- audio.lua - WAV in, log-mel spectrogram out.
--
-- This reproduces Hugging Face's WhisperFeatureExtractor exactly. The model
-- was trained on these numbers, so "close enough" is not a thing: a
-- spectrogram that is subtly wrong produces subtly worse transcripts, and
-- nothing downstream will tell you which stage went wrong.
--
-- The pipeline, in the order HF applies it:
--
--   1. reflect-pad the waveform by n_fft/2 = 200 on each side
--   2. frame at hop 160, multiply by a PERIODIC Hann window
--   3. |rfft|^2  -> 201 power bins per frame
--   4. mel_filters^T @ power, floored at 1e-10   -> 80 mel bands
--   5. log10
--   6. DROP THE LAST FRAME
--   7. clamp to (max - 8.0)
--   8. (x + 4.0) / 4.0
--
-- Four of those eight steps are conventions that are easy to get wrong and
-- invisible when you do:
--
--   * PERIODIC Hann (divide by n, not n-1). torch.hann_window defaults to
--     periodic; the symmetric variant is a different window.
--   * REFLECT padding, not zeros. [a,b,c,d] by 2 becomes [c,b,a,b,c,d,c,b]
--     -- the edge sample is not repeated.
--   * The last frame is dropped AFTER the log, not before. Framing gives
--     3001 frames for 30 seconds; the model wants 3000.
--   * The clamp is relative to the maximum OF THIS CLIP, so it is not a
--     fixed floor -- two clips normalise differently, and that is intended.
--
-- The mel filterbank itself is not built here. It is a fixed 201x80 matrix
-- that tools/export_whisper.py writes out, so this file never has to
-- implement the mel scale, the triangular filter construction, or the
-- Slaney normalisation HF uses. Exporting a constant beats reimplementing
-- the code that generates it.

local fft = require('fft')

local audio = {}

audio.SAMPLE_RATE = 16000
audio.N_FFT       = 400
audio.HOP         = 160
audio.N_MELS      = 80
audio.N_BINS      = 201            -- N_FFT // 2 + 1
audio.CHUNK_SEC   = 30
audio.N_FRAMES    = 3000           -- what the encoder expects
audio.MEL_FLOOR   = 1e-10

--------------------------------------------------------------------------
-- WAV
--------------------------------------------------------------------------

local function read_u32(s, i) return string.unpack("<I4", s, i) end
local function read_u16(s, i) return string.unpack("<I2", s, i) end

-- Minimal RIFF/WAVE reader: 16-bit PCM, which is what every recorder and
-- every `ffmpeg -ar 16000 -ac 1` produces. Anything else is rejected with a
-- message naming the fix rather than a confusing downstream failure.
--
-- Chunks are walked rather than assumed to be in order: real files carry
-- LIST/INFO chunks between `fmt ` and `data`, and a reader that assumes a
-- 44-byte header reads metadata as audio.
function audio.read_wav(path)
    local fh, err = io.open(path, "rb")
    if not fh then return nil, "cannot open " .. path .. ": " .. tostring(err) end
    local s = fh:read("a")
    fh:close()

    if #s < 12 or s:sub(1, 4) ~= "RIFF" or s:sub(9, 12) ~= "WAVE" then
        return nil, path .. " is not a RIFF/WAVE file"
    end

    local pos = 13
    local channels, rate, bits, data_start, data_len
    while pos + 8 <= #s do
        local id = s:sub(pos, pos + 3)
        local size = read_u32(s, pos + 4)
        local body = pos + 8
        if id == "fmt " then
            local format = read_u16(s, body)
            channels = read_u16(s, body + 2)
            rate = read_u32(s, body + 4)
            bits = read_u16(s, body + 14)
            if format ~= 1 then
                return nil, string.format(
                    "%s is WAV format %d; only 16-bit PCM (format 1) is supported. "
                    .. "Convert with:  ffmpeg -i in -ar 16000 -ac 1 -c:a pcm_s16le out.wav",
                    path, format)
            end
        elseif id == "data" then
            data_start, data_len = body, size
            break
        end
        pos = body + size + (size % 2)      -- chunks are word-aligned
    end

    if not data_start then return nil, path .. " has no data chunk" end
    if bits ~= 16 then
        return nil, string.format(
            "%s is %d-bit; only 16-bit PCM is supported. "
            .. "Convert with:  ffmpeg -i in -ar 16000 -ac 1 -c:a pcm_s16le out.wav",
            path, bits)
    end

    -- int16 -> float in [-1, 1), matching what soundfile/librosa hand to
    -- the feature extractor. Dividing by 32768 (not 32767) is the
    -- convention; the asymmetry is deliberate.
    local n = math.min(data_len, #s - data_start + 1) // (2 * channels)
    local out = {}
    local step = 2 * channels
    for i = 0, n - 1 do
        if channels == 1 then
            out[i + 1] = string.unpack("<i2", s, data_start + i * step) / 32768.0
        else
            -- Downmix to mono by averaging, which is what ffmpeg -ac 1 does.
            local acc = 0
            for c = 0, channels - 1 do
                acc = acc + string.unpack("<i2", s, data_start + i * step + c * 2)
            end
            out[i + 1] = (acc / channels) / 32768.0
        end
    end

    return out, { sample_rate = rate, channels = channels, bits = bits,
                  samples = n, seconds = n / rate }
end

--------------------------------------------------------------------------
-- Mel filterbank
--------------------------------------------------------------------------

-- Flat float32, written by tools/export_whisper.py:
--   int32 n_bins, int32 n_mels, then n_bins*n_mels float32 (row-major)
function audio.load_mel_filters(path)
    local fh, err = io.open(path, "rb")
    if not fh then return nil, "cannot open " .. path .. ": " .. tostring(err) end
    local hdr = fh:read(8)
    if not hdr or #hdr ~= 8 then fh:close() return nil, "truncated mel filter file" end
    local n_bins, n_mels = string.unpack("<i4i4", hdr)
    local body = fh:read(n_bins * n_mels * 4)
    fh:close()
    if not body or #body ~= n_bins * n_mels * 4 then
        return nil, "truncated mel filter data"
    end
    local w = { n_bins = n_bins, n_mels = n_mels }
    for i = 1, n_bins * n_mels do
        w[i] = string.unpack("<f", body, (i - 1) * 4 + 1)
    end
    return w
end

--------------------------------------------------------------------------
-- Spectrogram
--------------------------------------------------------------------------

-- np.pad(x, pad, mode="reflect"): the edge sample is NOT repeated.
--   [a,b,c,d] with pad 2  ->  [c,b,a, b,c,d, c,b]
local function reflect_pad(x, pad)
    local n = #x
    local out = {}
    for i = 1, pad do out[i] = x[pad + 2 - i] end
    for i = 1, n do out[pad + i] = x[i] end
    for j = 1, pad do out[pad + n + j] = x[n - j] end
    return out
end

-- Pad or trim to exactly 30 seconds. Whisper always encodes a fixed window;
-- shorter audio is zero-padded, longer audio is cut.
function audio.pad_or_trim(samples, target)
    target = target or (audio.SAMPLE_RATE * audio.CHUNK_SEC)
    local n = #samples
    if n == target then return samples end
    local out = {}
    for i = 1, math.min(n, target) do out[i] = samples[i] end
    for i = n + 1, target do out[i] = 0.0 end
    return out
end

-- samples -> log-mel, returned as mel[m][t], 1-indexed, n_mels x n_frames.
--
-- `mel_filters` comes from load_mel_filters. `opts.frames` caps the output
-- (defaults to N_FRAMES); pass the full count to compare against a
-- reference that has not been trimmed.
function audio.log_mel(samples, mel_filters, opts)
    opts = opts or {}
    local n_fft, hop = audio.N_FFT, audio.HOP
    local n_bins, n_mels = audio.N_BINS, mel_filters.n_mels

    local padded = reflect_pad(samples, n_fft // 2)
    local n_frames = 1 + (#padded - n_fft) // hop

    local window = fft.hann(n_fft)
    local plan = opts.plan or fft.plan(n_fft, n_bins)

    -- mel[m][t]; built column by column so one power spectrum is reused.
    local mel = {}
    for m = 1, n_mels do mel[m] = {} end

    local frame, power = {}, {}
    local hi = -math.huge

    for t = 1, n_frames do
        local off = (t - 1) * hop
        for i = 1, n_fft do frame[i] = padded[off + i] * window[i] end
        fft.power(plan, frame, power)

        for m = 1, n_mels do
            local acc = 0.0
            -- mel_filters is (n_bins x n_mels) row-major, so column m is
            -- strided. Walking it this way keeps the export identical to
            -- the numpy array and avoids a transpose at load time.
            local idx = m
            for b = 1, n_bins do
                acc = acc + power[b] * mel_filters[idx]
                idx = idx + n_mels
            end
            if acc < audio.MEL_FLOOR then acc = audio.MEL_FLOOR end
            local v = math.log(acc, 10)
            mel[m][t] = v
            if v > hi then hi = v end
        end
    end

    -- HF drops the final frame AFTER the log and BEFORE the clamp. The
    -- maximum used by the clamp is therefore taken over the kept frames.
    local keep = opts.frames or (n_frames - 1)
    if keep > n_frames then keep = n_frames end

    hi = -math.huge
    for m = 1, n_mels do
        for t = 1, keep do
            if mel[m][t] > hi then hi = mel[m][t] end
        end
    end

    local floor = hi - 8.0
    for m = 1, n_mels do
        local row = mel[m]
        for t = n_frames, keep + 1, -1 do row[t] = nil end
        for t = 1, keep do
            local v = row[t]
            if v < floor then v = floor end
            row[t] = (v + 4.0) / 4.0
        end
    end

    return mel, keep
end

return audio
