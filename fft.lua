-- fft.lua - the discrete Fourier transform behind Whisper's spectrogram.
--
-- Whisper's front end is a short-time Fourier transform with n_fft = 400,
-- hop 160, over 16 kHz audio. 400 is NOT a power of two, so the textbook
-- radix-2 FFT does not apply, and zero-padding to 512 is not a fix -- it is
-- a different transform, with bins at different frequencies.
--
-- The three real options:
--
--   mixed radix   400 = 2^4 * 5^2, so a radix-2/radix-5 hybrid works.
--                 Correct and fast, and a few hundred lines of index
--                 arithmetic that nobody reads twice.
--   Bluestein     arbitrary N via a power-of-two convolution. Correct,
--                 fast, and genuinely hard to follow.
--   direct DFT    the definition, with the twiddle factors precomputed.
--                 Twenty lines. Obviously correct by inspection.
--
-- This takes the third, and the reason is arithmetic rather than taste.
-- Whisper only needs the first 201 bins (the real-input spectrum of a
-- 400-point transform), so one frame is 201 * 400 multiply-accumulates
-- against cos and sin, and a 30-second clip is 3000 frames.
--
-- Measured, not estimated (bench_fft.lua):
--
--   per frame        1.5 ms
--   full 30s clip    4.5 s
--   share of the pipeline   1.2%
--
-- The encoder that consumes this spectrogram is ~37 GFLOP, about six
-- minutes in pure Lua. Bluestein could save at most four seconds of that.
-- Several hundred lines of index arithmetic to move a 1.2% number would be
-- a bad trade anywhere, and a worse one in a repo whose whole point is
-- being readable.
--
-- Correctness is checked against numpy.fft.rfft, not asserted: see
-- tools/check_fft.py.

local fft = {}

-- Periodic Hann window, which is what torch.hann_window gives by default
-- and therefore what Whisper was trained with. The symmetric variant
-- (dividing by n-1) is a different window and quietly shifts every
-- spectrogram value.
function fft.hann(n)
    local w = {}
    for i = 0, n - 1 do
        w[i + 1] = 0.5 - 0.5 * math.cos(2 * math.pi * i / n)
    end
    return w
end

-- Precompute the twiddle factors once.
--
-- Without this, every frame recomputes 201*400 calls to math.cos and
-- math.sin -- transcendentals in the inner loop, which dominates everything
-- else. With it the inner loop is multiply-add over a flat array, which is
-- the one thing Lua does at a reasonable rate.
--
-- The table is n_bins * n_fft entries for each of cos and sin: at 201x400
-- that is 80,400 floats each, about 1.3 MB together. Paid once.
function fft.plan(n_fft, n_bins)
    n_bins = n_bins or (n_fft // 2 + 1)
    local cos_t, sin_t = {}, {}
    local two_pi_over_n = 2 * math.pi / n_fft
    for k = 0, n_bins - 1 do
        local base = k * n_fft
        for t = 0, n_fft - 1 do
            local ang = two_pi_over_n * k * t
            cos_t[base + t + 1] = math.cos(ang)
            sin_t[base + t + 1] = math.sin(ang)
        end
    end
    return { n_fft = n_fft, n_bins = n_bins, cos = cos_t, sin = sin_t }
end

-- Power spectrum |X(k)|^2 of one real frame, for k = 0 .. n_bins-1.
--
-- Whisper wants magnitudes squared, so the square root is never taken --
-- one less pass over 201 values per frame, 3000 frames per clip.
--
-- `frame` is 1-indexed with n_fft samples; `out` is filled in place so the
-- caller can reuse one table across every frame instead of allocating
-- 3000 of them.
function fft.power(plan, frame, out)
    local n_fft, n_bins = plan.n_fft, plan.n_bins
    local cos_t, sin_t = plan.cos, plan.sin
    out = out or {}
    for k = 0, n_bins - 1 do
        local base = k * n_fft
        local re, im = 0.0, 0.0
        for t = 1, n_fft do
            local x = frame[t]
            local idx = base + t
            re = re + x * cos_t[idx]
            im = im - x * sin_t[idx]
        end
        out[k + 1] = re * re + im * im
    end
    return out
end

-- Complex spectrum, for checking against a reference that reports real and
-- imaginary parts. Not on the Whisper path; power() is.
function fft.rfft(plan, frame, re_out, im_out)
    local n_fft, n_bins = plan.n_fft, plan.n_bins
    local cos_t, sin_t = plan.cos, plan.sin
    re_out, im_out = re_out or {}, im_out or {}
    for k = 0, n_bins - 1 do
        local base = k * n_fft
        local re, im = 0.0, 0.0
        for t = 1, n_fft do
            local x = frame[t]
            re = re + x * cos_t[base + t]
            im = im - x * sin_t[base + t]
        end
        re_out[k + 1], im_out[k + 1] = re, im
    end
    return re_out, im_out
end

return fft
