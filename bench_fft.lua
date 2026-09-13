-- bench_fft.lua - what the spectrogram actually costs.
--
--   lua54 bench_fft.lua
--
-- fft.lua uses a direct DFT rather than a mixed-radix or Bluestein FFT, on
-- the argument that the spectrogram is a small share of total runtime and
-- not worth several hundred lines of index arithmetic to speed up.
--
-- That argument is only worth making if the number is real, so this
-- measures it: one full 30-second clip is 3000 frames of 400 samples,
-- reduced to 201 bins each.

package.path = "./?.lua;" .. package.path
local fft = require('fft')

local N_FFT, HOP, N_BINS = 400, 160, 201
local SR, SECONDS = 16000, 30
local frames = math.floor((SR * SECONDS) / HOP)        -- 3000

io.write(string.format("planning twiddles for %d bins x %d samples...\n",
    N_BINS, N_FFT))
local t0 = os.clock()
local plan = fft.plan(N_FFT, N_BINS)
io.write(string.format("  plan built in %.2fs (%d floats, ~%.1f MB)\n",
    os.clock() - t0, 2 * N_BINS * N_FFT, 2 * N_BINS * N_FFT * 8 / 1e6))

-- A frame of something signal-shaped; the cost does not depend on content.
local window = fft.hann(N_FFT)
local frame = {}
for i = 1, N_FFT do
    frame[i] = math.sin(2 * math.pi * 440 * (i - 1) / SR) * window[i]
end

local out = {}
local warm = 20
for _ = 1, warm do fft.power(plan, frame, out) end

local reps = 200
t0 = os.clock()
for _ = 1, reps do fft.power(plan, frame, out) end
local per_frame = (os.clock() - t0) / reps

local clip = per_frame * frames
io.write(string.format("\n  per frame      %.3f ms\n", per_frame * 1000))
io.write(string.format("  30s clip       %.1f s  (%d frames)\n", clip, frames))
io.write(string.format("  effective      %.0f MFLOP/s\n",
    (2.0 * N_BINS * N_FFT * 2) / per_frame / 1e6))

-- The comparison that decides whether a faster transform is worth writing.
-- Encoder cost is 4 layers over 1500 positions at d_model 384, which works
-- out near 37 GFLOP; see the README for the breakdown.
local encoder_gflop = 37.0
local lua_gflops = 0.104                      -- measured end to end elsewhere
local encoder_s = encoder_gflop / lua_gflops
io.write(string.format("\n  encoder (est)  %.0f s\n", encoder_s))
io.write(string.format("  spectrogram is %.1f%% of the pipeline\n",
    100 * clip / (clip + encoder_s)))
