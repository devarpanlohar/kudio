# test_cpp.py -- run from Kudio folder
import numpy as np
import kudio_dsp as cpp

SR    = 44100
CHUNK = 1024
x     = np.random.randn(CHUNK, 2).astype(np.float32) * 0.1

def rms(a): return round(float(np.sqrt(np.mean(a**2))), 5)

print(f"Input              rms={rms(x)}")

s = cpp.SpectralShaper("cinema", SR)
out = s.process(x.copy()); print(f"SpectralShaper     rms={rms(out)}")

p = cpp.PsychoStereo(haas_ms=1.2, crossfeed=0.18, cf_delay_ms=0.35, ild_cutoff=1400, sr=SR)
out = p.process(x.copy()); print(f"PsychoStereo       rms={rms(out)}")

r = cpp.FDNReverb("hall", 0.25, 1.0, SR)
out = r.process(x.copy()); print(f"FDNReverb          rms={rms(out)}")

h = cpp.HarmonicExciter(drive=0.35, even_mix=0.60, odd_mix=0.40, wet=0.22, excite_hz=2000, sr=SR)
out = h.process(x.copy()); print(f"HarmonicExciter    rms={rms(out)}")

lim = cpp.TruePeakLimiter(-1.0, 3.0, 80.0, SR, CHUNK)
# Warm up limiter with a few chunks (it has look-ahead delay)
for _ in range(8): lim.process(x.copy())
out = lim.process(x.copy()); print(f"TruePeakLimiter    rms={rms(out)}")

print()
print("Full chain (10 warmup chunks then measure):")
r2   = cpp.FDNReverb("hall", 0.25, 1.0, SR)
lim2 = cpp.TruePeakLimiter(-1.0, 3.0, 80.0, SR, CHUNK)
h2   = cpp.HarmonicExciter(drive=0.35, even_mix=0.60, odd_mix=0.40, wet=0.22, excite_hz=2000, sr=SR)
s2   = cpp.SpectralShaper("cinema", SR)
p2   = cpp.PsychoStereo(haas_ms=1.2, crossfeed=0.18, cf_delay_ms=0.35, ild_cutoff=1400, sr=SR)
for _ in range(10):
    t = x.copy()
    t = s2.process(t); t = p2.process(t)
    t = r2.process(t); t = h2.process(t)
    t = lim2.process(t)
out = t
print(f"After 10 chunks    rms={rms(out)}")