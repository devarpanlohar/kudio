"""
sound_check.py -- Kudio Sound Device Diagnostic
Run: C:\python3-12\python.exe sound_check.py
"""
import sys
import io
import numpy as np

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

print("=" * 60)
print("  Kudio Sound Device Diagnostic")
print("=" * 60)

# ── 1. List all devices ───────────────────────────────────────────────────────
try:
    import sounddevice as sd
    print("\n[1] All available audio devices:\n")
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        tag = ""
        if i == sd.default.device[1]:  # default output
            tag = "  <-- DEFAULT OUTPUT"
        if d['max_output_channels'] > 0:
            print(f"  [{i:2}] {d['name']}"
                  f"  (out_ch={d['max_output_channels']}"
                  f"  sr={int(d['default_samplerate'])}){tag}")
    print()
except Exception as e:
    print(f"  FAIL: sounddevice not working: {e}")
    sys.exit(1)

# ── 2. Show default device ────────────────────────────────────────────────────
try:
    default_out = sd.query_devices(kind='output')
    print(f"[2] Default output device:")
    print(f"    Name     : {default_out['name']}")
    print(f"    Channels : {default_out['max_output_channels']}")
    print(f"    Sample rate: {int(default_out['default_samplerate'])} Hz")
    print()
except Exception as e:
    print(f"  FAIL querying default output: {e}\n")

# ── 3. Try playing a 1-second test tone ──────────────────────────────────────
print("[3] Playing 1-second 440 Hz test tone on DEFAULT device...")
try:
    SR   = 44100
    t    = np.linspace(0, 1.0, SR, dtype=np.float32)
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).reshape(-1, 1)
    tone = np.hstack([tone, tone])   # stereo
    sd.play(tone, SR, blocking=True)
    print("    >> Did you hear a beep? If YES, audio is working.")
    print("    >> If NO, your default device is wrong (see step 4).")
    print()
except Exception as e:
    print(f"    FAIL: {e}\n")

# ── 4. Try each output device until one works ─────────────────────────────────
print("[4] Trying each output device individually...")
SR   = 44100
t    = np.linspace(0, 0.5, SR // 2, dtype=np.float32)
tone = (0.25 * np.sin(2 * np.pi * 440 * t)).reshape(-1, 1)
tone = np.hstack([tone, tone])

worked = []
for i, d in enumerate(sd.query_devices()):
    if d['max_output_channels'] < 2:
        continue
    try:
        sd.play(tone, SR, device=i, blocking=True)
        print(f"    [WORKS] device {i}: {d['name']}")
        worked.append(i)
    except Exception as e:
        print(f"    [fail]  device {i}: {d['name']}  -> {e}")

print()
if worked:
    print(f"[5] Working device index(es): {worked}")
    print(f"    Add this to streaming.py in the sd.OutputStream() call:")
    print(f"    device={worked[0]}")
    print()
    print(f"    OR run this to set it as system default for Kudio:")
    print(f"    sd.default.device = {worked[0]}")
else:
    print("[5] No working output devices found!")
    print("    Check Windows Sound settings -> Playback devices")
    print("    Make sure your headphones/speakers are set as Default Device")

print("=" * 60)