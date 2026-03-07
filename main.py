"""
main.py -- Kudio Entry Point v3.1
"""

import signal
import sys
import time
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import importlib.util
if importlib.util.find_spec("kudio_dsp") is None:
    print(f"[Kudio] WARNING: C++ engine not found for Python {sys.version[:6]}")
    print(f"[Kudio]   Rebuild: cmake .. && cmake --build . --config Release")
    print()

from streaming import stream_audio, search_music, stop

_REVERB_MODES = {
    '1': ('hall',      'Concert Hall   -- RT60 2.6s, orchestral depth'),
    '2': ('cathedral', 'Cathedral      -- RT60 5.0s, stone / ethereal'),
    '3': ('plate',     'Vintage Plate  -- RT60 1.8s, dense & intimate'),
    '4': ('chamber',   'Live Chamber   -- RT60 1.1s, wood & warmth'),
}

_NORM_MODES = {
    '1': ('quiet',  'Quiet  -- -23 LUFS, full dynamics preserved'),
    '2': ('normal', 'Normal -- -14 LUFS, balanced (default)'),
    '3': ('loud',   'Loud   -- -11 LUFS, noisy environments'),
}

_EQ_BANDS = ['bass', 'low_mid', 'mid', 'upper_mid', 'presence', 'treble']
_EQ_FREQ  = ['60 Hz', '150 Hz', '400 Hz', '1 kHz', '2.4 kHz', '15 kHz']


def signal_handler(sig, frame):
    print("\n\nStopping Kudio...")
    stop()
    time.sleep(0.4)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


def _ask_eq():
    """Interactive 6-band EQ setup. Returns gains dict."""
    print("\n  6-Band EQ (press Enter to skip / keep flat):")
    print(f"  {'Band':<12} {'Freq':<8}  Gain dB")
    print(f"  {'-'*36}")
    gains = {}
    for band, freq in zip(_EQ_BANDS, _EQ_FREQ):
        try:
            raw = input(f"  {band:<12} {freq:<8}  ").strip()
            if raw:
                gains[band] = float(raw)
        except (ValueError, EOFError):
            pass
    return gains


def main():
    print(r"""
  _  __         _ _
 | |/ /   _  __| (_) ___
 | ' / | | |/ _` | |/ _ \
 | . \ |_| | (_| | | (_) |
 |_|\_\__,_|\__,_|_|\___/

  Cinema DSP Engine v3.1  [C++ Accelerated]
""")

    # ── Reverb mode ───────────────────────────────────────────────────────────
    print("  Select reverb character:")
    for k, (_, desc) in _REVERB_MODES.items():
        print(f"    [{k}] {desc}")
    print()
    choice = input("  Choice [1-4, default=1]: ").strip() or '1'
    reverb_mode = _REVERB_MODES.get(choice, _REVERB_MODES['1'])[0]
    print(f"  -> Reverb: {reverb_mode}")

    # ── Loudness mode ─────────────────────────────────────────────────────────
    print("\n  Select loudness target:")
    for k, (_, desc) in _NORM_MODES.items():
        print(f"    [{k}] {desc}")
    print()
    nchoice = input("  Choice [1-3, default=2]: ").strip() or '2'
    norm_mode = _NORM_MODES.get(nchoice, _NORM_MODES['2'])[0]
    print(f"  -> Loudness: {norm_mode}")

    # ── EQ ────────────────────────────────────────────────────────────────────
    eq_raw = input("\n  Customize EQ? [y/N]: ").strip().lower()
    eq_gains = _ask_eq() if eq_raw == 'y' else {}

    # ── Song search ───────────────────────────────────────────────────────────
    query = input("\n  Enter song name: ").strip() or "starboy weeknd"
    print(f"\n  Searching: {query}")

    title, url = search_music(query)
    if not url:
        print("[Kudio] Could not retrieve audio URL.")
        return

    print(f"  Now playing: {title}\n")
    stream_audio(query, url,
                 reverb_mode=reverb_mode,
                 norm_mode=norm_mode,
                 eq_gains=eq_gains)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("\n[Kudio] Fatal error:")
        traceback.print_exc()
        input("\nPress Enter to exit...")