import sys
import os
import traceback
import subprocess
import time
import queue

# Force current directory into path
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_header(title):
    print(f"\n{'='*60}\n> {title}\n{'='*60}")

def run_diagnostics():
    all_ok = True
    
    print_header("1. SYSTEM & ENVIRONMENT")
    print(f"OS: {sys.platform}")
    print(f"Python: {sys.version.split()[0]}")
    
    print_header("2. IMPORT CHECKS")
    modules = [
        ("numpy", "import numpy as np"),
        ("sounddevice", "import sounddevice as sd"),
        ("kudio_dsp", "import kudio_dsp"),
        ("dsp_chain", "from dsp_chain import CinemaDSP"),
    ]
    
    for name, stmt in modules:
        try:
            exec(stmt)
            print(f"  [OK] {name}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            all_ok = False

    print_header("3. EXTERNAL DEPENDENCIES")
    # Check FFmpeg
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if r.returncode == 0:
            version_line = r.stdout.splitlines()[0]
            print(f"  [OK] FFmpeg found: {version_line[:40]}...")
        else:
            print(f"  [FAIL] FFmpeg returned error code {r.returncode}")
            all_ok = False
    except FileNotFoundError:
        print("  [FAIL] FFmpeg is NOT installed or NOT in system PATH.")
        all_ok = False

    # Check yt-dlp
    try:
        r = subprocess.run([sys.executable, "-m", "yt_dlp", "--version"], capture_output=True, text=True)
        if r.returncode == 0:
            print(f"  [OK] yt-dlp found: v{r.stdout.strip()}")
        else:
            print("  [FAIL] yt-dlp check failed.")
            all_ok = False
    except Exception as e:
        print(f"  [FAIL] yt-dlp error: {e}")
        all_ok = False

    if not all_ok:
        print("\n[!] Basic dependencies failed. Stopping tests here.")
        return

    print_header("4. AUDIO HARDWARE TEST")
    try:
        import sounddevice as sd
        import numpy as np
        
        SR = 44100
        CHUNK = 1024
        
        print(f"  Default output device: {sd.query_devices(sd.default.device[1])['name']}")
        print("  Attempting to open audio stream (44.1kHz, 2ch, float32)...")
        
        # Try writing a second of pure silence to see if the driver crashes
        with sd.OutputStream(samplerate=SR, channels=2, dtype='float32', blocksize=CHUNK, latency='high') as stream:
            silence = np.zeros((CHUNK, 2), dtype=np.float32)
            for _ in range(43): # roughly 1 second
                stream.write(silence)
        print("  [OK] Audio stream opened and closed successfully. Driver is happy.")
    except Exception as e:
        print(f"  [FAIL] Audio Hardware Error: {e}")
        traceback.print_exc()
        all_ok = False

    print_header("5. DSP ENGINE TEST")
    try:
        from dsp_chain import CinemaDSP
        dsp = CinemaDSP(reverb_mode="hall", sr=44100)
        test_audio = np.random.uniform(-0.1, 0.1, (1024, 2)).astype(np.float32)
        out = dsp.process(test_audio)
        if out.shape == test_audio.shape:
            print("  [OK] DSP Chain processed fake audio successfully.")
        else:
            print(f"  [FAIL] DSP Output shape mismatch: expected {test_audio.shape}, got {out.shape}")
            all_ok = False
    except Exception as e:
        print(f"  [FAIL] DSP Engine Error: {e}")
        traceback.print_exc()
        all_ok = False

    print_header("6. FFMPEG PIPE TEST (SIMULATION)")
    try:
        # We will make ffmpeg generate 2 seconds of a test sine wave internally
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2", 
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ar", "44100", "-ac", "2",
            "pipe:1"
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        
        bytes_read = 0
        read_size = 1024 * 2 * 4 # One chunk
        
        while True:
            data = proc.stdout.read(read_size)
            if not data:
                break
            bytes_read += len(data)
            
        proc.wait()
        
        if bytes_read > 0:
            print(f"  [OK] FFmpeg pipe works! Read {bytes_read} bytes of raw audio.")
        else:
            print("  [FAIL] FFmpeg pipe returned 0 bytes.")
            all_ok = False
            
    except Exception as e:
        print(f"  [FAIL] FFmpeg Pipe Error: {e}")
        traceback.print_exc()
        all_ok = False

    print_header("DIAGNOSTIC COMPLETE")
    if all_ok:
        print("[SUCCESS] All isolated systems are functioning perfectly.")
        print("If streaming.py still stalls, the issue is 100% in the live thread/queue logic or yt-dlp stream URL resolution.")
    else:
        print("[ERROR] One or more systems failed. See logs above.")

if __name__ == "__main__":
    run_diagnostics()