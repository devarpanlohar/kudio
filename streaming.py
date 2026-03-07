"""
streaming.py -- Kudio Streaming Engine
"""

import subprocess
import json
import sys
import threading
import queue
import time

import sounddevice as sd
import numpy as np

_YTDLP = [sys.executable, '-m', 'yt_dlp']

from dsp_chain import CinemaDSP

SR          = 44100
CHUNK       = 1024
BYTES       = CHUNK * 2 * 4    # 8192 bytes per stereo float32 chunk
MAX_RETRIES = 4

_FORMAT_PREF = (
    "bestaudio[ext=ogg]/bestaudio[ext=opus]/"
    "bestaudio[ext=webm]/bestaudio[ext=m4a]/"
    "bestaudio/best"
)

_playing     = True
_ffmpeg_proc = None


# ── Search ────────────────────────────────────────────────────────────────────

def search_music(query: str):
    """Primary: YouTube Music. Fallback: YouTube."""
    for extractor in ("ytmsearch", "ytsearch"):
        r = subprocess.run(
            _YTDLP + [
                f"{extractor}1:{query}",
                "--print", "%(id)s|||%(title)s|||%(uploader)s|||%(webpage_url)s",
                "--no-playlist", "--no-warnings", "--quiet",
            ],
            capture_output=True, text=True
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = r.stdout.strip().splitlines()[0].split("|||", 3)
            if len(parts) >= 4 and parts[3].strip().startswith("http"):
                return f"{parts[1].strip()} -- {parts[2].strip()}", parts[3].strip()

        r2 = subprocess.run(
            _YTDLP + [f"{extractor}1:{query}", "-j",
                      "--no-playlist", "--no-warnings"],
            capture_output=True, text=True
        )
        if r2.returncode == 0 and r2.stdout.strip():
            info  = json.loads(r2.stdout.strip().splitlines()[0])
            title = f"{info.get('title','?')} -- {info.get('uploader','?')}"
            url   = (info.get("webpage_url")
                     or f"https://music.youtube.com/watch?v={info['id']}")
            return title, url

    print("[yt-dlp] search failed for:", query)
    return None, None


def _get_stream_info(page_url: str):
    """Get direct CDN URL + HTTP headers via yt-dlp JSON."""
    r = subprocess.run(
        _YTDLP + [
            page_url,
            "-f", _FORMAT_PREF,
            "--extractor-args", "youtube:player_client=android,web",
            "-j", "--no-playlist", "--quiet",
        ],
        capture_output=True, text=True
    )
    if r.returncode == 0 and r.stdout.strip():
        try:
            info = json.loads(r.stdout.strip().splitlines()[0])
            url  = info.get("url", "")
            hdrs = info.get("http_headers", {})
            if url.startswith("http"):
                return url, hdrs
        except Exception:
            pass
    print("[yt-dlp] stream info failed:", r.stderr.strip()[:200])
    return None, {}


def _open_ffmpeg(stream_url: str, headers: dict):
    CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

    headers_args = []
    if headers:
        h_str = "".join(f"{k}: {v}\r\n" for k, v in headers.items())
        headers_args = ["-headers", h_str]

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-reconnect", "1",
        "-reconnect_streamed", "1",
        "-reconnect_delay_max", "5",
    ] + headers_args + [
        "-i", stream_url,
        "-vn",
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ar", str(SR), "-ac", "2",
        "pipe:1",
    ]

    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        creationflags=CREATE_NO_WINDOW
    )


def _kill(proc):
    if proc is None:
        return
    try: proc.terminate()
    except Exception: pass
    try: proc.wait(timeout=2)
    except Exception:
        try: proc.kill()
        except Exception: pass
    try: proc.stdout.close()
    except Exception: pass


# ── Main stream function ──────────────────────────────────────────────────────

def stream_audio(query: str, url: str, reverb_mode: str = "hall",
                 norm_mode: str = "normal", eq_gains: dict = None) -> None:
    global _playing, _ffmpeg_proc
    _playing = True

    # raw_q: raw bytes from ffmpeg, any size chunks
    # audio_q: assembled BYTES-sized chunks ready for DSP
    raw_q   = queue.Queue(maxsize=2000)
    audio_q = queue.Queue(maxsize=400)

    dsp = CinemaDSP(reverb_mode=reverb_mode, sr=SR,
                norm_mode=norm_mode, eq_gains=eq_gains or {})
    for _ in range(4):
        dsp.process(np.zeros((CHUNK, 2), dtype=np.float32))

    _lufs_run = [True]
    def _lufs_printer():
        while _lufs_run[0]:
            dsp.meter.print_meter()
            time.sleep(0.2)
    threading.Thread(target=_lufs_printer, daemon=True).start()

    # -- Pipe reader -----------------------------------------------------------
    # Dedicated thread whose ONLY job is to call stdout.read() in a tight loop.
    # On Windows, stdout.read(N) blocks until N bytes arrive -- if this runs in
    # the main reader thread it stalls all the retry/reconnect logic.
    # Isolating it here means the block is harmless.
    def pipe_reader():
        while _playing:
            proc = _ffmpeg_proc
            if proc is None or proc.poll() is not None:
                time.sleep(0.01)
                continue
            try:
                data = proc.stdout.read(4096)   # blocking, but isolated here
            except Exception:
                data = b""
            if data:
                try:
                    raw_q.put(data, timeout=1)
                except queue.Full:
                    try: raw_q.get_nowait()
                    except queue.Empty: pass
            else:
                time.sleep(0.002)

    # -- Assembler -------------------------------------------------------------
    # Takes variable-size blobs from raw_q, assembles exact BYTES chunks,
    # puts them in audio_q for the player.
    def assembler():
        buf = b""
        got_data = False
        while _playing:
            try:
                blob = raw_q.get(timeout=0.5)
            except queue.Empty:
                continue
            buf += blob
            if not got_data and len(buf) >= BYTES:
                got_data = True
                print("[Kudio] Stream connected, buffering...")
            while len(buf) >= BYTES:
                chunk = buf[:BYTES]
                buf   = buf[BYTES:]
                try:
                    audio_q.put(chunk, timeout=1)
                except queue.Full:
                    try: audio_q.get_nowait()
                    except queue.Empty: pass

    # -- Manager ---------------------------------------------------------------
    # Manages ffmpeg lifecycle: open, monitor, retry on failure.
    def manager():
        global _playing, _ffmpeg_proc
        retries          = 0
        current_page_url = url

        while _playing:
            if _ffmpeg_proc is None or _ffmpeg_proc.poll() is not None:
                stream_url, hdrs = _get_stream_info(current_page_url)
                if not stream_url:
                    print("[Kudio] Could not get stream URL, retrying in 3s...")
                    time.sleep(3)
                    continue
                _ffmpeg_proc = _open_ffmpeg(stream_url, hdrs)
                retries = 0

            # Monitor: if ffmpeg dies unexpectedly, trigger retry
            rc = _ffmpeg_proc.poll()
            if rc is not None:
                if not _playing:
                    break
                retries += 1
                if retries > MAX_RETRIES:
                    print("\n[Kudio] Max retries. Stopping.")
                    _playing = False
                    break
                wait = 2.0 * (2 ** (retries - 1))
                print(f"\n[Kudio] ffmpeg exited (rc={rc})."
                      f" Retry {retries}/{MAX_RETRIES} in {wait:.0f}s...")
                _kill(_ffmpeg_proc)
                _ffmpeg_proc = None
                time.sleep(wait)
                _, fresh = search_music(query)
                if fresh:
                    current_page_url = fresh
                continue

            time.sleep(0.5)  # poll every 500ms

    # -- Player ----------------------------------------------------------------
    def player():
        global _playing
        with sd.OutputStream(
            samplerate=SR, channels=2, dtype='float32',
            blocksize=CHUNK, latency='high'
        ) as stream:
            while _playing:
                try:
                    raw = audio_q.get(timeout=1.0)
                except queue.Empty:
                    if not _playing:
                        return
                    try:
                        stream.write(np.zeros((CHUNK, 2), dtype=np.float32))
                    except Exception:
                        return
                    continue

                chunk = np.frombuffer(raw, dtype=np.float32).reshape(CHUNK, 2)
                try:
                    out = dsp.process(chunk.copy())
                    stream.write(out.astype(np.float32))
                except Exception as e:
                    if '-9983' not in str(e) and 'underflow' not in str(e).lower():
                        print(f"\n[Kudio] Player error: {e}")
                    return

    # -- Launch ----------------------------------------------------------------
    try:
        threads = [
            threading.Thread(target=pipe_reader, daemon=True),
            threading.Thread(target=assembler,   daemon=True),
            threading.Thread(target=manager,     daemon=True),
            threading.Thread(target=player,      daemon=True),
        ]
        for t in threads:
            t.start()
        # Wait for manager and player to finish
        threads[2].join()   # manager
        threads[3].join()   # player
        print("\nPlayback finished.")
    except Exception as e:
        print(f"[Kudio] Fatal: {e}")
        import traceback; traceback.print_exc()
    finally:
        _lufs_run[0] = False
        _kill(_ffmpeg_proc)


def stop():
    global _playing
    _playing = False