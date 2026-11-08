"""SSTV encoder, for tests: image in, audio out.

    python -m sstv_decoder.synth picture.png --mode "Robot 36" -o test.wav
"""

import argparse
import wave

import numpy as np

from .modes import BLACK_HZ, BY_NAME, SYNC_HZ, WHITE_HZ, Mode

RATE = 12000


def rgb_to_ycc(rgb):
    r, g, b = (rgb[..., i].astype(float) for i in range(3))
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return np.clip(y, 0, 255), np.clip(cr, 0, 255), np.clip(cb, 0, 255)


def _hz(v):
    return BLACK_HZ + np.asarray(v, dtype=float) / 255.0 * (WHITE_HZ - BLACK_HZ)


def vis_events(code: int):
    """(duration_ms, Hz) for the calibration header and VIS code."""
    ev = [(300, 1900), (10, 1200), (300, 1900), (30, 1200)]
    bits = [(code >> i) & 1 for i in range(7)]
    bits.append(sum(bits) % 2)  # even parity
    ev += [(30, 1100 if b else 1300) for b in bits]
    ev.append((30, 1200))
    return ev


def image_events(img: np.ndarray, mode: Mode):
    """(duration_ms, Hz) for each scan of the image; img is height x width x 3 RGB."""
    ev = []
    if mode.pre_ms:
        ev.append((mode.pre_ms, SYNC_HZ))
    y, cr, cb = rgb_to_ycc(img)
    for n in range(mode.lines):
        line = []  # (start_ms, duration_ms, Hz or per-pixel array)
        line.append((mode.sync_at, mode.sync_ms, SYNC_HZ))
        row = n * mode.rows_per_line
        for comp, start, dur in mode.scans:
            if comp == "SEP":
                line.append((start, dur, BLACK_HZ if n % 2 == 0 else WHITE_HZ))
                continue
            if comp in "RGB":
                vals = img[row, :, "RGB".index(comp)]
            elif comp in ("Y", "Y0"):
                vals = y[row]
            elif comp == "Y1":
                vals = y[row + 1]
            elif comp == "C":
                vals = cr[row] if n % 2 == 0 else cb[row]
            elif comp == "RY":
                vals = cr[row] if mode.color != "PD" else (cr[row] + cr[row + 1]) / 2
            elif comp == "BY":
                vals = cb[row] if mode.color != "PD" else (cb[row] + cb[row + 1]) / 2
            line.append((start, dur, _hz(vals)))
        # Fill the gaps (porches, separators) between scans.
        line.sort(key=lambda s: s[0])
        t = 0.0
        for start, dur, hz in line:
            if start > t + 1e-9:
                ev.append((start - t, BLACK_HZ))
            if np.ndim(hz):
                px = dur / len(hz)
                ev += [(px, f) for f in hz]
            else:
                ev.append((dur, hz))
            t = start + dur
        if mode.line_ms > t + 1e-9:
            ev.append((mode.line_ms - t, 1500.0))
    return ev


def render(events, rate=RATE, clock=1.0, start_phase=0.0):
    """Continuous-phase FM audio. clock > 1 plays it slow (a sound card running fast)."""
    durs = np.array([d for d, _ in events]) * clock / 1000.0
    freqs = np.array([f for _, f in events], dtype=float)
    edges = np.concatenate([[0.0], np.cumsum(durs)])
    n = int(edges[-1] * rate)
    t = np.arange(n) / rate
    f = freqs[np.minimum(np.searchsorted(edges, t, side="right") - 1, len(freqs) - 1)]
    phase = start_phase + 2 * np.pi * np.cumsum(f) / rate
    return np.sin(phase).astype(np.float32)


def encode(img: np.ndarray, mode: Mode, rate=RATE, clock=1.0, offset_hz=0.0) -> np.ndarray:
    ev = vis_events(mode.vis) + image_events(img, mode)
    if offset_hz:
        ev = [(d, f + offset_hz) for d, f in ev]
    return render(ev, rate, clock) * 0.5


def color_bars(width, height) -> np.ndarray:
    """Colour bars over a grey ramp, with a checkerboard corner."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    bars = [(255, 255, 255), (255, 255, 0), (0, 255, 255), (0, 255, 0),
            (255, 0, 255), (255, 0, 0), (0, 0, 255), (0, 0, 0)]
    top = height * 2 // 3
    for i, c in enumerate(bars):
        img[:top, i * width // 8 : (i + 1) * width // 8] = c
    ramp = np.linspace(0, 255, width).astype(np.uint8)
    img[top:, :, :] = ramp[None, :, None]
    s = max(width // 40, 4)
    yy, xx = np.mgrid[0 : height // 6, 0 : width // 4]
    img[: height // 6, : width // 4] = (((yy // s + xx // s) % 2) * 255)[..., None]
    return img


def write_wav(path, audio, rate=RATE):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("image", nargs="?", help="picture to send (default: test bars)")
    p.add_argument("--mode", default="Robot 36", choices=list(BY_NAME))
    p.add_argument("--rate", type=int, default=RATE)
    p.add_argument("-o", "--output", required=True)
    a = p.parse_args()
    mode = BY_NAME[a.mode]
    if a.image:
        from PIL import Image

        img = np.asarray(Image.open(a.image).convert("RGB").resize((mode.width, mode.height)))
    else:
        img = color_bars(mode.width, mode.height)
    write_wav(a.output, encode(img, mode, a.rate), a.rate)


if __name__ == "__main__":
    main()
