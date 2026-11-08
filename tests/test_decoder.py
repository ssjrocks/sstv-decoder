"""Encode test pictures, decode them back.

    python3 -m pytest tests
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import butter, sosfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sstv_decoder.decoder import RATE, SSTVDecoder, decode_audio  # noqa: E402
from sstv_decoder.modes import BY_NAME, MODES  # noqa: E402
from sstv_decoder.synth import encode, color_bars  # noqa: E402

FAST = ["Robot 36", "Martin M2", "Scottie S2", "PD 50"]


def pad(a, before=1.0, after=1.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    out = np.concatenate([np.zeros(int(RATE * before)), a, np.zeros(int(RATE * after))])
    return (out + rng.standard_normal(len(out)) * noise).astype(np.float32)


def error(img, ref):
    return np.abs(img.pixels.astype(int) - ref.astype(int))[:, 4:-4].mean()


@pytest.mark.parametrize("mode", MODES, ids=lambda m: m.name)
def test_every_mode(mode):
    ref = color_bars(mode.width, mode.height)
    imgs = decode_audio(pad(encode(ref, mode)))
    assert len(imgs) == 1
    img = imgs[0]
    assert img.mode is mode and img.vis and img.complete
    assert error(img, ref) < 4


@pytest.mark.parametrize("name", FAST)
def test_clock_error_and_mistuning(name):
    # A sound card / WebSDR stream 0.3% fast, and a receiver 80 Hz low (SSB).
    mode = BY_NAME[name]
    ref = color_bars(mode.width, mode.height)
    img = decode_audio(pad(encode(ref, mode, clock=0.997, offset_hz=-80)))[0]
    assert img.complete
    assert abs(img.offset_hz + 80) < 5
    assert abs(img.slant_ppm + 3000) < 100
    assert error(img, ref) < 4


@pytest.mark.parametrize("name", FAST)
def test_noise(name):
    mode = BY_NAME[name]
    ref = color_bars(mode.width, mode.height)
    a = encode(ref, mode)
    img = decode_audio(pad(a, noise=0.5 / np.sqrt(2) * 10 ** (-12 / 20)))[0]
    assert img.complete and error(img, ref) < 25


@pytest.mark.parametrize("name", ["Robot 36", "Robot 72", "Martin M1", "Scottie DX", "PD 120"])
def test_pickup_without_vis(name):
    # Tuned in 30% into the picture: found from its sync pulses.
    mode = BY_NAME[name]
    ref = color_bars(mode.width, mode.height)
    a = encode(ref, mode, clock=1.001, offset_hz=40)
    imgs = decode_audio(pad(a[int(len(a) * 0.3) :], before=2, after=2, noise=0.05))
    assert [(i.mode, i.vis) for i in imgs] == [(mode, False)]
    assert imgs[0].rows_done > 0.55 * mode.height
    assert abs(imgs[0].offset_hz - 40) < 10


def test_two_pictures_back_to_back():
    m1, m2 = BY_NAME["Robot 36"], BY_NAME["Martin M2"]
    a = np.concatenate([encode(color_bars(320, 240), m1), np.zeros(RATE), encode(color_bars(320, 256), m2)])
    imgs = decode_audio(pad(a))
    assert [(i.mode, i.complete) for i in imgs] == [(m1, True), (m2, True)]


def test_nothing_from_noise_or_voice():
    rng = np.random.default_rng(7)
    noise = sosfilt(butter(4, [300, 3000], btype="band", fs=RATE, output="sos"),
                    rng.standard_normal(RATE * 120)) * 0.3
    t = np.arange(RATE * 60) / RATE
    f0 = 140 + 50 * np.sin(2 * np.pi * 0.5 * t)
    ph = np.cumsum(2 * np.pi * f0 / RATE)
    voice = sum(np.sin(k * ph) / k for k in range(1, 25)) * (np.sin(2 * np.pi * 2.5 * t) > 0) * 0.2
    assert decode_audio(np.concatenate([noise, voice]).astype(np.float32)) == []


@pytest.mark.parametrize("seed", range(6))
def test_chunk_size_does_not_matter(seed):
    # Live audio arrives in pieces of any size. Judging the VIS on part of
    # them once moved the picture's start, and line 0 locked onto the VIS
    # stop bit. Where the picture starts must not depend on the pieces.
    mode = BY_NAME["Robot 36"]
    ref = color_bars(320, 240)
    a = pad(encode(ref, mode), noise=0.03, seed=seed)
    rng = np.random.default_rng(seed)
    starts = []
    for sizes in ([2400], [1, 7, 120, 128, 256, 1200, 2048, 4000], [60, 61, 333]):
        dec, done, i = SSTVDecoder(), [], 0
        while i < len(a):
            k = int(rng.choice(sizes))
            for kind, img in dec.feed(a[i : i + k], now=0):
                if kind == "start":
                    starts.append(dec._t0)
                elif kind == "done":
                    done.append(img)
            i += k
        assert len(done) == 1 and error(done[0], ref) < 15
    # Within half a millisecond (the VIS search runs on a 1 ms grid); the
    # bug moved it by 3-12 ms.
    assert max(starts) - min(starts) <= RATE * 0.0005


@pytest.mark.parametrize("name", ["Robot 36", "Martin M1", "Scottie S1", "PD 120"])
def test_audio_dropout(name):
    # A WebSDR stream losing 80 ms of audio part way through the picture.
    mode = BY_NAME[name]
    ref = color_bars(mode.width, mode.height)
    a = encode(ref, mode)
    cut = int(len(a) * 0.4)
    a = np.concatenate([a[:cut], a[cut + int(0.08 * RATE) :]])
    img = decode_audio(pad(a))[0]
    assert img.rows_done >= mode.height - 2 * mode.rows_per_line
    # Rows after the dropout are aligned again (allow a few rows around it).
    rows = int(0.5 * mode.height)
    err = np.abs(img.pixels[rows:].astype(int) - ref[rows : len(img.pixels)].astype(int))[:, 4:-4]
    assert err.mean() < 6
