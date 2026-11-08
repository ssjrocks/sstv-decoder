"""Decode audio made by PySSTV, an independent encoder, to check mode timings and colours.

    uv venv --system-site-packages --python /usr/bin/python3 build/xcheck
    uv pip install --python build/xcheck/bin/python pysstv
    build/xcheck/bin/python tests/crosscheck_pysstv.py [output dir]
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pysstv import color

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sstv_decoder.decoder import RATE, decode_audio  # noqa: E402
from sstv_decoder.modes import BY_NAME  # noqa: E402
from sstv_decoder.synth import color_bars  # noqa: E402

PAIRS = {
    "Robot 36": color.Robot36, "Martin M1": color.MartinM1, "Martin M2": color.MartinM2,
    "Scottie S1": color.ScottieS1, "Scottie S2": color.ScottieS2, "Scottie DX": color.ScottieDX,
    "PD 90": color.PD90, "PD 120": color.PD120, "PD 160": color.PD160, "PD 180": color.PD180,
    "PD 240": color.PD240, "PD 290": color.PD290, "Wraase SC2-180": color.WraaseSC2180,
}

out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
failed = 0
for name, cls in PAIRS.items():
    mode = BY_NAME[name]
    ref = color_bars(mode.width, mode.height)
    sstv = cls(Image.fromarray(ref), RATE, 16)
    audio = np.fromiter(sstv.gen_samples(), dtype=np.float32) / 32768.0
    audio = np.concatenate([np.zeros(RATE, np.float32), audio, np.zeros(RATE, np.float32)])
    imgs = decode_audio(audio)
    if not imgs or imgs[0].mode is not mode:
        print(f"{name:15s} FAIL: decoded {[i.mode.name for i in imgs]}")
        failed += 1
        continue
    im = imgs[0]
    if name == "Martin M2":
        # PySSTV sends M2 160 pixels wide (the left half of the picture) over the same scan time.
        ref = np.repeat(ref[:, :160], 2, axis=1)
    err = np.abs(im.pixels.astype(int) - ref.astype(int))[:, 4:-4]
    ok = err.mean() < 12
    failed += not ok
    print(f"{name:15s} {'ok  ' if ok else 'FAIL'} rows {im.rows_done}/{mode.height} "
          f"slant {im.slant_ppm:+.0f} ppm, mean error {err.mean():.1f} "
          f"(R {err[..., 0].mean():.1f} G {err[..., 1].mean():.1f} B {err[..., 2].mean():.1f})")
    if out:
        Image.fromarray(im.pixels).save(out / f"pysstv_{name.replace(' ', '_')}.png")
sys.exit(1 if failed else 0)
