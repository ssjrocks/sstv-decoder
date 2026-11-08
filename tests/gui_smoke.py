"""Headless window check: feeds synthetic pictures through the real window.

    QT_QPA_PLATFORM=offscreen python3 tests/gui_smoke.py [screenshot.png]
"""

import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.excepthook = lambda *a: traceback.print_exception(*a)

import numpy as np  # noqa: E402
from PyQt6.QtCore import QSettings, QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from sstv_decoder import appreports, gui  # noqa: E402
from sstv_decoder.decoder import RATE  # noqa: E402
from sstv_decoder.modes import BY_NAME  # noqa: E402
from sstv_decoder.synth import color_bars, encode  # noqa: E402

appreports.init = lambda *a, **k: None  # never send test failures as real reports
appreports.send_report = lambda *a, **k: None
appreports.report_error = lambda *a, **k: None


class FakeCapture:
    def __init__(self, audio):
        self.audio, self.pos = audio, 0

    def start(self):
        pass

    def read(self):
        chunk = self.audio[self.pos : self.pos + 2400]
        self.pos += 2400
        return chunk if len(chunk) else np.zeros(600, dtype=np.float32)

    def stop(self):
        pass


r36, m2 = BY_NAME["Robot 36"], BY_NAME["Martin M2"]
gap = np.zeros(RATE, np.float32)
audio = np.concatenate([gap, encode(color_bars(320, 240), r36), gap, encode(color_bars(320, 256), m2), gap])
gui.AudioCapture = lambda target=None: FakeCapture(audio)
ini = tempfile.mktemp(suffix=".ini")
gui.QSettings = lambda *a: QSettings(ini, QSettings.Format.IniFormat)
pics = Path(tempfile.mkdtemp())

app = QApplication([])
win = gui.MainWindow(Path(tempfile.mkdtemp()), pictures_dir=pics)
win.timer.setInterval(1)
win.show()
shot = sys.argv[1] if len(sys.argv) > 1 else None


def mid():
    if shot:
        win.grab().save(shot.replace(".png", "_mid.png"))


QTimer.singleShot(2500, mid)
QTimer.singleShot(9000, app.quit)
app.exec()
if shot:
    win.grab().save(shot)
saved = sorted(p.name for p in pics.glob("*.png"))
print("saved:", saved, "gallery:", win.gallery.count())
ok = (len(saved) == 2 and saved[0].endswith("_Robot36.png") and saved[1].endswith("_MartinM2.png")
      and win.gallery.count() == 2)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
