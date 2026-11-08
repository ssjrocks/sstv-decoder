"""Main window: source picker, spectrum, the picture coming in, received pictures."""

import logging
import time
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QSettings, QSize, Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QDesktopServices, QFont, QIcon, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from . import APP_ID, APP_NAME, appreports
from .capture import AudioCapture, list_outputs
from .decoder import RATE, Image, SSTVDecoder
from .modes import MODES

log = logging.getLogger(__name__)

SPEC_LO, SPEC_HI = 1000, 2500
MARKS = [(1200, "sync"), (1500, "black"), (1900, "VIS"), (2300, "white")]
MIN_SAVE_FRACTION = 0.1
GALLERY_MAX = 60


def to_qimage(pixels: np.ndarray) -> QImage:
    h, w, _ = pixels.shape
    data = np.ascontiguousarray(pixels)
    return QImage(data.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class Spectrum(QWidget):
    """Audio spectrum SPEC_LO..SPEC_HI Hz with the SSTV frequencies marked."""

    def __init__(self):
        super().__init__()
        self.setFixedHeight(70)
        self.db = None
        self.freqs = None
        self.offset = 0.0
        self.setToolTip(
            "Audio spectrum. An SSTV signal fills 1500-2300 Hz with its sync at 1200 Hz. "
            "Marks move with the tuning error measured on the last picture."
        )

    def set_spectrum(self, freqs, db):
        self.freqs, self.db = freqs, db
        self.update()

    def _x(self, f):
        return (f - SPEC_LO) / (SPEC_HI - SPEC_LO) * self.width()

    def paintEvent(self, _ev):
        p = QPainter(self)
        pal = self.palette()
        p.fillRect(self.rect(), pal.base())
        w, h = self.width(), self.height()
        x0, x1 = self._x(1500 + self.offset), self._x(2300 + self.offset)
        p.fillRect(int(x0), 0, int(x1 - x0), h, QColor(80, 140, 255, 40))
        small = QFont(self.font())
        small.setPointSizeF(small.pointSizeF() * 0.8)
        p.setFont(small)
        for f, label in MARKS:
            x = int(self._x(f + self.offset))
            p.setPen(QPen(QColor(255, 140, 0, 160), 1, Qt.PenStyle.DashLine))
            p.drawLine(x, 0, x, h)
            p.setPen(pal.placeholderText().color())
            p.drawText(x + 3, 11, f"{label}")
        if self.db is not None:
            lo, hi = -90.0, -10.0
            p.setPen(QPen(pal.text().color(), 1))
            ys = h - (np.clip(self.db, lo, hi) - lo) / (hi - lo) * (h - 14)
            xs = (self.freqs - SPEC_LO) / (SPEC_HI - SPEC_LO) * w
            for i in range(1, len(xs)):
                p.drawLine(int(xs[i - 1]), int(ys[i - 1]), int(xs[i]), int(ys[i]))
        p.end()


class PictureView(QWidget):
    """Shows a picture scaled to fit, with the line being received marked."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(320, 240)
        self.image: QImage | None = None
        self.row = None
        self.placeholder = "Waiting for a picture…"

    def show_image(self, image: QImage | None, row=None):
        self.image, self.row = image, row
        self.update()

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(24, 24, 24))
        if self.image is None:
            p.setPen(QColor(160, 160, 160))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.placeholder)
            p.end()
            return
        iw, ih = self.image.width(), self.image.height()
        # SSTV pictures are 4:3 whatever their pixel count (PD 160 is 512x400).
        aspect = 4 / 3
        w = min(self.width(), int(self.height() * aspect))
        h = int(w / aspect)
        x, y = (self.width() - w) // 2, (self.height() - h) // 2
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.drawImage(self.rect().adjusted(x, y, -(self.width() - w - x), -(self.height() - h - y)), self.image)
        if self.row is not None and self.row < ih:
            ly = y + int(self.row / ih * h)
            p.setPen(QPen(QColor(255, 140, 0), 1))
            p.drawLine(x, ly, x + w, ly)
        p.end()


class MainWindow(QMainWindow):
    report_finished = pyqtSignal(bool)

    def __init__(self, state_dir: Path, pictures_dir: Path | None = None):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.settings = QSettings("ssjrocks", "sstv-decoder")
        self.pictures_dir = pictures_dir or Path.home() / "Pictures" / "SSTV"
        self.decoder = SSTVDecoder()
        self.capture: AudioCapture | None = None
        self.audio_tail = np.zeros(0, np.float32)
        self._spec_avg = None
        self._last_level = -120.0
        self._shown_live = True       # the view follows the incoming picture

        self.source_combo = QComboBox()
        self.source_combo.setToolTip("The output to listen to. Default follows your speaker setting.")
        refresh = QToolButton()
        refresh.setText("⟳")
        refresh.setToolTip("Refresh the list of outputs")
        refresh.clicked.connect(self.fill_sources)
        self.start_btn = QPushButton("Start")
        self.start_btn.setCheckable(True)
        self.start_btn.toggled.connect(self.on_start_toggled)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Any mode (automatic)", "")
        for m in MODES:
            self.mode_combo.addItem(f"{m.name} only", m.name)
        self.mode_combo.setToolTip(
            "Any mode starts on the VIS code at the start of a picture, or on the sync "
            "pulses if you tuned in part way through. Pick a mode to ignore the others."
        )
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        self.stop_pic_btn = QPushButton("End picture")
        self.stop_pic_btn.setToolTip("Stop receiving this picture now and save what came in")
        self.stop_pic_btn.clicked.connect(self.end_picture)
        self.stop_pic_btn.setEnabled(False)
        folder_btn = QPushButton("Open folder")
        folder_btn.setToolTip(f"Received pictures are saved in {self.pictures_dir}")
        folder_btn.clicked.connect(self.open_folder)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Listen to"))
        row1.addWidget(self.source_combo, 1)
        row1.addWidget(refresh)
        row1.addWidget(self.start_btn)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Mode"))
        row2.addWidget(self.mode_combo)
        row2.addWidget(self.stop_pic_btn)
        row2.addStretch(1)
        row2.addWidget(folder_btn)

        self.spectrum = Spectrum()
        self.view = PictureView()
        self.info = QLabel("Waiting for a picture (VIS code or sync pulses)…")
        self.info.setWordWrap(True)

        self.gallery = QListWidget()
        self.gallery.setViewMode(QListWidget.ViewMode.IconMode)
        self.gallery.setIconSize(QSize(144, 108))
        self.gallery.setGridSize(QSize(160, 156))
        self.gallery.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.gallery.setMovement(QListWidget.Movement.Static)
        self.gallery.setWordWrap(True)
        self.gallery.setToolTip("Received pictures. Click one to look at it; double-click to open it.")
        self.gallery.itemClicked.connect(self.show_saved)
        self.gallery.itemDoubleClicked.connect(
            lambda it: QDesktopServices.openUrl(QUrl.fromLocalFile(it.data(Qt.ItemDataRole.UserRole)))
        )
        self.live_btn = QPushButton("Back to live")
        self.live_btn.clicked.connect(self.show_live)
        self.live_btn.setVisible(False)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(self.view, 1)
        ll.addWidget(self.info)
        ll.addWidget(self.live_btn)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(self.gallery)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)

        save_btn = QPushButton("Save as…")
        save_btn.clicked.connect(self.save_as)
        copy_btn = QPushButton("Copy picture")
        copy_btn.clicked.connect(self.copy_picture)
        report_btn = QPushButton("Report a problem…")
        report_btn.clicked.connect(self.report_problem)
        btns = QHBoxLayout()
        btns.addWidget(save_btn)
        btns.addWidget(copy_btn)
        btns.addStretch(1)
        btns.addWidget(report_btn)

        lay = QVBoxLayout()
        lay.addLayout(row1)
        lay.addLayout(row2)
        lay.addWidget(self.spectrum)
        lay.addWidget(split, 1)
        lay.addLayout(btns)
        central = QWidget()
        central.setLayout(lay)
        self.setCentralWidget(central)
        self.level_label = QLabel("")
        self.statusBar().addPermanentWidget(self.level_label)

        self.report_finished.connect(self._on_report_finished)
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.tick)

        self.fill_sources()
        i = self.mode_combo.findData(self.settings.value("mode", ""))
        self.mode_combo.setCurrentIndex(max(i, 0))
        self.on_mode_changed(self.mode_combo.currentIndex())
        self.load_gallery()
        geo = self.settings.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        else:
            self.resize(1000, 720)
        self.start_btn.setChecked(True)

    # ---- source and settings -----------------------------------------

    def fill_sources(self):
        current = (self.source_combo.currentData() if self.source_combo.count()
                   else self.settings.value("source", ""))
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        self.source_combo.addItem("Default output (follows your speaker setting)", "")
        for name, desc in list_outputs():
            self.source_combo.addItem(desc, name)
        i = self.source_combo.findData(current or "")
        self.source_combo.setCurrentIndex(max(i, 0))
        self.source_combo.blockSignals(False)
        try:
            self.source_combo.currentIndexChanged.disconnect(self.on_source_changed)
        except TypeError:
            pass
        self.source_combo.currentIndexChanged.connect(self.on_source_changed)

    def on_source_changed(self, _i):
        self.settings.setValue("source", self.source_combo.currentData() or "")
        if self.start_btn.isChecked():
            self.stop_capture()
            self.start_btn.setChecked(True)

    def on_mode_changed(self, _i):
        name = self.mode_combo.currentData()
        self.decoder.forced = next((m for m in MODES if m.name == name), None)
        self.settings.setValue("mode", name or "")

    # ---- capture -------------------------------------------------------

    def on_start_toggled(self, on):
        if on:
            self.start_capture()
        else:
            self.stop_capture()

    def start_capture(self):
        target = self.source_combo.currentData() or None
        self.capture = AudioCapture(target)
        try:
            self.capture.start()
        except RuntimeError as e:
            self.capture = None
            self.start_btn.setChecked(False)
            QMessageBox.warning(self, APP_NAME, str(e))
            return
        log.info("listening to %s", target or "default output")
        self.start_btn.setText("Stop")
        self.statusBar().showMessage(f"Listening to {self.source_combo.currentText()}")
        self.timer.start()

    def stop_capture(self):
        self.timer.stop()
        if self.capture:
            self.capture.stop()
            self.capture = None
        for kind, img in self.decoder.stop_image():
            self.handle(kind, img)
        self.start_btn.blockSignals(True)
        self.start_btn.setChecked(False)
        self.start_btn.blockSignals(False)
        self.start_btn.setText("Start")
        self.statusBar().showMessage("Stopped")

    def tick(self):
        try:
            audio = self.capture.read()
        except RuntimeError as e:
            log.warning("capture stopped: %s", e)
            self.stop_capture()
            QMessageBox.warning(self, APP_NAME, f"Listening stopped.\n\n{e}")
            return
        if not len(audio):
            return
        self.audio_tail = np.concatenate([self.audio_tail, audio])[-4096:]
        try:
            events = self.decoder.feed(audio)
        except Exception as e:  # keep listening; send the bug in
            log.exception("decoder failed")
            appreports.report_error(e, "decoding audio")
            self.decoder = SSTVDecoder()
            self.on_mode_changed(self.mode_combo.currentIndex())
            return
        for kind, img in events:
            self.handle(kind, img)
        self.update_spectrum()

    # ---- pictures --------------------------------------------------------

    def handle(self, kind: str, img: Image):
        mode = img.mode
        if kind == "start":
            how = "VIS" if img.vis else "sync pulses, no VIS: joined part way through"
            log.info("receiving %s (%s), offset %+.0f Hz", mode.name, how, img.offset_hz)
            self.stop_pic_btn.setEnabled(True)
            self.spectrum.offset = img.offset_hz
        if kind in ("start", "rows"):
            if self._shown_live:
                self.view.show_image(to_qimage(img.pixels), img.rows_done)
            self.info.setText(self._describe(img, receiving=True))
        elif kind == "done":
            self.stop_pic_btn.setEnabled(False)
            self.info.setText(self._describe(img, receiving=False))
            if self._shown_live:
                self.view.show_image(to_qimage(img.pixels), None)
            self.save_picture(img)

    def _describe(self, img: Image, receiving: bool) -> str:
        m = img.mode
        started = time.strftime("%H:%M:%S", time.localtime(img.started))
        parts = [f"<b>{m.name}</b> {m.width}×{m.height}", f"started {started}"]
        if receiving:
            parts.append(f"row {img.rows_done} of {m.height}")
        else:
            parts.append("complete" if img.complete else f"ended at row {img.rows_done} of {m.height}")
        parts.append(f"tuning {round(img.offset_hz) + 0:+d} Hz")
        parts.append(f"clock {round(img.slant_ppm) + 0:+d} ppm")
        if not img.vis:
            parts.append("no VIS (joined part way through)")
        return " · ".join(parts)

    def save_picture(self, img: Image):
        if img.rows_done < MIN_SAVE_FRACTION * img.mode.height:
            log.info("%s: only %d rows, not saved", img.mode.name, img.rows_done)
            return
        stamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(img.started))
        name = f"{stamp}_{img.mode.name.replace(' ', '')}.png"
        try:
            self.pictures_dir.mkdir(parents=True, exist_ok=True)
            path = self.pictures_dir / name
            if not to_qimage(img.pixels).save(str(path), "PNG"):
                raise OSError("Qt could not write the PNG")
        except OSError as e:
            log.warning("could not save %s: %s", name, e)
            self.statusBar().showMessage(f"Could not save the picture: {e}", 8000)
            return
        log.info("saved %s", path)
        self.statusBar().showMessage(f"Saved {path}", 8000)
        self.add_to_gallery(path, first=True)

    def load_gallery(self):
        if not self.pictures_dir.is_dir():
            return
        files = sorted(self.pictures_dir.glob("*.png"), reverse=True)[:GALLERY_MAX]
        for path in files:
            self.add_to_gallery(path)

    def add_to_gallery(self, path: Path, first=False):
        pix = QPixmap(str(path))
        if pix.isNull():
            return
        # Saved as 2026-11-08_14-02-11_Robot36: shown as time and mode, then the date.
        date, _, rest = path.stem.partition("_")
        tm, _, mode = rest.partition("_")
        label = f"{tm.replace('-', ':')} {mode}\n{date}"
        item = QListWidgetItem(QIcon(pix.scaled(144, 108, Qt.AspectRatioMode.IgnoreAspectRatio,
                                                Qt.TransformationMode.SmoothTransformation)), label)
        item.setData(Qt.ItemDataRole.UserRole, str(path))
        item.setToolTip(str(path))
        if first:
            self.gallery.insertItem(0, item)
        else:
            self.gallery.addItem(item)
        while self.gallery.count() > GALLERY_MAX:
            self.gallery.takeItem(self.gallery.count() - 1)

    def show_saved(self, item: QListWidgetItem):
        self._shown_live = False
        self.view.show_image(QImage(item.data(Qt.ItemDataRole.UserRole)))
        self.live_btn.setVisible(True)

    def show_live(self):
        self._shown_live = True
        self.live_btn.setVisible(False)
        self.gallery.clearSelection()
        img = self.decoder.image
        self.view.show_image(to_qimage(img.pixels) if img is not None else None,
                             img.rows_done if img is not None else None)

    def end_picture(self):
        for kind, img in self.decoder.stop_image():
            self.handle(kind, img)

    def _current_qimage(self):
        return self.view.image

    def save_as(self):
        image = self._current_qimage()
        if image is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save picture", str(self.pictures_dir / "sstv.png"), "PNG (*.png)"
        )
        if path:
            image.save(path, "PNG")

    def copy_picture(self):
        image = self._current_qimage()
        if image is not None:
            QApplication.clipboard().setImage(image)

    def open_folder(self):
        self.pictures_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.pictures_dir)))

    def update_spectrum(self):
        n = 4096
        if len(self.audio_tail) < n:
            return
        x = self.audio_tail[-n:] * np.hanning(n)
        p = np.abs(np.fft.rfft(x)) ** 2 / (n * n / 8)
        freqs = np.fft.rfftfreq(n, 1 / RATE)
        band = (freqs >= SPEC_LO) & (freqs <= SPEC_HI)
        db = 10 * np.log10(p[band] + 1e-12)
        if self._spec_avg is None or len(self._spec_avg) != len(db):
            self._spec_avg = db
        else:
            self._spec_avg = self._spec_avg * 0.6 + db * 0.4
        self.spectrum.set_spectrum(freqs[band], self._spec_avg)
        level = 10 * np.log10(np.mean(self.audio_tail**2) + 1e-12)
        self.level_label.setText(f"Audio {level:.0f} dBFS")

    # ---- reports -------------------------------------------------------

    def report_problem(self):
        note, ok = QInputDialog.getMultiLineText(
            self,
            "Report a problem",
            "What went wrong? This is sent with the app's log and settings, "
            "never the pictures.",
        )
        if ok:
            img = self.decoder.image
            extra = {
                "mode_setting": self.mode_combo.currentText(),
                "source": "default" if not self.source_combo.currentData() else "other output",
                "listening": str(self.capture is not None),
                "receiving": img.mode.name if img is not None else "nothing",
            }
            appreports.send_report(note, extra, on_done=self.report_finished.emit)

    def _on_report_finished(self, sent: bool):
        if sent:
            QMessageBox.information(self, "Report a problem", "Report sent. Thanks.")
        else:
            QMessageBox.warning(self, "Report a problem", "The report could not be sent.")

    def closeEvent(self, ev):
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop_capture()
        super().closeEvent(ev)


def run_gui(state_dir: Path) -> int:
    app = QApplication.instance() or QApplication([])
    app.setApplicationName(APP_NAME)
    app.setDesktopFileName(APP_ID)
    win = MainWindow(state_dir)
    win.show()
    return app.exec()
