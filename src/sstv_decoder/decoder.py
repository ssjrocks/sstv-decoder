"""SSTV receive: audio -> instantaneous frequency -> VIS -> image rows.

The decoder works on the frequency track. It looks for the VIS header (two
1900 Hz leaders, then a 1200 Hz start bit, 7 data bits, parity and a stop
bit), measures the leader's frequency to correct a mistuned receiver (SSB on
HF), and then reads the mode's lines. Each line's sync pulse is found, and a
straight line fitted through the pulses gives the line period, so a sound card
or WebSDR stream whose clock is off doesn't slant the picture. Scans are read
as the mean frequency over each pixel.
"""

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, sosfilt

from .modes import BLACK_HZ, BY_VIS, MODES, SYNC_HZ, WHITE_HZ, Mode

log = logging.getLogger(__name__)

RATE = 12000
CENTER_HZ = 1900.0
MAX_OFFSET_HZ = 250.0


class FMDemod:
    """Streaming instantaneous-frequency estimate, one value per sample."""

    def __init__(self, rate: int = RATE):
        self.rate = rate
        self._sos = butter(6, 1100, fs=rate, output="sos")
        self._zi = np.zeros((self._sos.shape[0], 2, 2))
        self._phase = 0.0
        self._last = 1 + 0j
        self._pw_sos = butter(2, 100, fs=rate, output="sos")
        self._pw_zi = np.zeros((self._pw_sos.shape[0], 2))
        self._wide_sos = butter(4, [400, 3400], btype="band", fs=rate, output="sos")
        self._wide_zi = np.zeros((self._wide_sos.shape[0], 2))
        self._wpw_zi = np.zeros((self._pw_sos.shape[0], 2))

    def process(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (frequency in Hz, fraction of the audio inside the SSTV band)."""
        x = np.asarray(x, dtype=np.float64)
        n = len(x)
        if not n:
            return np.zeros(0, np.float32), np.zeros(0, np.float32)
        w = 2 * np.pi * CENTER_HZ / self.rate
        ph = self._phase + w * np.arange(n)
        self._phase = float((self._phase + w * n) % (2 * np.pi))
        bb = np.stack([x * np.cos(ph), -x * np.sin(ph)])
        bb, self._zi = sosfilt(self._sos, bb, axis=1, zi=self._zi)
        z = bb[0] + 1j * bb[1]
        prev = np.concatenate([[self._last], z[:-1]])
        self._last = z[-1]
        freq = CENTER_HZ + np.angle(z * np.conj(prev)) * self.rate / (2 * np.pi)
        # How much of the audio is in the SSTV band: noise and voice score low.
        inband, self._pw_zi = sosfilt(self._pw_sos, 2 * np.abs(z) ** 2, zi=self._pw_zi)
        wide, self._wide_zi = sosfilt(self._wide_sos, x, zi=self._wide_zi)
        wide_p, self._wpw_zi = sosfilt(self._pw_sos, wide * wide, zi=self._wpw_zi)
        quality = np.clip(inband / np.maximum(wide_p, 1e-12), 0, 1)
        return freq.astype(np.float32), quality.astype(np.float32)


@dataclass
class Image:
    mode: Mode
    started: float                        # time.time() at the VIS
    offset_hz: float                      # receiver mistuning, measured on the leader
    pixels: np.ndarray                    # height x width x 3 uint8
    rows_done: int = 0
    slant_ppm: float = 0.0
    finished: bool = False
    complete: bool = False                # every line was received
    vis: bool = True                      # False: picked up from sync pulses, mid-picture
    _y: dict = field(default_factory=dict, repr=False)   # Robot 36: row -> (Y, C, is_RY)


class SSTVDecoder:
    """Feed audio; returns events: ("start", Image), ("rows", Image), ("done", Image)."""

    def __init__(self, rate: int = RATE):
        self.rate = rate
        self.demod = FMDemod(rate)
        self.freq = np.zeros(0, np.float32)
        self.quality = np.zeros(0, np.float32)
        self.base = 0                         # absolute sample index of freq[0]
        self.end = 0                          # absolute index after the last sample
        self._vis_from = 0                    # VIS search resumes here (absolute)
        self.image: Image | None = None
        self.forced: Mode | None = None       # decode only this mode (None = any)
        self._line = 0
        self._t0 = 0.0                        # absolute sample of line 0's start
        self._syncs: list[tuple[int, float]] = []
        self._period = 0.0                    # samples per line, fitted
        self._misses = 0
        self._redrawn = False                 # first lines redrawn once the slant is known
        self._train_from = 0                  # sync-train search resumes here (absolute)
        self._train_after = 0                 # pulses before this belong to a finished picture

    # ---- input -----------------------------------------------------------

    def feed(self, audio: np.ndarray, now: float | None = None) -> list:
        now = time.time() if now is None else now
        f, q = self.demod.process(audio)
        self.freq = np.concatenate([self.freq, f])
        self.quality = np.concatenate([self.quality, q])
        self.end += len(f)
        events = []
        vis = self._find_vis()
        if vis is not None:
            start, mode, offset = vis
            if self.image is not None:
                events.append(self._finish())
            if self.forced is None or self.forced is mode:
                events.append(self._begin(start, mode, offset, now))
        elif self.image is None:
            train = self._find_sync_train()
            if train is not None:
                start, mode, offset = train
                events.append(self._begin(start, mode, offset, now, vis=False))
        if self.image is not None:
            events += self._decode_lines()
        self._trim()
        return events

    def stop_image(self) -> list:
        """End the image being received now (the user pressed Stop)."""
        return [self._finish()] if self.image is not None else []

    def _trim(self):
        keep_from = self.end - int(14.0 * self.rate)
        if self.image is not None:
            first = 0 if not self._redrawn else self._line
            keep_from = min(keep_from, int(self._t0 + first * self._period) - int(0.2 * self.rate))
        cut = keep_from - self.base
        if cut > self.rate:
            self.freq = self.freq[cut:]
            self.quality = self.quality[cut:]
            self.base += cut

    def _ms(self, ms: float) -> float:
        return ms * self.rate / 1000.0

    def _cumsum(self):
        return np.concatenate([[0.0], np.cumsum(self.freq, dtype=np.float64)])

    # ---- VIS -------------------------------------------------------------

    def _find_vis(self):
        """Look for a complete VIS in audio not searched yet.

        Candidates are start-bit positions on a 1 ms grid. The leader before
        it must be steady near 1900 Hz; the start bit, bits and stop bit are
        judged relative to the measured leader, so a mistuned signal still
        decodes.
        """
        ms = self._ms
        need_before, need_after = int(ms(300)), int(ms(300))
        lo = max(self._vis_from, self.base + need_before)
        hi = self.end - need_after
        if hi <= lo:
            return None
        step = int(ms(1))
        cand = np.arange(int(lo), int(hi), step) - self.base
        self._vis_from = hi
        c = self._cumsum()

        def mean(a, b):
            return (c[cand + int(b)] - c[cand + int(a)]) / (int(b) - int(a))

        # A steady leader: both halves at the same frequency. (The per-sample
        # frequency is too noisy on a weak signal for a variance test.)
        first, second = mean(-ms(250), -ms(130)), mean(-ms(130), -ms(15))
        lead = (first + second) / 2
        ok = (np.abs(lead - CENTER_HZ) < MAX_OFFSET_HZ) & (np.abs(first - second) < 40.0)
        if not ok.any():
            return None
        tol = 90.0
        start_bit = mean(ms(5), ms(25))
        ok &= np.abs(start_bit - (lead - 700)) < tol
        stop_bit = mean(ms(275), ms(295))
        ok &= np.abs(stop_bit - (lead - 700)) < tol
        if not ok.any():
            return None
        bits = []
        err = np.abs(start_bit - (lead - 700)) + np.abs(stop_bit - (lead - 700))
        for k in range(8):
            v = mean(ms(35 + 30 * k), ms(55 + 30 * k)) - lead
            one = v < -700
            ok &= np.abs(v - np.where(one, -800, -600)) < tol
            err = err + np.abs(v - np.where(one, -800, -600))
            bits.append(one)
        idx = np.flatnonzero(ok)
        if not len(idx):
            return None
        if idx[-1] >= len(cand) - int(ms(15)) // step:
            # The run of matches reaches the end of what has arrived: decide
            # when all of it is here, or the chosen position depends on how
            # the audio happened to be chunked.
            self._vis_from = cand[idx[0]] + self.base - int(ms(20))
            return None
        # The matches form a run around the true position; take the best one,
        # then put the start bit's edge where the frequency crosses halfway.
        best = idx[np.argmin(err[idx])]
        bits = [bool(b[best]) for b in bits]
        edge = self._vis_edge(cand[best], float(lead[best]))
        if sum(bits) % 2:
            log.info("VIS parity error, ignored")
            return None
        code = sum(1 << i for i, b in enumerate(bits[:7]) if b)
        offset = float(lead[best] - CENTER_HZ)
        start = edge + self.base
        self._vis_from = int(start) + int(ms(300))
        mode = BY_VIS.get(code)
        if mode is None:
            log.info("VIS code %d at sample %d is not a supported mode", code, start)
            return None
        log.info("VIS %d: %s, offset %+.0f Hz", code, mode.name, offset)
        return start + ms(300), mode, offset

    def _vis_edge(self, cand: int, lead: float) -> float:
        """The leader-to-start-bit edge near *cand* (buffer index), to a fraction of a ms."""
        ms = self._ms
        a, b = int(cand - ms(10)), int(cand + ms(10))
        k = max(int(ms(1)), 1)
        f = np.convolve(self.freq[a - k : b + k], np.ones(k) / k, mode="same")[k:-k]
        below = np.flatnonzero(f < lead - 350)
        return float(a + below[0]) if len(below) else float(cand)

    # ---- no VIS: a train of sync pulses -----------------------------------

    def _find_sync_train(self):
        """Pick up a picture already in progress from its sync pulses.

        Looks at the last 12 s every half second for at least 5 pulses of one
        mode's sync length, spaced by whole multiples of its line period.
        Decoding starts at the latest of them, as line 0.
        """
        ms = self._ms
        if self.end - self._train_from < ms(500) or self.end - self.base < ms(3000):
            return None
        self._train_from = self.end
        span = min(int(ms(12000)), self.end - max(self._train_after, self.base))
        if span < ms(3000):
            return None
        f = self.freq[-span:]
        q = self.quality[-span:]
        k = max(int(ms(1.5)), 1)
        smooth = np.convolve(f, np.ones(k) / k, mode="same")
        # Without a VIS there is no leader to measure the tuning on, so this
        # takes syncs up to ~150 Hz off; black (1500 Hz) must stay out.
        low = (smooth < 1360) & (smooth > 950) & (q > 0.5)
        edges = np.flatnonzero(np.diff(low.astype(np.int8)))
        if low[0]:
            edges = edges[1:]
        starts, ends = edges[::2] + 1, edges[1::2] + 1
        n = min(len(starts), len(ends))
        if n < 5:
            return None
        starts, ends = starts[:n], ends[:n]
        widths = (ends - starts) * 1000.0 / self.rate
        offset0 = len(self.freq) - len(f)
        for mode in sorted(MODES, key=lambda m: -m.sync_ms):
            if self.forced is not None and mode is not self.forced:
                continue
            sel = np.abs(widths - mode.sync_ms) < max(1.5, 0.3 * mode.sync_ms)
            if sel.sum() < 5:
                continue
            t = starts[sel].astype(float)
            anchor = t[-1]
            # The sender's clock may be off by up to 0.5%: over 12 s of a slow
            # mode that moves the pulses tens of ms, so try a range of periods.
            fit = None
            for scale in np.linspace(0.995, 1.005, 41):
                period = ms(mode.line_ms) * scale
                k = (anchor - t) / period
                ok = np.abs(k - np.round(k)) * period < ms(1.5)
                if fit is None or ok.sum() > fit.sum():
                    fit, kk = ok, k
            lines = np.round(kk[fit])
            # Most lines in the span must have their pulse; this also keeps
            # Robot 72 (300 ms) from being taken for Robot 36 (150 ms).
            if fit.sum() < 5 or lines.max() < 4 or len(np.unique(lines)) < 0.7 * (lines.max() + 1):
                continue
            pulses = [(s, e) for s, e, ok in zip(starts[sel], ends[sel], fit) if ok]
            sync_f = np.median(np.concatenate([f[s + 2 : e - 2] for s, e in pulses if e - s > 4]))
            offset = float(np.clip(sync_f - SYNC_HZ, -MAX_OFFSET_HZ, MAX_OFFSET_HZ))
            line0 = anchor + offset0 + self.base - ms(mode.sync_at)
            log.info("sync pulses of %s (%d), no VIS, offset %+.0f Hz", mode.name, fit.sum(), offset)
            return line0 - ms(mode.pre_ms), mode, offset
        return None

    # ---- image -----------------------------------------------------------

    def _begin(self, start, mode: Mode, offset, now, vis=True):
        self.image = Image(mode, now, offset, np.zeros((mode.height, mode.width, 3), np.uint8), vis=vis)
        self._t0 = start + self._ms(mode.pre_ms)
        self._period = self._ms(mode.line_ms)
        self._line = 0
        self._syncs = []
        self._misses = 0
        self._redrawn = False
        self._last_good = -1
        return ("start", self.image)

    def _finish(self):
        img = self.image
        self._train_after = self.end
        img.finished = True
        img.complete = img.rows_done >= img.mode.height
        self.image = None
        return ("done", img)

    def _sync_score(self, lo, hi, width, offset):
        """Best sync-pulse start in [lo, hi) (absolute samples) and how sync-like it is."""
        a, b = int(lo) - self.base, int(hi) - self.base
        w = int(width)
        pre = max(int(self._ms(2.0)), 1)
        a = max(a, pre)
        b = min(b, len(self.freq) - w)
        if b <= a:
            return None, 0.0
        f = self.freq - offset
        s = np.clip((BLACK_HZ - f) / (BLACK_HZ - SYNC_HZ), 0.0, 1.0)
        c = np.concatenate([[0.0], np.cumsum(s, dtype=np.float64)])
        t = np.arange(a, b)
        inside = (c[t + w] - c[t]) / w
        before = (c[t] - c[t - pre]) / pre
        # A sync is a pulse of its own length: the 30 ms VIS stop bit just
        # before line 0 must not pass for one.
        after = (c[np.minimum(t + w + pre, len(c) - 1)] - c[t + w]) / pre
        score = inside - (before + after) / 2
        i = int(np.argmax(score))
        return t[i] + self.base, float(inside[i])

    def _decode_lines(self):
        img, mode = self.image, self.image.mode
        events = []
        ms = self._ms
        search = ms(max(4.0, mode.sync_ms))
        while self._line < mode.lines:
            n = self._line
            start = self._t0 + n * self._period
            expect = start + ms(mode.sync_at)
            # Wait for the whole line, and for half a line past the expected
            # pulse in case it has to be searched for.
            margin = ms(max(15.0, 2 * mode.sync_ms)) + ms(mode.sync_ms)
            if max(start + self._period, expect + self._period / 2) + margin >= self.end:
                break
            # Until a few pulses are locked, look wider: some senders leave out
            # Scottie's start sync, and the VIS timing is only good to a few ms.
            wide = search if len(self._syncs) >= 3 else ms(max(15.0, 2 * mode.sync_ms))
            found, score = self._sync_score(expect - wide, expect + wide, ms(mode.sync_ms), img.offset_hz)
            if (found is None or score <= 0.6) and self._misses >= 2:
                # Pulses missing where expected: audio was lost (a WebSDR stream
                # drops out) or the start was misjudged. Look across the whole
                # line and, if the pulse is there, carry on from it.
                found, score = self._sync_score(expect - self._period / 2, expect + self._period / 2,
                                                ms(mode.sync_ms), img.offset_hz)
                if found is not None and score > 0.8:
                    log.info("%s: line %d re-synced %+.1f ms", mode.name, n, (found - expect) * 1000 / self.rate)
                    scale = self._period / ms(mode.line_ms)
                    self._t0 = found - ms(mode.sync_at) * scale - n * self._period
                    self._syncs = []
                else:
                    found, score = None, 0.0
            if found is not None and score > 0.6:
                scale = self._period / ms(mode.line_ms)
                self._syncs.append((n, found - ms(mode.sync_at) * scale))
                self._misses = 0
                self._last_good = n
                self._refit(mode)
                start = self._t0 + n * self._period
            else:
                self._misses += 1
            self._render_line(n, start)
            self._line += 1
            if not self._redrawn and len(self._syncs) >= 10:
                # The first lines were drawn before the slope was known.
                self._redrawn = True
                for k in range(self._line):
                    self._render_line(k, self._t0 + k * self._period)
            if self._misses >= 12:
                log.info("%s: signal lost after line %d", mode.name, self._last_good)
                # The lines since the last pulse were noise.
                keep = (self._last_good + 1) * mode.rows_per_line
                img.pixels[keep:] = 0
                img.rows_done = keep
                events.append(("rows", img))
                events.append(self._finish())
                return events
        img.slant_ppm = (self._period / ms(mode.line_ms) - 1) * 1e6
        events.append(("rows", img))
        if self._line >= mode.lines:
            events.append(self._finish())
        return events

    def _refit(self, mode: Mode):
        """Line start = t0 + n * period, fitted to the recent sync pulses."""
        nominal = self._ms(mode.line_ms)
        pts = np.array(self._syncs[-40:], dtype=float)
        if len(pts) >= 4:
            n, t = pts[:, 0], pts[:, 1]
            for _ in range(2):
                period, t0 = np.polyfit(n, t, 1)
                resid = t - (t0 + period * n)
                keep = np.abs(resid) < max(self._ms(1.5), 3 * np.median(np.abs(resid)))
                if keep.sum() < 4 or keep.all():
                    break
                n, t = n[keep], t[keep]
            if abs(period / nominal - 1) < 0.02:
                self._period, self._t0 = period, t0
                return
        # Too few pulses for a slope: keep the period, re-anchor on the latest pulse.
        ln, lt = self._syncs[-1]
        self._t0 = lt - ln * self._period

    def _scan(self, c, start, comp_start_ms, dur_ms, width, offset):
        """Mean frequency of each pixel of a scan, as 0..255."""
        # Scale times inside the line by the measured clock error too.
        ms = lambda t: self._ms(t) * self._period / self._ms(self.image.mode.line_ms)  # noqa: E731
        edges = start + ms(comp_start_ms) + np.arange(width + 1) * ms(dur_ms) / width - self.base
        edges = np.clip(edges, 0, len(c) - 1)
        csum = np.interp(edges, np.arange(len(c)), c)
        mean = np.diff(csum) / np.maximum(np.diff(edges), 1e-9) - offset
        return np.clip((mean - BLACK_HZ) / (WHITE_HZ - BLACK_HZ) * 255.0, 0, 255)

    def _render_line(self, n, start):
        img, mode = self.image, self.image.mode
        c = self._cumsum()
        vals = {}
        for comp, s, d in mode.scans:
            width = 8 if comp == "SEP" else mode.width
            vals[comp] = self._scan(c, start, s, d, width, img.offset_hz)
        px = img.pixels
        if mode.color == "GBR" or mode.color == "RGB":
            px[n] = np.stack([vals["R"], vals["G"], vals["B"]], axis=-1).astype(np.uint8)
            img.rows_done = n + 1
        elif mode.color == "R72":
            px[n] = ycc_to_rgb(vals["Y"], vals["RY"], vals["BY"])
            img.rows_done = n + 1
        elif mode.color == "PD":
            px[2 * n] = ycc_to_rgb(vals["Y0"], vals["RY"], vals["BY"])
            px[2 * n + 1] = ycc_to_rgb(vals["Y1"], vals["RY"], vals["BY"])
            img.rows_done = 2 * n + 2
        elif mode.color == "R36":
            # The separator says which chroma the line carries: 1500 Hz R-Y, 2300 Hz B-Y.
            is_ry = float(np.median(vals["SEP"])) < 128
            img._y[n] = (vals["Y"], vals["C"], is_ry)
            self._robot36_rows(n)

    def _robot36_rows(self, n):
        """Colour a row once the neighbouring row has brought the other chroma."""
        img = self.image
        for row in (n - 1, n):
            if row < 0 or row not in img._y:
                continue
            y, c, is_ry = img._y[row]
            other = img._y.get(row + 1) if row == n - 1 else img._y.get(row - 1)
            if other is None or other[2] == is_ry:
                if row == n:
                    # No partner yet: grey for now, coloured when the next line comes.
                    img.pixels[row] = np.repeat(y.astype(np.uint8)[:, None], 3, axis=1)
                continue
            cr, cb = (c, other[1]) if is_ry else (other[1], c)
            img.pixels[row] = ycc_to_rgb(y, cr, cb)
        img.rows_done = n + 1
        img._y.pop(n - 2, None)


def ycc_to_rgb(y, cr, cb) -> np.ndarray:
    cr = cr - 128.0
    cb = cb - 128.0
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)


def decode_audio(audio: np.ndarray, rate: int = RATE, chunk: int = 2400) -> list[Image]:
    """Decode a whole recording; returns the images, finished."""
    dec = SSTVDecoder(rate)
    done = []
    for i in range(0, len(audio), chunk):
        done += [img for kind, img in dec.feed(audio[i : i + chunk], now=i / rate) if kind == "done"]
    done += [img for kind, img in dec.feed(np.zeros(rate * 3, np.float32)) if kind == "done"]
    done += [img for kind, img in dec.stop_image()]
    return done
