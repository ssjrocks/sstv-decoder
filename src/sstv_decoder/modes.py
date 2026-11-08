"""SSTV mode timings.

Times are in ms from the start of a transmitted line. Frequencies: sync 1200 Hz,
black 1500 Hz, white 2300 Hz. Sources: JL Barber's "Proposal for SSTV mode
specifications" (Dayton 2000) and the timings used by slowrx and QSSTV.
"""

from dataclasses import dataclass

SYNC_HZ = 1200.0
BLACK_HZ = 1500.0
WHITE_HZ = 2300.0


@dataclass(frozen=True)
class Mode:
    name: str
    vis: int
    width: int
    height: int                 # image rows
    line_ms: float              # one transmitted line
    sync_ms: float
    sync_at: float              # where the line's sync pulse starts
    scans: tuple                # (component, start_ms, duration_ms)
    color: str                  # "GBR", "RGB", "R36", "R72", "PD"
    pre_ms: float = 0.0         # from the end of VIS to line 0 (Scottie's start sync)

    @property
    def rows_per_line(self) -> int:
        return 2 if self.color == "PD" else 1

    @property
    def lines(self) -> int:
        return self.height // self.rows_per_line

    @property
    def duration_s(self) -> float:
        return (self.pre_ms + self.lines * self.line_ms) / 1000


def _martin(name, vis, scan):
    sync, porch, sep = 4.862, 0.572, 0.572
    g = sync + porch
    b = g + scan + sep
    r = b + scan + sep
    return Mode(name, vis, 320, 256, r + scan + sep, sync, 0.0,
                (("G", g, scan), ("B", b, scan), ("R", r, scan)), "GBR")


def _scottie(name, vis, scan):
    sep, sync, porch = 1.5, 9.0, 1.5
    g = sep
    b = g + scan + sep
    s = b + scan
    r = s + sync + porch
    return Mode(name, vis, 320, 256, r + scan, sync, s,
                (("G", g, scan), ("B", b, scan), ("R", r, scan)), "GBR", pre_ms=sync)


def _pd(name, vis, width, height, scan):
    sync, porch = 20.0, 2.08
    y0 = sync + porch
    return Mode(name, vis, width, height, y0 + 4 * scan, sync, 0.0,
                (("Y0", y0, scan), ("RY", y0 + scan, scan), ("BY", y0 + 2 * scan, scan),
                 ("Y1", y0 + 3 * scan, scan)), "PD")


MODES = [
    Mode("Robot 36", 8, 320, 240, 150.0, 9.0, 0.0,
         (("Y", 12.0, 88.0), ("SEP", 100.0, 4.5), ("C", 106.0, 44.0)), "R36"),
    Mode("Robot 72", 12, 320, 240, 300.0, 9.0, 0.0,
         (("Y", 12.0, 138.0), ("RY", 156.0, 69.0), ("BY", 231.0, 69.0)), "R72"),
    _martin("Martin M1", 44, 146.432),
    _martin("Martin M2", 40, 73.216),
    _scottie("Scottie S1", 60, 138.240),
    _scottie("Scottie S2", 56, 88.064),
    _scottie("Scottie DX", 76, 345.600),
    _pd("PD 50", 93, 320, 256, 91.520),
    _pd("PD 90", 99, 320, 256, 170.240),
    _pd("PD 120", 95, 640, 496, 121.600),
    _pd("PD 160", 98, 512, 400, 195.584),
    _pd("PD 180", 96, 640, 496, 183.040),
    _pd("PD 240", 97, 640, 496, 244.480),
    _pd("PD 290", 94, 800, 616, 228.800),
    Mode("Wraase SC2-180", 55, 320, 256, 711.0225, 5.5225, 0.0,
         (("R", 6.0225, 235.0), ("G", 241.0225, 235.0), ("B", 476.0225, 235.0)), "RGB"),
]
BY_VIS = {m.vis: m for m in MODES}
BY_NAME = {m.name: m for m in MODES}
