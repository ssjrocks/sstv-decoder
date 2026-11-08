"""Records what is playing to an output device, via PipeWire's pw-record.

``stream.capture.sink=true`` records an output's monitor, not a microphone.
With no target it records the default output and follows it if it changes.
"""

import json
import queue
import shutil
import subprocess
import threading

import numpy as np

from .decoder import RATE

_READ_BYTES = RATE // 10 * 2  # 0.1 s of s16 mono


def list_outputs() -> list[tuple[str, str]]:
    """(node name, description) of each audio output, from pw-dump."""
    try:
        out = subprocess.run(["pw-dump"], capture_output=True, timeout=5, check=True).stdout
        objs = json.loads(out)
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    sinks = []
    for o in objs:
        props = (o.get("info") or {}).get("props") or {}
        if props.get("media.class") == "Audio/Sink" and props.get("node.name"):
            sinks.append((props["node.name"], props.get("node.description") or props["node.name"]))
    return sorted(sinks, key=lambda s: s[1].lower())


class AudioCapture:
    """Stream an output's audio as 12 kHz mono float32 from a pw-record subprocess."""

    def __init__(self, target: str | None = None):
        self._target = target
        self._process: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._chunks: queue.Queue[bytes] = queue.Queue()
        self._stopping = False

    def start(self) -> None:
        if not shutil.which("pw-record"):
            raise RuntimeError("pw-record not found. Install it with: sudo apt install pipewire-bin")
        cmd = [
            "pw-record",
            "-P", '{ stream.capture.sink=true node.name=sstv-decoder media.name="SSTV Decoder" }',
            "--format", "s16",
            "--rate", str(RATE),
            "--channels", "1",
            "--latency", "20ms",
            "--raw",        # without this, a header is written into the stream
        ]
        if self._target:
            cmd += ["--target", self._target]
        cmd.append("-")
        # pw-record writes through stdio, which buffers 4 KB (256 ms here) on
        # a pipe; unbuffered, audio arrives every ~10 ms.
        if shutil.which("stdbuf"):
            cmd = ["stdbuf", "-o0"] + cmd
        self._stopping = False
        self._process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        while True:
            data = self._process.stdout.read1(_READ_BYTES)
            if not data:
                break
            self._chunks.put(data)

    def read(self) -> np.ndarray:
        """All audio captured since the last call (may be empty).

        Raises RuntimeError if pw-record has exited.
        """
        parts = []
        while True:
            try:
                parts.append(self._chunks.get_nowait())
            except queue.Empty:
                break
        if not parts:
            if self._process.poll() is not None and not self._stopping:
                stderr = self._process.stderr.read().decode(errors="replace")
                raise RuntimeError(
                    f"pw-record exited with code {self._process.returncode}: "
                    f"{stderr.strip() or '(no error output)'}"
                )
            return np.zeros(0, dtype=np.float32)
        data = b"".join(parts)
        # An odd byte can only come from a pipe read splitting a sample.
        usable = len(data) // 2 * 2
        if usable != len(data):
            self._chunks.queue.appendleft(data[usable:])
        return np.frombuffer(data[:usable], dtype=np.int16).astype(np.float32) / 32768.0

    def stop(self) -> None:
        self._stopping = True
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._thread.join(timeout=2)
            self._process.stdout.close()
            self._process.stderr.close()
            self._process = None
