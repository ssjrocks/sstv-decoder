"""Entry point: logging, problem reports, then the window."""

import logging
import logging.handlers
import os
import sys
import traceback
from pathlib import Path

from sstv_decoder import APP_ID, APP_NAME, __version__, appreports


def state_dir() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or Path.home() / ".local/state"
    path = Path(base) / "sstv-decoder"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _log_exception(exc_type, exc, tb) -> None:
    # Without a hook, PyQt6 aborts the process on an exception in a slot.
    traceback.print_exception(exc_type, exc, tb)
    logging.error("uncaught exception", exc_info=(exc_type, exc, tb))


def main():
    log_file = state_dir() / "app.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.handlers.RotatingFileHandler(log_file, maxBytes=512 * 1024, backupCount=1)],
    )
    logging.info("%s %s starting", APP_NAME, __version__)
    sys.excepthook = _log_exception
    appreports.init(APP_ID, APP_NAME, __version__, state_dir(), log_file=log_file)

    from sstv_decoder.gui import run_gui

    sys.exit(run_gui(state_dir()))


if __name__ == "__main__":
    main()
