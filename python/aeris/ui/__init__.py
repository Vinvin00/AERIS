"""Local desktop demonstration for recorded-video AERIS runs."""

from pathlib import Path
import os
import stat
import sys


def ensure_qt_plugins_visible():
    """Work around managed macOS installs that mark Qt platform plugins hidden."""
    if sys.platform != "darwin":
        return
    import PySide6

    directory = Path(PySide6.__file__).parent / "Qt/plugins/platforms"
    for path in directory.glob("libq*.dylib"):
        flags = path.stat().st_flags
        if flags & stat.UF_HIDDEN:
            try:
                os.chflags(path, flags & ~stat.UF_HIDDEN)
            except OSError:
                pass


ensure_qt_plugins_visible()
