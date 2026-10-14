"""Local desktop demonstration for recorded-video AERIS runs."""

from pathlib import Path
import os
import shutil
import sys
import tempfile


_plugin_copy = None


def ensure_qt_plugins_visible():
    """Expose Qt platform plugins through a process-local directory on macOS."""
    global _plugin_copy
    if sys.platform != "darwin":
        return
    if _plugin_copy is not None:
        return
    import PySide6

    directory = Path(PySide6.__file__).parent / "Qt/plugins/platforms"
    plugins = list(directory.glob("libq*.dylib"))
    if plugins:
        # Some managed macOS installs hide these files from Qt even when Python's
        # st_flags does not report UF_HIDDEN. Avoid mutating site-packages.
        _plugin_copy = tempfile.TemporaryDirectory(prefix="aeris-qt-platforms-")
        for path in plugins:
            shutil.copyfile(path, Path(_plugin_copy.name) / path.name)
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _plugin_copy.name


ensure_qt_plugins_visible()
