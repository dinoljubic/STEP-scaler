"""Headless smoke test for the full window: load flow, enable states, export."""

import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication

import main as app_module

SAMPLE = sys.argv[1]

app = QApplication([])
win = app_module.MainWindow()

assert not win.scale_button.isEnabled(), "scale button should start disabled"
assert all(not e.isEnabled() for e in win.scale_edits.values()), \
    "scale edits should start disabled"

tmp = Path(tempfile.mkdtemp())
step = tmp / "part.step"
shutil.copyfile(SAMPLE, step)

win._load_file(str(step))
assert win.scale_button.isEnabled(), "scale button should be enabled after load"
assert all(e.isEnabled() and e.text() == "1" for e in win.scale_edits.values()), \
    "scale edits should be enabled with default 1"

win.scale_edits["Y"].setText("2.5")
win._on_scale_clicked()
exported = tmp / "part_SCALED.step"
assert exported.exists(), "exported file missing"
assert exported.read_bytes() == step.read_bytes(), "dummy export should be a copy"
print("status:", win.statusBar().currentMessage())
print("OK")
