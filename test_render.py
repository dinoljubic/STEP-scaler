"""Headless smoke test: load a STEP file, render the preview, save a PNG."""

import os
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication

import main as app_module

STEP = sys.argv[1]
OUT = sys.argv[2]

app = QApplication([])
tris, colors = app_module.load_step_triangles(STEP)
print(f"triangles: {len(tris):,} (colors: {colors is not None})")

preview = app_module.PreviewWidget()
preview.resize(640, 480)
preview.set_triangles(tris, colors)
pixmap = preview._render_model()

painter_widget = preview  # also draw the gizmo onto the pixmap for inspection
from PySide6.QtGui import QPainter
p = QPainter(pixmap)
preview._draw_axes_gizmo(p)
p.end()

pixmap.save(OUT)
print(f"saved: {OUT}")
