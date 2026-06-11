"""Headless smoke test for the full window: load flow, enable states, export.

Verifies that the exported STEP file is really scaled by re-importing it and
comparing bounding-box extents.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication, QMessageBox

import main as app_module


def _no_dialogs(*args, **kwargs):
    # Modal dialogs would block forever on the offscreen platform; fail loudly.
    raise AssertionError(f"dialog shown during test: {args}")


QMessageBox.critical = staticmethod(_no_dialogs)
QMessageBox.warning = staticmethod(_no_dialogs)


def extents(step_path):
    tris, _ = app_module.load_step_triangles(step_path)
    points = tris.reshape(-1, 3)
    return points.max(axis=0) - points.min(axis=0)


def count_colored_shapes(step_path):
    """Number of subshapes with a surface/curve color in a STEP file."""
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDF import TDF_LabelSequence
    from OCP.TDocStd import TDocStd_Document
    from OCP.TopLoc import TopLoc_Location
    from OCP.XCAFDoc import XCAFDoc_DocumentTool
    from OCP.XCAFPrs import XCAFPrs, XCAFPrs_IndexedDataMapOfShapeStyle

    reader = STEPCAFControl_Reader()
    reader.SetColorMode(True)
    assert reader.ReadFile(str(step_path)) == IFSelect_RetDone
    doc = TDocStd_Document(TCollection_ExtendedString("XmlXCAF"))
    assert reader.Transfer(doc)
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    roots = TDF_LabelSequence()
    shape_tool.GetFreeShapes(roots)
    styles = XCAFPrs_IndexedDataMapOfShapeStyle()
    for i in range(1, roots.Length() + 1):
        XCAFPrs.CollectStyleSettings_s(roots.Value(i), TopLoc_Location(), styles)
    count = 0
    for i in range(1, styles.Extent() + 1):
        style = styles.FindFromIndex(i)
        if style.IsSetColorSurf() or style.IsSetColorCurv():
            count += 1
    return count


SAMPLE = sys.argv[1]

app = QApplication([])
win = app_module.MainWindow()

assert not win.scale_button.isEnabled(), "scale button should start disabled"
assert all(not e.isEnabled() for e in win.scale_edits.values()), \
    "scale edits should start disabled"

tmp = Path(tempfile.mkdtemp())
step = tmp / "part.step"
shutil.copyfile(SAMPLE, step)

assert not win.size_scale_button.isEnabled(), \
    "scale-to-size button should start disabled"
assert all(not e.isEnabled() for e in win.size_edits.values()), \
    "size edits should start disabled"

win._load_file(str(step))
assert win.scale_button.isEnabled(), "scale button should be enabled after load"
assert all(e.isEnabled() and e.text() == "1" for e in win.scale_edits.values()), \
    "scale edits should be enabled with default 1"

original = extents(step)

assert win.size_scale_button.isEnabled(), \
    "scale-to-size button should be enabled after load"
shown = np.array([float(e.text()) for e in win.size_edits.values()])
print("model extents:", np.round(original, 4), "shown:", shown)
assert np.allclose(shown, original, rtol=1e-4), \
    f"size boxes should default to model dimensions, got {shown}"

# Non-uniform scale through the UI handler.
win.scale_edits["Y"].setText("2.5")
win.scale_edits["Z"].setText("0.5")
win._on_scale_clicked()
exported = tmp / "part_SCALED.step"
assert exported.exists(), "exported file missing"
scaled = extents(exported)
ratios = scaled / original
print("non-uniform extents ratios:", np.round(ratios, 4))
assert np.allclose(ratios, [1.0, 2.5, 0.5], rtol=0.02), \
    f"unexpected non-uniform scale ratios: {ratios}"

colors_before = count_colored_shapes(step)
colors_after = count_colored_shapes(exported)
print(f"colored shapes: {colors_before} before, {colors_after} after")
assert colors_before > 0, "sample model should have colors"
assert colors_after >= colors_before, "colors lost during scaling"

# Scale-to-size: ask for specific target dimensions.
targets = original * np.array([2.0, 1.0, 3.0])
for value, edit in zip(targets, win.size_edits.values()):
    edit.setText(f"{value:.9g}")
win._on_scale_to_size_clicked()
sized = extents(exported)  # overwrites part_SCALED.step
print("scale-to-size extents:", np.round(sized, 4),
      "targets:", np.round(targets, 4))
assert np.allclose(sized, targets, rtol=0.02), \
    f"scale-to-size missed targets: {sized} vs {targets}"

# Uniform scale through the library function (gp_Trsf path).
uniform = tmp / "part_uniform.step"
app_module.scale_step_file(step, uniform, 2.0, 2.0, 2.0)
ratios = extents(uniform) / original
print("uniform extents ratios:", np.round(ratios, 4))
assert np.allclose(ratios, [2.0, 2.0, 2.0], rtol=0.02), \
    f"unexpected uniform scale ratios: {ratios}"

print("status:", win.statusBar().currentMessage())
print("OK")
