"""STEP Scaler — load a .step file, preview it, and export a scaled copy.

The Scale button scales the B-rep geometry about the global origin and
exports it with a "_SCALED" suffix.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh
import cascadio

from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import (
    QColor,
    QDoubleValidator,
    QFont,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

STEP_EXTENSIONS = {".step", ".stp"}

# "Home" view: camera placed along (+X, -Y, +Z) looking at the origin, Z up.
# This shows the model between its top, front and right faces.
_VIEW = np.array([1.0, -1.0, 1.0]) / np.sqrt(3.0)   # toward the camera
_RIGHT = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)   # screen right
_UP = np.cross(_VIEW, _RIGHT)                        # screen up

_LIGHT = np.array([0.5, -1.0, 1.5])
_LIGHT = _LIGHT / np.linalg.norm(_LIGHT)

def _geometry_face_colors(geom):
    """Per-face RGB colors in 0..1 for one geometry, or None."""
    visual = geom.visual
    material = getattr(visual, "material", None)
    if material is not None:
        try:
            color = np.asarray(material.main_color, dtype=np.float64)[:3]
            return np.tile(color / 255.0, (len(geom.faces), 1))
        except Exception:
            pass
    try:
        face_colors = np.asarray(visual.face_colors, dtype=np.float64)
        if face_colors.ndim == 2 and len(face_colors) == len(geom.faces):
            return face_colors[:, :3] / 255.0
    except Exception:
        pass
    return None


def load_step_triangles(path):
    """Convert a STEP file to a mesh.

    Returns (triangles, colors): triangles as (n, 3, 3) in mm, colors as
    (n, 3) RGB in 0..1 or None if the file defines no colors.
    """
    fd, glb_path = tempfile.mkstemp(suffix=".glb")
    os.close(fd)
    try:
        cascadio.step_to_glb(str(path), glb_path)
        loaded = trimesh.load(glb_path)
    finally:
        try:
            os.remove(glb_path)
        except OSError:
            pass

    # Walk the scene nodes ourselves: trimesh's merged-scene visuals don't
    # reliably expose colors, but per-geometry materials do.
    if isinstance(loaded, trimesh.Scene):
        instances = []
        for node in loaded.graph.nodes_geometry:
            transform, geom_name = loaded.graph[node]
            instances.append((np.asarray(transform, dtype=np.float64),
                              loaded.geometry[geom_name]))
    else:
        instances = [(np.eye(4), loaded)]

    default = np.array([160.0, 175.0, 200.0]) / 255.0  # PreviewWidget.BODY
    triangle_parts, color_parts, any_colors = [], [], False
    for transform, geom in instances:
        tris = np.asarray(geom.triangles, dtype=np.float64)
        if tris.size == 0:
            continue
        points = tris.reshape(-1, 3) @ transform[:3, :3].T + transform[:3, 3]
        triangle_parts.append(points.reshape(-1, 3, 3))
        colors = _geometry_face_colors(geom)
        if colors is None:
            colors = np.tile(default, (len(tris), 1))
        else:
            any_colors = True
        color_parts.append(colors)

    if not triangle_parts:
        raise ValueError("No geometry found in file.")
    # cascadio keeps the STEP axes (no glTF Y-up conversion) but converts
    # units to meters; scale back to millimetres, the STEP working unit.
    triangles = np.concatenate(triangle_parts) * 1000.0
    colors = np.concatenate(color_parts) if any_colors else None
    return triangles, colors


def scale_step_file(source, target, sx, sy, sz):
    """Read a STEP file, scale it about the global origin, write it back.

    A uniform scale uses gp_Trsf, which keeps analytic surfaces (planes,
    cylinders, ...) intact. A non-uniform scale requires gp_GTrsf, which
    converts affected surfaces to B-splines.

    Reading and writing goes through XCAF documents so that color
    assignments survive: styles are collected per subshape before the
    transform and re-applied to the matching scaled subshapes using the
    transform's modification history.
    """
    # OCP is heavy to import, so only pull it in when scaling is requested.
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_GTransform,
        BRepBuilderAPI_Transform,
    )
    from OCP.gp import gp_GTrsf, gp_Mat, gp_Pnt, gp_Trsf
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPCAFControl import STEPCAFControl_Reader, STEPCAFControl_Writer
    from OCP.STEPControl import STEPControl_AsIs
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDF import TDF_LabelSequence
    from OCP.TDocStd import TDocStd_Document
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS_Compound
    from OCP.XCAFDoc import (
        XCAFDoc_ColorCurv,
        XCAFDoc_ColorSurf,
        XCAFDoc_DocumentTool,
    )
    from OCP.XCAFPrs import XCAFPrs, XCAFPrs_IndexedDataMapOfShapeStyle

    reader = STEPCAFControl_Reader()
    reader.SetColorMode(True)
    reader.SetNameMode(True)
    if reader.ReadFile(str(source)) != IFSelect_RetDone:
        raise ValueError(f"Could not read STEP file: {source}")
    src_doc = TDocStd_Document(TCollection_ExtendedString("XmlXCAF"))
    if not reader.Transfer(src_doc):
        raise ValueError(f"Could not transfer STEP data: {source}")
    src_shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(src_doc.Main())

    roots = TDF_LabelSequence()
    src_shape_tool.GetFreeShapes(roots)
    if roots.Length() == 0:
        raise ValueError("No shape found in STEP file.")

    # Flatten all root shapes into one compound and record every style
    # (color) assignment, keyed by the located subshape it applies to.
    compound = TopoDS_Compound()
    BRep_Builder().MakeCompound(compound)
    styles = XCAFPrs_IndexedDataMapOfShapeStyle()
    for i in range(1, roots.Length() + 1):
        label = roots.Value(i)
        BRep_Builder().Add(compound, src_shape_tool.GetShape_s(label))
        XCAFPrs.CollectStyleSettings_s(label, TopLoc_Location(), styles)

    if sx == sy == sz:
        trsf = gp_Trsf()
        trsf.SetScale(gp_Pnt(0.0, 0.0, 0.0), sx)
        builder = BRepBuilderAPI_Transform(compound, trsf, True)
    else:
        gtrsf = gp_GTrsf()
        gtrsf.SetVectorialPart(gp_Mat(sx, 0.0, 0.0,
                                      0.0, sy, 0.0,
                                      0.0, 0.0, sz))
        builder = BRepBuilderAPI_GTransform(compound, gtrsf, True)
    if not builder.IsDone():
        raise ValueError("Scaling transform failed.")

    # New document: the scaled compound plus the remapped colors.
    out_doc = TDocStd_Document(TCollection_ExtendedString("XmlXCAF"))
    out_shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(out_doc.Main())
    out_color_tool = XCAFDoc_DocumentTool.ColorTool_s(out_doc.Main())
    root_label = out_shape_tool.AddShape(builder.Shape(), False)

    for i in range(1, styles.Extent() + 1):
        old_shape = styles.FindKey(i)
        style = styles.FindFromIndex(i)
        try:
            new_shape = builder.ModifiedShape(old_shape)
        except RuntimeError:
            continue
        if new_shape.IsNull():
            continue
        sub_label = out_shape_tool.AddSubShape(root_label, new_shape)
        if sub_label.IsNull():
            continue
        if style.IsSetColorSurf():
            out_color_tool.SetColor(sub_label, style.GetColorSurfRGBA(),
                                    XCAFDoc_ColorSurf)
        if style.IsSetColorCurv():
            out_color_tool.SetColor(sub_label, style.GetColorCurv(),
                                    XCAFDoc_ColorCurv)

    writer = STEPCAFControl_Writer()
    writer.SetColorMode(True)
    writer.SetNameMode(True)
    writer.Transfer(out_doc, STEPControl_AsIs)
    if writer.Write(str(target)) != IFSelect_RetDone:
        raise ValueError(f"Could not write STEP file: {target}")


class PreviewWidget(QFrame):
    """Non-interactive preview: static home view plus a permanent axes gizmo."""

    BACKGROUND = QColor(38, 41, 48)
    BODY = QColor(160, 175, 200)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumSize(480, 380)
        self._triangles = None
        self._colors = None
        self._pixmap = None

    def set_triangles(self, triangles, colors=None):
        self._triangles = triangles
        self._colors = colors
        self._pixmap = None
        self.update()

    def resizeEvent(self, event):
        self._pixmap = None
        super().resizeEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.BACKGROUND)

        if self._triangles is None:
            painter.setPen(QColor(150, 150, 150))
            painter.setFont(QFont("Segoe UI", 11))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "Load or drop a .step file")
        else:
            if self._pixmap is None or self._pixmap.size() != self.size():
                self._pixmap = self._render_model()
            painter.drawPixmap(0, 0, self._pixmap)

        self._draw_axes_gizmo(painter)
        painter.end()

    def _render_model(self):
        pixmap = QPixmap(self.size())
        pixmap.fill(self.BACKGROUND)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        tris = self._triangles
        # Project every vertex onto the view plane.
        sx = tris @ _RIGHT
        sy = tris @ _UP
        depth = (tris @ _VIEW).mean(axis=1)

        # Fit the projected model into the widget with a margin.
        min_x, max_x = sx.min(), sx.max()
        min_y, max_y = sy.min(), sy.max()
        span = max(max_x - min_x, max_y - min_y, 1e-12)
        margin = 0.85
        scale = margin * min(self.width(), self.height()) / span
        cx = self.width() / 2 - scale * (min_x + max_x) / 2
        cy = self.height() / 2 + scale * (min_y + max_y) / 2

        px = sx * scale + cx
        py = -sy * scale + cy

        # Flat shading from a fixed light.
        normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        lengths[lengths == 0] = 1.0
        normals /= lengths[:, None]
        shade = 0.25 + 0.75 * np.clip(np.abs(normals @ _LIGHT), 0.0, 1.0)

        if self._colors is not None:
            base = self._colors * 255.0
        else:
            base = np.full((1, 3), 0.0) + np.array(
                [self.BODY.red(), self.BODY.green(), self.BODY.blue()])
        colors = np.clip(base * shade[:, None], 0.0, 255.0).astype(np.uint8)

        # Painter's algorithm: far triangles first.
        order = np.argsort(depth)
        for i in order:
            color = QColor(int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))
            painter.setBrush(color)
            painter.setPen(QPen(color, 0.5))  # cover hairline gaps between faces
            painter.drawConvexPolygon(QPolygonF([
                QPointF(px[i, 0], py[i, 0]),
                QPointF(px[i, 1], py[i, 1]),
                QPointF(px[i, 2], py[i, 2]),
            ]))

        painter.end()
        return pixmap

    def _draw_axes_gizmo(self, painter):
        origin = QPointF(38.0, self.height() - 38.0)
        length = 28.0
        axes = [
            (np.array([1.0, 0.0, 0.0]), QColor(220, 60, 60), "X"),
            (np.array([0.0, 1.0, 0.0]), QColor(70, 200, 70), "Y"),
            (np.array([0.0, 0.0, 1.0]), QColor(80, 120, 255), "Z"),
        ]
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        for direction, color, label in axes:
            dx = float(direction @ _RIGHT) * length
            dy = -float(direction @ _UP) * length
            end = QPointF(origin.x() + dx, origin.y() + dy)
            painter.setPen(QPen(color, 2))
            painter.drawLine(origin, end)
            painter.drawText(QPointF(origin.x() + dx * 1.45 - 4,
                                     origin.y() + dy * 1.45 + 4), label)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("STEP Scaler")
        self.setAcceptDrops(True)
        self._loaded_path = None
        self._extents = None

        self.preview = PreviewWidget()

        self.load_button = QPushButton("Load…")
        self.load_button.clicked.connect(self._on_load_clicked)

        validator = QDoubleValidator(1e-9, 1e9, 6, self)
        validator.setNotation(QDoubleValidator.StandardNotation)

        def make_axis_edits(row):
            edits = {}
            for axis in ("X", "Y", "Z"):
                row.addWidget(QLabel(f"{axis}:"))
                edit = QLineEdit()
                edit.setValidator(validator)
                edit.setFixedWidth(70)
                edit.setEnabled(False)
                edits[axis] = edit
                row.addWidget(edit)
            return edits

        # Row 1: absolute target dimensions, prefilled with the model size.
        size_row = QHBoxLayout()
        size_row.addStretch()
        size_row.addWidget(QLabel("Size (mm):"))
        self.size_edits = make_axis_edits(size_row)
        self.size_scale_button = QPushButton("Scale to size")
        self.size_scale_button.setFixedWidth(110)
        self.size_scale_button.setEnabled(False)
        self.size_scale_button.clicked.connect(self._on_scale_to_size_clicked)
        size_row.addWidget(self.size_scale_button)

        # Row 2: relative scale factors.
        scale_row = QHBoxLayout()
        scale_row.addWidget(self.load_button)
        scale_row.addStretch()
        scale_row.addWidget(QLabel("Factor:"))
        self.scale_edits = make_axis_edits(scale_row)
        self.scale_button = QPushButton("Scale")
        self.scale_button.setFixedWidth(110)
        self.scale_button.setEnabled(False)
        self.scale_button.clicked.connect(self._on_scale_clicked)
        scale_row.addWidget(self.scale_button)

        layout = QVBoxLayout()
        layout.addWidget(self.preview, stretch=1)
        layout.addLayout(size_row)
        layout.addLayout(scale_row)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Load a .step file to begin.")

    # ---- loading -------------------------------------------------------

    def _on_load_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open STEP file", "", "STEP files (*.step *.stp)")
        if path:
            self._load_file(path)

    def dragEnterEvent(self, event):
        if self._dropped_step_path(event) is not None:
            event.acceptProposedAction()

    def dropEvent(self, event):
        path = self._dropped_step_path(event)
        if path is not None:
            event.acceptProposedAction()
            self._load_file(path)

    @staticmethod
    def _dropped_step_path(event):
        urls = event.mimeData().urls()
        if len(urls) == 1 and urls[0].isLocalFile():
            path = urls[0].toLocalFile()
            if Path(path).suffix.lower() in STEP_EXTENSIONS:
                return path
        return None

    def _load_file(self, path):
        self.statusBar().showMessage(f"Loading {Path(path).name}…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            triangles, colors = load_step_triangles(path)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            self.statusBar().showMessage("Load failed.")
            QMessageBox.critical(self, "Load failed",
                                 f"Could not load {path}:\n{exc}")
            return
        QApplication.restoreOverrideCursor()

        self._loaded_path = Path(path)
        self.preview.set_triangles(triangles, colors)
        points = triangles.reshape(-1, 3)
        self._extents = points.max(axis=0) - points.min(axis=0)
        for edit in self.scale_edits.values():
            edit.setEnabled(True)
            edit.setText("1")
        for extent, edit in zip(self._extents, self.size_edits.values()):
            edit.setEnabled(True)
            edit.setText(f"{extent:.6g}")
        self.scale_button.setEnabled(True)
        self.size_scale_button.setEnabled(True)
        self.statusBar().showMessage(
            f"Loaded {self._loaded_path.name} "
            f"({len(triangles):,} triangles).")

    # ---- scaling -------------------------------------------------------

    def _read_positive_values(self, edits, what):
        values = {}
        for axis, edit in edits.items():
            try:
                value = float(edit.text().replace(",", "."))
            except ValueError:
                value = 0.0
            if value <= 0.0:
                QMessageBox.warning(self, f"Invalid {what}",
                                    f"{axis} {what} must be a positive number.")
                return None
            values[axis] = value
        return values

    def _on_scale_clicked(self):
        factors = self._read_positive_values(self.scale_edits, "scale")
        if factors is not None:
            self._export_scaled(factors)

    def _on_scale_to_size_clicked(self):
        targets = self._read_positive_values(self.size_edits, "size")
        if targets is None:
            return
        factors = {}
        for axis, extent in zip(("X", "Y", "Z"), self._extents):
            if extent <= 0.0:
                # Model is flat along this axis; no factor can change it.
                factors[axis] = 1.0
            else:
                factors[axis] = targets[axis] / extent
        self._export_scaled(factors)

    def _export_scaled(self, factors):
        source = self._loaded_path
        target = source.with_name(source.stem + "_SCALED" + source.suffix)
        self.statusBar().showMessage(f"Scaling {source.name}…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            scale_step_file(source, target,
                            factors["X"], factors["Y"], factors["Z"])
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            self.statusBar().showMessage("Export failed.")
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(
            f"Exported {target.name} (scale X={factors['X']:g} "
            f"Y={factors['Y']:g} Z={factors['Z']:g}).")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(720, 560)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
