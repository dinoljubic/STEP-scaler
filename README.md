# STEP Scaler

Small desktop tool for scaling 3D models in STEP format.

## Run

```
pip install -r requirements.txt
python main.py
```

## Usage

1. Click **Load…** (or drag a `.step`/`.stp` file onto the window).
2. The model appears in the preview in a fixed isometric "home" view
   (between top, front and right). The XYZ axes gizmo (X red, Y green,
   Z blue) is always shown in the bottom-left corner.
3. The X/Y/Z scale boxes and the **Scale** button become enabled
   (default value 1). Enter scale factors and press **Scale**.
4. A file named `<original>_SCALED.step` is written next to the original.

> **Note:** scaling is not implemented yet — the export is currently an
> unmodified copy of the loaded file. The entered factors are validated
> and reported in the status bar, ready to be wired up to real geometry
> scaling later (see the `TODO` in `MainWindow._on_scale_clicked`).

## How it works

- `cascadio` (OpenCASCADE) converts the STEP file to a temporary GLB mesh.
- `trimesh` loads the mesh; triangles are rotated from glTF Y-up back to
  STEP Z-up.
- The preview is rendered with QPainter: orthographic projection from the
  (+X, −Y, +Z) octant, flat shading, painter's-algorithm depth sort. It is
  intentionally non-interactive.

## Tests

Headless smoke tests (need a sample STEP file, e.g. from KiCad's 3D models):

```
python test_render.py <sample.step> out.png   # renders the preview to a PNG
python test_app.py <sample.step>              # load flow + dummy export
```
