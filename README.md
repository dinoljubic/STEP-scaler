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
3. The scale controls become enabled, in two rows:
   - **Size (mm)** boxes, prefilled with the model's bounding-box
     dimensions. Type target dimensions and press **Scale to size** to
     scale the model to those dimensions without computing factors.
   - **Factor** boxes (default 1). Enter relative scale factors and press
     **Scale**.
4. Either button writes `<original>_SCALED.step` next to the original,
   scaled about the global origin.

## How it works

- `cascadio` (OpenCASCADE) converts the STEP file to a temporary GLB mesh
  for the preview; `trimesh` loads it. Vertices keep the original STEP
  coordinates.
- The preview is rendered with QPainter: orthographic projection from the
  (+X, −Y, +Z) octant, flat shading, painter's-algorithm depth sort. It is
  intentionally non-interactive.
- Scaling is done on the real B-rep geometry via OpenCASCADE
  (`cadquery-ocp`): uniform scales use `gp_Trsf` and keep analytic surfaces
  intact; non-uniform scales use `gp_GTrsf`/`BRepBuilderAPI_GTransform`,
  which converts affected surfaces to B-splines (unavoidable — e.g. a
  non-uniformly scaled cylinder is no longer a cylinder).
- Colors survive scaling: the file is read into an XCAF document
  (`STEPCAFControl`), color styles are collected per subshape, and after
  the transform they are re-applied to the matching scaled subshapes via
  the transform's modification history. The preview shows the same colors,
  taken from the GLB materials. Assembly structure and part names are not
  preserved (the output is a flat compound).

## Tests

Headless smoke tests (need a sample STEP file, e.g. from KiCad's 3D models):

```
python test_render.py <sample.step> out.png   # renders the preview to a PNG
python test_app.py <sample.step>              # load flow + dummy export
```
