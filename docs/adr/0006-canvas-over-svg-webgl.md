# ADR 0006: Canvas 2D for the world view

Status: accepted

## Decision
A custom Canvas 2D renderer with its own camera draws the world every animation frame from
the store; React renders only the HMI panels.

## Consequences
- Hundreds of polygons plus a 300x200 grid image redraw at 60 fps without React
  reconciliation; the grid is uploaded once per version to an offscreen canvas.
- SVG would create thousands of DOM nodes per frame; WebGL would be faster still but adds
  shader code and complexity the scene does not need.
