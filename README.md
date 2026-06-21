# RDC-XPBD Taichi

Taichi implementation of a residual-driven Chebyshev accelerated Jacobi
block Neo-Hookean XPBD solver.

## Environment

- Python `>=3.10,<3.11`
- `uv` for dependency resolution and running commands
- Taichi `1.6.0`
- A modified MeshTaichi Patcher submodule for Taichi mesh metadata:
  https://github.com/RaymondMcGuire/meshtaichi_patcher

## Quickstart

```bat
git clone --recursive https://github.com/RaymondMcGuire/rdc-xpbd-taichi.git
cd rdc-xpbd-taichi
scripts\init.bat
scripts\demo.bat
```

The demo writes a USD animation to `output\`.
