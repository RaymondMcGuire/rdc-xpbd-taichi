# Residual-Driven Chebyshev XPBD

Taichi reference implementation for the Jacobi block Neo-Hookean XPBD solver
used in the CGI 2026 TVC manuscript.

## Execution Order

### 1. Initialize the Environment

Run this first after cloning the repository:

```bat
scripts\init.bat
```

This script does three things:

- creates `.venv` with Python 3.10 if it does not already exist
- initializes/updates the `third_party\meshtaichi_patcher` submodule
- installs project dependencies with `uv sync`

After this step, verify that the local patcher can be imported:

```bat
.venv\Scripts\python.exe -c "import meshtaichi_patcher_core as m; print(m.__file__)"
```

### 2. Rebuild meshtaichi_patcher When Needed

You only need this step if `third_party\meshtaichi_patcher` was changed, updated,
or if the import verification above points to the wrong package/build:

```bat
scripts\rebuild_patcher.bat
```

This rebuilds and reinstalls the local editable `meshtaichi_patcher` package into
the existing `.venv`. Run `scripts\init.bat` before this script on a fresh clone.

### 3. Reproduce Figure 7

Run the Armadillo convergence-error comparison:

```bat
scripts\figure7_armadillo_convergence.bat
```

The script runs four methods on `assets\tetmesh\armadillo.node`:

- `NH`: split Neo-Hookean
- `NHC`: split Neo-Hookean with Chebyshev acceleration
- `BNH`: block Neo-Hookean
- `BNHC`: block Neo-Hookean with Chebyshev acceleration

It writes per-method convergence JSON/PNG files under `output\armadillo\`
and the Figure 7 comparison plot under `output\figure7_armadillo\`.

## Direct Runner

The reusable simulation entry is:

```bat
uv run python run_neohookean_usd.py --model assets/tetmesh/armadillo.node --scale 0.01 --neohookean-block --cheb --monitor-convergence --plot-frames --plot-frame-ids 300 --no-usd
```
