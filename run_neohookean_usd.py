import argparse
import time

import numpy as np
import taichi as ti

import convergence_report
import usd_export
import xpbd_gpu.neohookean_solver as xpbd
from xpbd_gpu.sdf import PlaneSdf, SphereSdf


parser = argparse.ArgumentParser(description="Run Neo-Hookean XPBD.")

parser.add_argument("--model", default="assets/tetmesh/cow.node")
parser.add_argument("--scale", default=1.0, type=float)
parser.add_argument("--arch", default="gpu", choices=["cpu", "gpu"])

parser.add_argument("--frames", default=300, type=int)
parser.add_argument("--fps", default=60, type=int)
parser.add_argument("--dt", default=0.01, type=float)
parser.add_argument("--substeps", default=1, type=int)
parser.add_argument("--iterations", default=200, type=int)

parser.add_argument("--young", default=5e8, type=float)
parser.add_argument("--poisson", default=0.4999, type=float)
parser.add_argument("--hydro-relax", default=1.2, type=float)
parser.add_argument("--dev-relax", default=1.2, type=float)

method_group = parser.add_mutually_exclusive_group()
method_group.add_argument(
    "--block-neohookean",
    dest="block_neohookean",
    action="store_true",
    default=True,
    help="Use the block Neo-Hookean constraint solve (default)",
)
method_group.add_argument(
    "--split-neohookean",
    dest="block_neohookean",
    action="store_false",
    help="Use the decoupled split Neo-Hookean baseline",
)

cheb_group = parser.add_mutually_exclusive_group()
cheb_group.add_argument(
    "--cheb",
    dest="cheb",
    action="store_true",
    default=True,
    help="Use residual-driven dynamic Chebyshev acceleration (default)",
)
cheb_group.add_argument(
    "--no-cheb",
    dest="cheb",
    action="store_false",
    help="Disable dynamic Chebyshev acceleration",
)
parser.add_argument("--cheb-warmup", default=5, type=int)
parser.add_argument("--cheb-gamma", default=0.666, type=float)

parser.add_argument("--optimize-usd", action="store_true")
parser.add_argument("--use-binary", action="store_true", default=True)
parser.add_argument("--streaming", action="store_true")
parser.add_argument("--usd-root-prim", default="/SimAsset")
parser.add_argument("--no-usd", action="store_true")

parser.add_argument("--monitor-convergence", action="store_true")
parser.add_argument("--print-convergence", action="store_true")
parser.add_argument("--plot-frames", action="store_true")
parser.add_argument("--plot-frame-ids", type=int, nargs="+", default=None)
parser.add_argument("--export-csv", action="store_true")

args = parser.parse_args()

physics_dt = args.dt
fps = args.fps
solver_frame_dt = 1.0 / fps
total_frames = args.frames

ti.init(arch=getattr(ti, args.arch))

sdf = [
    PlaneSdf(point=[0.0, -2.0, 0.0], dir=[0.0, 1.0, 0.0], fixed=True),
    SphereSdf(center=[0.0, -1.0, 0.0], radius=0.3, fixed=True),
]

solver = xpbd.XPBDNeoHookeanSolver(
    rest_pose=args.model,
    sdf=sdf,
    scale=args.scale,
    offset=(0.0, 1.0, 0.0),
    hydrostatic_relaxation=args.hydro_relax,
    deviatoric_relaxation=args.dev_relax,
    frame_dt=solver_frame_dt,
    dt=physics_dt,
    substeps_num=args.substeps,
    constraints_iter=args.iterations,
    young_modulus=args.young,
    poisson_ratio=args.poisson,
    cheb_enable=args.cheb,
    cheb_warmup=args.cheb_warmup,
    cheb_gamma=args.cheb_gamma,
    block_neohookean_enable=args.block_neohookean,
    monitor_convergence=args.monitor_convergence,
)

output_dir, run_output_dir, usd_output_path = usd_export.build_output_paths(
    args.model,
    block_neohookean=args.block_neohookean,
    dynamic_cheby=args.cheb,
    young=args.young,
    use_binary=args.use_binary,
)
usd_renderer = usd_export.create_renderer(args, usd_output_path, total_frames, fps)
usd_export.add_solver_mesh(usd_renderer, solver)
usd_export.add_sdfs(usd_renderer, sdf)

if args.monitor_convergence and args.plot_frames:
    conv_plot_dir = convergence_report.setup_convergence_callback(
        solver,
        run_output_dir,
        frame_ids=args.plot_frame_ids,
        verbose=args.print_convergence,
    )
else:
    conv_plot_dir = None

solver_times = []
usd_export_times = []
total_start_time = time.time()

print("=" * 70)
print("NEO-HOOKEAN XPBD SIMULATION")
print("=" * 70)
print(f"Model: {args.model}")
print(f"Output directory: {run_output_dir}")
print(f"Architecture: {args.arch.upper()}")
print(f"Method: {'Block Neo-Hookean' if args.block_neohookean else 'Split Neo-Hookean'}")
print(f"Dynamic Chebyshev: {'ON' if args.cheb else 'OFF'}")
print(f"Frames: {total_frames}, substeps: {args.substeps}, iterations: {args.iterations}")
print(f"Young's modulus: {args.young:.2e}, Poisson ratio: {args.poisson}")
print(f"USD export: {'OFF' if args.no_usd else usd_output_path}")
print("=" * 70)

for frame in range(total_frames):
    if frame % 10 == 0:
        progress = frame / total_frames * 100
        print(f"\n[Frame {frame}/{total_frames}] ({progress:.1f}%)")

    solver_start = time.time()
    solver.solve()
    solver_times.append(time.time() - solver_start)

    if args.monitor_convergence and args.print_convergence and not args.plot_frames:
        convergence_report.print_frame_convergence(solver, frame, args.substeps)

    if usd_renderer is not None:
        usd_start = time.time()
        usd_renderer.render()
        usd_export_times.append(time.time() - usd_start)

print("\n" + "=" * 70)
if usd_renderer is not None:
    print("Saving USD file...")
    save_start = time.time()
    usd_renderer.save()
    save_time = time.time() - save_start
else:
    save_time = 0.0
total_time = time.time() - total_start_time

avg_solver_time = np.mean(solver_times)
total_solver_time = np.sum(solver_times)

print("\nPERFORMANCE")
print("=" * 70)
print(f"Total runtime: {total_time:.3f} s")
print(f"Solver total: {total_solver_time:.3f} s")
print(f"Solver per frame: {avg_solver_time:.4f} s")
print(f"Solver FPS: {1.0 / avg_solver_time:.2f}")
print(f"USD save time: {save_time:.3f} s")

if usd_renderer is not None and usd_export_times:
    print(f"USD export total: {np.sum(usd_export_times):.3f} s")
    print(f"USD file size: {usd_renderer.get_file_size_mb():.1f} MB")

total_iterations = total_frames * args.substeps * args.iterations
print(f"Total constraint iterations: {total_iterations:,}")
print(f"Time per iteration: {total_solver_time / total_iterations * 1e6:.2f} us")

convergence_report.print_convergence_analysis(
    solver,
    iterations=args.iterations,
    run_output_dir=run_output_dir,
    export_csv=args.export_csv,
)

print("\nOUTPUTS")
print("=" * 70)
if usd_renderer is not None:
    print(f"USD animation: {usd_output_path}")
if conv_plot_dir is not None:
    print(f"Convergence plots: {conv_plot_dir}")
print("=" * 70)
