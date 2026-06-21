import os
import time
import taichi as ti
import numpy as np
import xpbd_gpu.neohookean_solver as xpbd
from xpbd_gpu.sdf import *
from utils import renderer
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(
    description='XPBD Neo-Hookean Solver with USD Export and Convergence Monitoring')

# Mesh and scene
parser.add_argument('--model', default="assets/tetmesh/cube.1.node",
                    help='Path to tetrahedral mesh file')
parser.add_argument('--scale', default=1.0, type=float,
                    help='Uniform scale applied to the input mesh')
parser.add_argument('--arch', default='gpu', choices=['cpu', 'gpu'],
                    help='Taichi architecture')

# Simulation parameters
parser.add_argument('--frames', default=300, type=int,
                    help='Number of frames to simulate')
parser.add_argument('--fps', default=60, type=int,
                    help='Target frames per second')
parser.add_argument('--dt', default=0.01, type=float,
                    help='Physics time step')
parser.add_argument('--substeps', default=20, type=int,
                    help='Number of substeps per frame')
parser.add_argument('--iterations', default=10, type=int,
                    help='Constraint solver iterations per substep')

# Material parameters
parser.add_argument('--young', default=5e8, type=float,
                    help='Young\'s modulus')
parser.add_argument('--poisson', default=0.4999, type=float,
                    help='Poisson ratio')
parser.add_argument('--hydro-relax', default=1.2, type=float,
                    help='Hydrostatic constraint relaxation factor')
parser.add_argument('--dev-relax', default=1.2, type=float,
                    help='Deviatoric constraint relaxation factor')

parser.add_argument('--neohookean-block', action='store_true',
                    help='Enable Solver with Neo-Hookean block constraints')

# Chebyshev acceleration
parser.add_argument('--cheb', action='store_true',
                    help='Enable Chebyshev acceleration')
parser.add_argument('--cheb-rho', default=0.999, type=float,
                    help='Chebyshev spectral radius')
parser.add_argument('--dynamic-cheb-rho', dest='dynamic_cheb_rho', action='store_true',
                    help='Use adaptive rho_hat for Chebyshev omega update (default)')
parser.add_argument('--static-cheb-rho', dest='dynamic_cheb_rho', action='store_false',
                    help='Use fixed --cheb-rho for Chebyshev omega update')
parser.add_argument('--cheb-warmup', default=5, type=int,
                    help='Chebyshev warmup iterations')
parser.add_argument('--cheb-gamma', default=0.666, type=float,
                    help='Chebyshev relaxation parameter')
parser.set_defaults(dynamic_cheb_rho=True)

# USD export options
parser.add_argument('--optimize-usd', action='store_true',
                    help='Use optimized USD renderer for large files')
parser.add_argument('--use-binary', action='store_true',
                    default=True, help='Use binary USD format (.usdc)')
parser.add_argument('--streaming', action='store_true',
                    help='Enable streaming saves to reduce memory usage')
parser.add_argument('--usd-root-prim', default='/SimAsset',
                    help='Root prim path for exported simulation data (for easy referencing/composition)')
parser.add_argument('--no-usd', action='store_true',
                    help='Disable USD export for convergence-only runs')

# Convergence monitoring
parser.add_argument('--monitor-convergence', action='store_true',
                    help='Enable constraint convergence monitoring')
parser.add_argument('--print-convergence', action='store_true',
                    help='Print convergence info during simulation (verbose)')
parser.add_argument('--plot-frames', action='store_true',
                    help='Plot convergence curves for each substep (generates many plots)')
parser.add_argument('--plot-frame-ids', type=int, nargs='+', default=None,
                    help='Selected frame IDs to plot (e.g., --plot-frame-ids 0 1 5), default: ALL frames')
parser.add_argument('--export-csv', action='store_true',
                    help='Export convergence data to CSV file')

# Rho monitoring
parser.add_argument('--monitor-rho', action='store_true',
                    help='Enable rho tracking (rho_hat/r_hat/omega)')
parser.add_argument('--plot-rho', action='store_true',
                    help='Plot rho/r_hat/omega curves for each substep (generates many plots)')
parser.add_argument('--plot-rho-frame-ids', type=int, nargs='+', default=None,
                    help='Selected frame IDs to plot for rho (e.g., --plot-rho-frame-ids 0 1 5), default: ALL frames')

args = parser.parse_args()

physics_dt = args.dt
fps = args.fps
solver_frame_dt = 1.0 / fps

total_frames = args.frames

ti.init(arch=getattr(ti, args.arch))

# Example SDF configurations (uncomment one to test):

# Fixed hang points:
# sdf = HangSdfModel(np.array([[0.025, 0.075, 0.075],
#                              [0.075, 0.075, 0.075],
#                              [0.025, 0.025, 0.075],
#                              [0.075, 0.025, 0.075]]))

# Ground plane (y=0, normal pointing up):
# sdf = PlaneSdf(point=[0.0, -2.0, 0.0], dir=[0.0, 1.0, 0.0], fixed=True)

# Sphere obstacle:
# sdf = SphereSdf(center=[0.0, -1.0, 0.0], radius=0.3, fixed=True)

# Cube obstacle:
# sdf = CubeSdf(center=[0.5, 0.2, 0.5], size=[0.2, 0.1, 0.2], fixed=True)

# Multiple obstacles example (sequential collision projection):
sdf = [
    PlaneSdf(point=[0.0, -2.0, 0.0], dir=[0.0, 1.0, 0.0], fixed=True),
    SphereSdf(center=[0.0, -1.0, 0.0], radius=0.3, fixed=True),
]


solver = xpbd.XPBDNeoHookeanSolver(rest_pose=args.model,
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


                                   cheb_rho=args.cheb_rho,
                                   dynamic_cheb_rho=args.dynamic_cheb_rho,
                                   cheb_gamma=args.cheb_gamma,

                                   neohookean_block_enable=args.neohookean_block,
                                   monitor_convergence=args.monitor_convergence,
                                   monitor_rho=args.monitor_rho,
                                   )


# Build output directory from model prefix:
# e.g. assets/tetmesh/cow.node -> output/cow/
model_name = Path(args.model).name
model_prefix = model_name.split(
    '.')[0] if '.' in model_name else Path(model_name).stem
if not model_prefix:
    model_prefix = "model"
output_dir = os.path.join(os.getcwd(), 'output', model_prefix)

# Select appropriate USD format and renderer
solver_tag = 'block_neohookean' if args.neohookean_block else 'neohookean'
cheb_tag = 'cheb' if args.cheb else 'no_cheb'
rho_mode_tag = 'rho_dyn' if args.dynamic_cheb_rho else 'rho_static'
young_tag = f"young_{args.young:.3e}".replace('+', '').replace('.', 'p')
usd_ext = 'usdc' if args.use_binary else 'usda'
usd_filename = f'xpbd_{solver_tag}_{cheb_tag}_{rho_mode_tag}_{young_tag}.{usd_ext}'
usd_output_path = os.path.join(output_dir, usd_filename)
os.makedirs(output_dir, exist_ok=True)

# Store plots/analysis under a config-specific folder to avoid overwriting
run_output_dir = os.path.join(output_dir, Path(usd_filename).stem)
os.makedirs(run_output_dir, exist_ok=True)

usd_renderer = None
if args.no_usd:
    print("USD export disabled (--no-usd)")
elif args.optimize_usd:
    print("Using OptimizedUSDMeshRenderer for large file handling")
    usd_renderer = renderer.OptimizedUSDMeshRenderer(
        usd_output_path, total_frames, fps, root_prim_path=args.usd_root_prim)
else:
    usd_renderer = renderer.USDMeshRenderer(usd_output_path, total_frames, fps,
                                            use_binary=args.use_binary,
                                            enable_streaming=args.streaming,
                                            root_prim_path=args.usd_root_prim)

if usd_renderer is not None:
    usd_renderer.add_dynamic_mesh(
        solver.mesh.verts.x, solver.indices, 'deformable_mesh')


def add_sdf_to_usd(usd_renderer, sdf_obj, name_suffix=''):
    """Export selected SDF obstacle as a separate USD prim."""
    if isinstance(sdf_obj, SphereSdf):
        center = sdf_obj.center[None].to_numpy()
        radius = float(sdf_obj.radius[None])
        usd_renderer.add_static_sphere(
            center=center, radius=radius, name=f'sdf_sphere{name_suffix}')
    elif isinstance(sdf_obj, CubeSdf):
        center = sdf_obj.center[None].to_numpy()
        size = (sdf_obj.half_size[None].to_numpy() * 2.0).astype(np.float32)
        usd_renderer.add_static_cube(
            center=center, size=size, name=f'sdf_cube{name_suffix}')
    elif isinstance(sdf_obj, PlaneSdf):
        point = sdf_obj.point[None].to_numpy()
        normal = sdf_obj.dir[None].to_numpy()
        usd_renderer.add_static_plane(
            point=point, normal=normal, extent=4.0, name=f'sdf_plane{name_suffix}')
    else:
        print(f"SDF type {type(sdf_obj).__name__} has no USD export hook yet.")


def add_sdfs_to_usd(usd_renderer, sdf_config):
    if isinstance(sdf_config, (list, tuple)):
        for i, sdf_obj in enumerate(sdf_config):
            add_sdf_to_usd(usd_renderer, sdf_obj, name_suffix=f'_{i}')
    else:
        add_sdf_to_usd(usd_renderer, sdf_config)


if usd_renderer is not None:
    add_sdfs_to_usd(usd_renderer, sdf)

# Performance statistics
solver_times = []
usd_export_times = []
total_start_time = time.time()

print("=" * 70)
print("XPBD NEO-HOOKEAN SOLVER - SIMULATION START")
print("=" * 70)
print(f"\nOutput directory: {run_output_dir}")
print(f"Architecture: {args.arch.upper()}")
if args.no_usd:
    print("USD Export: DISABLED")
else:
    print(f"USD animation: {usd_output_path}")
    print(f"USD Format: {'Binary (.usdc)' if args.use_binary else 'Text (.usda)'}")
    print(f"USD Renderer: {'Optimized' if args.optimize_usd else 'Standard'}")
    print(f"Streaming: {'Enabled' if args.streaming else 'Disabled'}")
    print(f"USD Root Prim: {args.usd_root_prim}")

print(f"\n--- MESH INFORMATION ---")
print(f"Scale: {args.scale}")
print(f"Vertices: {len(solver.mesh.verts)}")
print(f"Tetrahedra: {len(solver.mesh.cells)}")
print(f"Faces: {len(solver.mesh.faces)}")
print(f"Edges: {len(solver.mesh.edges)}")

print(f"\n--- SIMULATION PARAMETERS ---")
print(f"Total frames: {total_frames}")
print(f"Target FPS: {fps}")
print(f"Frame dt: {solver_frame_dt:.6f} s")
print(f"Physics dt: {physics_dt:.6f} s")
print(f"Substeps per frame: {args.substeps}")
print(f"Total substeps: {total_frames * args.substeps}")

print(f"\n--- CONSTRAINT SOLVER ---")
print(f"Iterations per substep: {args.iterations}")
print(f"Total iterations per frame: {args.substeps * args.iterations}")
print(f"Hydrostatic relaxation: {args.hydro_relax}")
print(f"Deviatoric relaxation: {args.dev_relax}")
print(f"Chebyshev acceleration: {'ENABLED' if args.cheb else 'DISABLED'}")
if args.cheb:
    print(f"  Spectral radius (rho): {args.cheb_rho}")
    print(
        f"  Rho mode: {'Dynamic (rho_hat)' if args.dynamic_cheb_rho else 'Static (fixed cheb_rho)'}")
    print(f"  Warmup iterations: {args.cheb_warmup}")

print(f"\n--- MATERIAL PROPERTIES ---")
print(f"Young's modulus: {args.young:.2e} Pa")
print(f"Poisson ratio: {args.poisson}")
# Calculate and show Lamé parameters
lame_mu = args.young / (2.0 * (1.0 + args.poisson))
lame_lambda = 2.0 * args.poisson / (1.0 - 2.0 * args.poisson) * lame_mu
print(f"First Lamé parameter (λ): {lame_lambda:.2e} Pa")
print(f"Second Lamé parameter (μ): {lame_mu:.2e} Pa")
bulk_modulus = lame_lambda + 2.0 * lame_mu / 3.0
print(f"Bulk modulus (K): {bulk_modulus:.2e} Pa")

print(f"\n--- CONVERGENCE MONITORING ---")
print(f"Status: {'ENABLED' if args.monitor_convergence else 'DISABLED'}")
if args.monitor_convergence:
    print(f"Print during sim: {'Yes' if args.print_convergence else 'No'}")
    print(f"Plot substeps: {'Yes' if args.plot_frames else 'No'}")
    print(f"Export CSV: {'Yes' if args.export_csv else 'No'}")
    if args.plot_frames:
        if args.plot_frame_ids:
            print(f"  Frame IDs to plot: {args.plot_frame_ids}")
        else:
            print(f"  Frame IDs to plot: ALL frames")

print(f"\n--- RHO MONITORING ---")
print(f"Status: {'ENABLED' if args.monitor_rho else 'DISABLED'}")
if args.monitor_rho:
    print(f"Plot substeps: {'Yes' if args.plot_rho else 'No'}")
    if args.plot_rho:
        if args.plot_rho_frame_ids:
            print(f"  Frame IDs to plot: {args.plot_rho_frame_ids}")
        else:
            print(f"  Frame IDs to plot: ALL frames")

# Estimate expected file size
estimated_size_mb = (len(solver.mesh.verts) * 3 * 4 *
                     # 3 coords * 4 bytes * frames
                     total_frames) / (1024 * 1024)
print(f"\n--- FILE SIZE ESTIMATE ---")
if args.no_usd:
    print("USD export disabled; no animation file will be written.")
else:
    print(f"Estimated USD size: ~{estimated_size_mb:.1f} MB (vertex data only)")
if (not args.no_usd) and estimated_size_mb > 1000:
    print("WARNING: Large file expected! Consider using --optimize-usd flag")

print("\n" + "=" * 70)
print("SIMULATION RUNNING...")
print("=" * 70)

# Setup convergence plotting callback if requested
if args.monitor_convergence and args.plot_frames:
    # Create output directory for substep plots
    conv_plot_dir = os.path.join(run_output_dir, 'convergence_plots')
    os.makedirs(conv_plot_dir, exist_ok=True)

    # Determine which frames to plot (None means all frames)
    plot_frame_ids_set = set(
        args.plot_frame_ids) if args.plot_frame_ids else None

    def convergence_callback(frame, substep, convergence_data):
        """Callback to plot convergence after each substep"""
        # If plot_frame_ids_set is None, plot all frames; otherwise only plot selected frames
        if plot_frame_ids_set is None or frame in plot_frame_ids_set:
            plot_path = os.path.join(
                conv_plot_dir, f'f{frame:04d}_s{substep:02d}.png')
            solver.plot_substep_convergence(
                frame, substep, convergence_data, plot_path)
            if args.print_convergence:
                print(f"  └─ Saved: {plot_path}")

    solver.convergence_callback = convergence_callback

    if plot_frame_ids_set is None:
        print(
            f"Convergence plots will be saved to: {conv_plot_dir}/ (ALL frames)")
    else:
        print(
            f"Convergence plots will be saved to: {conv_plot_dir}/ (frames: {sorted(plot_frame_ids_set)})")

# Setup rho plotting callback if requested
if args.monitor_rho and args.plot_rho and hasattr(solver, 'plot_substep_rho'):
    rho_plot_dir = os.path.join(run_output_dir, 'rho_plots')
    os.makedirs(rho_plot_dir, exist_ok=True)

    plot_rho_frame_ids_set = set(
        args.plot_rho_frame_ids) if args.plot_rho_frame_ids else None

    def rho_callback(frame, substep, rho_data):
        """Callback to plot rho after each substep"""
        if plot_rho_frame_ids_set is None or frame in plot_rho_frame_ids_set:
            plot_path = os.path.join(
                rho_plot_dir, f'f{frame:04d}_s{substep:02d}.png')
            solver.plot_substep_rho(
                frame, substep, rho_data, plot_path)
            print(f"  Saved rho plot: {plot_path}")

    solver.rho_callback = rho_callback

    if plot_rho_frame_ids_set is None:
        print(f"Rho plots will be saved to: {rho_plot_dir}/ (ALL frames)")
    else:
        print(
            f"Rho plots will be saved to: {rho_plot_dir}/ (frames: {sorted(plot_rho_frame_ids_set)})")

for frame in range(total_frames):
    # Progress printing
    if frame % 10 == 0:
        progress = frame / total_frames * 100
        print(f"\n[Frame {frame}/{total_frames}] ({progress:.1f}%)")

    # Time solver
    solver_start = time.time()
    solver.solve()
    solver_time = time.time() - solver_start
    solver_times.append(solver_time)

    # Print convergence info for this frame if requested
    if args.monitor_convergence and args.print_convergence and not args.plot_frames:
        # Only print if not plotting (plotting callback will print)
        frame_data = [d for d in solver.convergence_history
                      if d['frame'] == frame]
        if frame_data:
            # Show summary of last substep
            last_substep_data = [
                d for d in frame_data if d['substep'] == args.substeps - 1]
            if last_substep_data:
                last_iter = last_substep_data[-1]
                print(f"  Final substep {args.substeps-1}, Iter {last_iter['iter']}: "
                      f"t_iter={last_iter['iter_compute_time_sec']*1e3:.3f} ms, "
                      f"H_err={last_iter['hydro_error']:.3e}, "
                      f"D_err={last_iter['dev_error']:.3e}")

    if usd_renderer is not None:
        usd_start = time.time()
        usd_renderer.render()
        usd_time = time.time() - usd_start
        usd_export_times.append(usd_time)

print("\n" + "=" * 70)
if usd_renderer is not None:
    print("Simulation complete! Saving USD file...")
    save_start = time.time()
    usd_renderer.save()
    save_time = time.time() - save_start
else:
    print("Simulation complete! USD export was disabled.")
    save_time = 0.0
total_time = time.time() - total_start_time

if usd_renderer is not None:
    print(f"\nUSD file saved to: {usd_output_path}")
    print("View with: usdview, Blender USD importer, or Omniverse")

# Performance statistics
print("\n" + "=" * 70)
print("PERFORMANCE STATISTICS")
print("=" * 70)
print(f"Architecture: {args.arch.upper()}")
print(f"\nTiming Breakdown:")
print(f"  Total runtime: {total_time:.3f} s")
print(f"  USD save time: {save_time:.3f} s")
print(f"  Effective FPS: {total_frames/total_time:.2f}")
print(f"  Time per frame: {total_time/total_frames:.4f} s")

print("\nSolver Performance:")
avg_solver_time = np.mean(solver_times)
min_solver_time = np.min(solver_times)
max_solver_time = np.max(solver_times)
total_solver_time = np.sum(solver_times)
print(
    f"  Total: {total_solver_time:.3f} s ({total_solver_time/total_time*100:.1f}% of total)")
print(
    f"  Per frame: {avg_solver_time:.4f} s (avg), {min_solver_time:.4f} s (min), {max_solver_time:.4f} s (max)")
print(f"  Solver FPS: {1.0/avg_solver_time:.2f}")

if usd_renderer is not None:
    print("\nUSD Export Performance:")
    avg_usd_time = np.mean(usd_export_times)
    total_usd_time = np.sum(usd_export_times)
    print(
        f"  Total: {total_usd_time:.3f} s ({total_usd_time/total_time*100:.1f}% of total)")
    print(f"  Per frame: {avg_usd_time:.4f} s (avg)")

print("\nComputational Load:")
total_iterations = total_frames * args.substeps * args.iterations
print(f"  Total constraint iterations: {total_iterations:,}")
print(f"  Iterations per second: {total_iterations/total_solver_time:.0f}")
print(f"  Time per iteration: {total_solver_time/total_iterations*1e6:.2f} μs")

if usd_renderer is not None:
    print("\nUSD File Info:")
    final_file_size = usd_renderer.get_file_size_mb() if hasattr(
        usd_renderer, 'get_file_size_mb') else 0
    print(f"  Size: {final_file_size:.1f} MB")
    print(f"  Format: {'Binary (.usdc)' if args.use_binary else 'Text (.usda)'}")

print("=" * 70)

# Convergence analysis
if args.monitor_convergence:
    print("\n" + "=" * 70)
    print("CONSTRAINT CONVERGENCE ANALYSIS")
    print("=" * 70)

    # Get and print statistics
    stats = solver.get_convergence_stats()
    if stats:
        print(f"\nData Collection:")
        print(f"  Total records: {stats['total_records']:,}")
        print(f"  Substeps monitored: {stats['num_substeps']}")

        print(f"\n--- Hydrostatic Constraint: C_H = det(F) - 1 ---")
        print(f"  Mean final error: {stats['hydro']['mean_final']:.6e}")
        print(f"  Max final error:  {stats['hydro']['max_final']:.6e}")
        print(f"  Min final error:  {stats['hydro']['min_final']:.6e}")

        print(f"\n--- Deviatoric Constraint: C_D = ||F||² - 3 ---")
        print(f"  Mean final error: {stats['dev']['mean_final']:.6e}")
        print(f"  Max final error:  {stats['dev']['max_final']:.6e}")
        print(f"  Min final error:  {stats['dev']['min_final']:.6e}")

        # Convergence rate analysis (compare first vs last iteration)
        print(f"\n--- Convergence Rate Analysis ---")
        first_iter_data = [
            d for d in solver.convergence_history if d['iter'] == 0]
        last_iter_data = [
            d for d in solver.convergence_history if d['iter'] == args.iterations - 1]

        if first_iter_data and last_iter_data:
            avg_hydro_first = np.mean([d['hydro_error']
                                      for d in first_iter_data])
            avg_hydro_last = np.mean([d['hydro_error']
                                     for d in last_iter_data])
            avg_dev_first = np.mean([d['dev_error'] for d in first_iter_data])
            avg_dev_last = np.mean([d['dev_error'] for d in last_iter_data])

            hydro_reduction = avg_hydro_first / \
                avg_hydro_last if avg_hydro_last > 0 else float('inf')
            dev_reduction = avg_dev_first / \
                avg_dev_last if avg_dev_last > 0 else float('inf')

            print(
                f"  Hydrostatic: {avg_hydro_first:.3e} → {avg_hydro_last:.3e} ({hydro_reduction:.2f}x reduction)")
            print(
                f"  Deviatoric:  {avg_dev_first:.3e} → {avg_dev_last:.3e} ({dev_reduction:.2f}x reduction)")

    # Export to CSV if requested
    if args.export_csv:
        csv_path = os.path.join(run_output_dir, 'convergence_data.csv')
        print(f"\nExporting convergence data to CSV...")
        print(f"  Path: {csv_path}")

        import csv
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=['frame', 'substep', 'iter',
                               'iter_compute_time_sec', 'hydro_error', 'dev_error'])
            writer.writeheader()
            writer.writerows(solver.convergence_history)

        print(f"  Exported {len(solver.convergence_history)} records")

    print("=" * 70)

# Final summary
print("\n" + "=" * 70)
print("SIMULATION COMPLETE")
print("=" * 70)
print(f"\nOutputs:")
if usd_renderer is not None:
    print(f"  USD animation: {usd_output_path}")
if args.monitor_convergence:
    if args.plot_frames:
        conv_plot_dir = os.path.join(run_output_dir, 'convergence_plots')
        num_plots = len([f for f in os.listdir(conv_plot_dir) if f.endswith(
            '.png')]) if os.path.exists(conv_plot_dir) else 0
        print(
            f"  Substep convergence plots: {conv_plot_dir}/ ({num_plots} plots)")
    if args.export_csv:
        csv_path = os.path.join(run_output_dir, 'convergence_data.csv')
        print(f"  Convergence CSV: {csv_path}")
if args.monitor_rho and args.plot_rho and hasattr(solver, 'plot_substep_rho'):
    rho_plot_dir = os.path.join(run_output_dir, 'rho_plots')
    num_rho_plots = len([f for f in os.listdir(rho_plot_dir) if f.endswith(
        '.png')]) if os.path.exists(rho_plot_dir) else 0
    print(f"  Rho plots: {rho_plot_dir}/ ({num_rho_plots} plots)")
print(f"\nNext steps:")
if usd_renderer is not None:
    print(f"  - View USD: usdview {usd_output_path}")
if args.monitor_convergence and args.plot_frames:
    print(f"  - View substep plots: {conv_plot_dir}/f0000_s00.png (example)")
print("=" * 70)
