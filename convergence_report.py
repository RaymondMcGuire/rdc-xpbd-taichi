import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def get_compute_timing_stats(solver):
    avg_solve_step = (
        solver.solve_compute_time_total / solver.solve_compute_calls
        if solver.solve_compute_calls > 0
        else 0.0
    )
    avg_iteration = (
        solver.solve_compute_time_total / solver.solve_compute_iterations
        if solver.solve_compute_iterations > 0
        else 0.0
    )
    return {
        "solve_step_compute_count": solver.solve_compute_calls,
        "constraint_iteration_count": solver.solve_compute_iterations,
        "total_compute_time_sec": solver.solve_compute_time_total,
        "avg_compute_time_per_solve_step_sec": avg_solve_step,
        "avg_compute_time_per_iteration_sec": avg_iteration,
    }


def save_substep_convergence(
    solver,
    frame,
    substep,
    convergence_data,
    save_path,
    log_scale=False,
    export_data=True,
):
    if not convergence_data:
        return

    iterations = [d["iter"] for d in convergence_data]
    iter_compute_times_sec = [d["iter_compute_time_sec"] for d in convergence_data]
    hydro_errors = [d["hydro_error"] for d in convergence_data]
    dev_errors = [d["dev_error"] for d in convergence_data]

    if export_data:
        data_path = Path(save_path).with_suffix(".json")
        export_dict = {
            "frame": frame,
            "substep": substep,
            "iterations": iterations,
            "iter_compute_times_sec": iter_compute_times_sec,
            "hydro_errors": hydro_errors,
            "dev_errors": dev_errors,
            "records": convergence_data,
            "solver_type": solver.__class__.__name__,
            "compute_timing": get_compute_timing_stats(solver),
            "config": {
                "constraints_iter": solver.constraints_iter,
                "hydrostatic_relaxation": solver.hydrostatic_relaxation,
                "deviatoric_relaxation": solver.deviatoric_relaxation,
                "cheb_enable": solver.cheb_enable,
                "cheb_gamma": solver.cheb_gamma,
                "cheb_warmup": solver.cheb_warmup,
            },
        }
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(export_dict, f, indent=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    _plot_error_axis(
        ax1,
        iterations,
        hydro_errors,
        "Hydrostatic Error |det(F) - 1|",
        f"Frame {frame}, Substep {substep} - Hydrostatic",
        "#2E86AB",
        "o",
        log_scale,
    )
    _plot_error_axis(
        ax2,
        iterations,
        dev_errors,
        "Deviatoric Error ||F||^2 - 3|",
        f"Frame {frame}, Substep {substep} - Deviatoric",
        "#A23B72",
        "s",
        log_scale,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_error_axis(
    ax,
    iterations,
    errors,
    ylabel,
    title,
    color,
    marker,
    log_scale,
):
    ax.plot(
        iterations,
        errors,
        marker + "-",
        color=color,
        linewidth=2,
        markersize=5,
        alpha=0.8,
    )
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")

    if log_scale and len(errors) > 0 and max(errors) > 0:
        ax.set_yscale("log")
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
        formatter = ticker.ScalarFormatter(useMathText=True, useOffset=False)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 3))
        ax.yaxis.set_major_formatter(formatter)

    if len(errors) > 1:
        initial = errors[0]
        final = errors[-1]
        reduction = initial / final if final > 0 else float("inf")
        ax.text(
            0.95,
            0.95,
            f"Initial: {initial:.2e}\nFinal: {final:.2e}\nReduction: {reduction:.1f}x",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
        )


def setup_convergence_callback(solver, run_output_dir, frame_ids=None, verbose=False):
    plot_dir = os.path.join(run_output_dir, "convergence_plots")
    os.makedirs(plot_dir, exist_ok=True)
    frame_id_set = set(frame_ids) if frame_ids else None

    def convergence_callback(frame, substep, convergence_data):
        if frame_id_set is None or frame in frame_id_set:
            plot_path = os.path.join(plot_dir, f"f{frame:04d}_s{substep:02d}.png")
            save_substep_convergence(solver, frame, substep, convergence_data, plot_path)
            if verbose:
                print(f"  Saved convergence plot: {plot_path}")

    solver.convergence_callback = convergence_callback
    return plot_dir


def print_frame_convergence(solver, frame, substeps):
    frame_data = [d for d in solver.convergence_history if d["frame"] == frame]
    if not frame_data:
        return

    last_substep_data = [d for d in frame_data if d["substep"] == substeps - 1]
    if not last_substep_data:
        return

    last_iter = last_substep_data[-1]
    print(
        f"  Final substep {substeps - 1}, Iter {last_iter['iter']}: "
        f"t_iter={last_iter['iter_compute_time_sec'] * 1e3:.3f} ms, "
        f"H_err={last_iter['hydro_error']:.3e}, "
        f"D_err={last_iter['dev_error']:.3e}"
    )


def get_convergence_stats(solver):
    if not solver.monitor_convergence or len(solver.convergence_history) == 0:
        return None

    grouped = defaultdict(lambda: {"hydro": [], "dev": []})
    for entry in solver.convergence_history:
        key = (entry["frame"], entry["substep"])
        grouped[key]["hydro"].append(entry["hydro_error"])
        grouped[key]["dev"].append(entry["dev_error"])

    final_hydro = [errors["hydro"][-1] for errors in grouped.values()]
    final_dev = [errors["dev"][-1] for errors in grouped.values()]

    return {
        "total_records": len(solver.convergence_history),
        "num_substeps": len(grouped),
        "hydro": {
            "mean_final": np.mean(final_hydro),
            "max_final": np.max(final_hydro),
            "min_final": np.min(final_hydro),
        },
        "dev": {
            "mean_final": np.mean(final_dev),
            "max_final": np.max(final_dev),
            "min_final": np.min(final_dev),
        },
    }


def print_convergence_analysis(solver, iterations, run_output_dir, export_csv=False):
    if not solver.monitor_convergence:
        return

    print("\n" + "=" * 70)
    print("CONSTRAINT CONVERGENCE ANALYSIS")
    print("=" * 70)

    stats = get_convergence_stats(solver)
    if stats:
        print(f"\nData Collection:")
        print(f"  Total records: {stats['total_records']:,}")
        print(f"  Substeps monitored: {stats['num_substeps']}")
        print(f"\n--- Hydrostatic Constraint: C_H = det(F) - 1 ---")
        print(f"  Mean final error: {stats['hydro']['mean_final']:.6e}")
        print(f"  Max final error:  {stats['hydro']['max_final']:.6e}")
        print(f"  Min final error:  {stats['hydro']['min_final']:.6e}")
        print(f"\n--- Deviatoric Constraint: C_D = ||F||^2 - 3 ---")
        print(f"  Mean final error: {stats['dev']['mean_final']:.6e}")
        print(f"  Max final error:  {stats['dev']['max_final']:.6e}")
        print(f"  Min final error:  {stats['dev']['min_final']:.6e}")

        first_iter_data = [d for d in solver.convergence_history if d["iter"] == 0]
        last_iter_data = [
            d for d in solver.convergence_history if d["iter"] == iterations - 1
        ]
        if first_iter_data and last_iter_data:
            avg_hydro_first = np.mean([d["hydro_error"] for d in first_iter_data])
            avg_hydro_last = np.mean([d["hydro_error"] for d in last_iter_data])
            avg_dev_first = np.mean([d["dev_error"] for d in first_iter_data])
            avg_dev_last = np.mean([d["dev_error"] for d in last_iter_data])
            hydro_reduction = (
                avg_hydro_first / avg_hydro_last
                if avg_hydro_last > 0
                else float("inf")
            )
            dev_reduction = (
                avg_dev_first / avg_dev_last if avg_dev_last > 0 else float("inf")
            )
            print(f"\n--- Convergence Rate Analysis ---")
            print(
                f"  Hydrostatic: {avg_hydro_first:.3e} -> "
                f"{avg_hydro_last:.3e} ({hydro_reduction:.2f}x reduction)"
            )
            print(
                f"  Deviatoric:  {avg_dev_first:.3e} -> "
                f"{avg_dev_last:.3e} ({dev_reduction:.2f}x reduction)"
            )

    if export_csv:
        csv_path = os.path.join(run_output_dir, "convergence_data.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "frame",
                    "substep",
                    "iter",
                    "iter_compute_time_sec",
                    "hydro_error",
                    "dev_error",
                ],
            )
            writer.writeheader()
            writer.writerows(solver.convergence_history)
        print(f"\nConvergence CSV: {csv_path}")

    print("=" * 70)
