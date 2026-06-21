#!/usr/bin/env python3
"""
Compare convergence curves from multiple solvers.

Usage:
    python compare_convergence.py data1.json data2.json [data3.json ...]
    python compare_convergence.py --pattern "outputs/frame_*.json" --output comparison.png

Optional JSON fields (per file):
    "name": "Method A"                # legend label
    "highlight": true                 # make this curve visually prominent
    "plot_style": {                   # optional style override
        "color": "#D1495B",
        "linestyle": "--",
        "linewidth": 3.0,
        "alpha": 1.0
    }
"""

import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path
import glob


def load_convergence_data(filepath):
    """
    Load convergence data from a JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Dictionary with convergence data
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def plot_comparison(data_list, output_path, log_scale=False, show_legend=True,
                    plot_type='both', title_suffix=''):
    """
    Plot comparison of convergence curves from multiple solvers.

    Args:
        data_list: List of dicts, each containing loaded convergence data
        output_path: Path to save the comparison plot
        log_scale: Use logarithmic scale for y-axis
        show_legend: Whether to show legend
        plot_type: 'hydro', 'dev', or 'both'
        title_suffix: Additional text to append to title
    """
    if not data_list:
        print("No data to plot.")
        return

    # Determine subplot layout based on plot_type
    if plot_type == 'both':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        axes = [ax1, ax2]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        axes = [ax]

    # Color palette for different solvers
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E',
              '#BC4B51', '#5E60CE', '#7209B7', '#F72585', '#4361EE']

    # Linestyles for different solvers
    linestyles = ['--', '-.', ':']
    highlight_colors = ['#FF0054', '#FF7A00', '#00C2FF', '#00D084', '#7A5CFF']

    def apply_axis_style(ax):
        ax.set_facecolor('#FAFBFC')
        ax.grid(True, alpha=0.35, linestyle='-',
                linewidth=0.8, color='#C7D0DD')
        ax.minorticks_on()
        ax.grid(which='minor', alpha=0.2, linestyle=':',
                linewidth=0.6, color='#D8DEE8')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#A0AABA')
        ax.spines['bottom'].set_color('#A0AABA')

    has_highlight = any(bool(d['data'].get('highlight', False))
                        for d in data_list)

    def resolve_line_style(data_info, idx):
        data = data_info['data']
        style_cfg = data.get('plot_style', {})
        if not isinstance(style_cfg, dict):
            style_cfg = {}

        highlight = bool(data.get('highlight', False))
        default_color = colors[idx % len(colors)]
        if has_highlight and highlight:
            default_color = highlight_colors[idx % len(highlight_colors)]
        if has_highlight:
            # Keep all lines readable; only make highlighted one stronger.
            default_ls = '-' if highlight else linestyles[idx % len(linestyles)]
            linewidth = 4.0 if highlight else 2.2
            alpha = 1.0 if highlight else 0.95
            zorder = 5 if highlight else 2
        else:
            default_ls = '-' if idx == 0 else linestyles[(idx - 1) % len(linestyles)]
            linewidth = 2.2
            alpha = 0.95
            zorder = 3

        return {
            'highlight': highlight,
            'color': style_cfg.get('color', default_color),
            'linestyle': style_cfg.get('linestyle', default_ls),
            'linewidth': float(style_cfg.get('linewidth', linewidth)),
            'alpha': float(style_cfg.get('alpha', alpha)),
            'zorder': int(style_cfg.get('zorder', zorder)),
        }

    # Plot hydrostatic errors
    if plot_type in ['hydro', 'both']:
        ax_hydro = axes[0] if plot_type == 'both' else axes[0]

        for idx, data_info in enumerate(data_list):
            data = data_info['data']
            label = data_info['label']
            style = resolve_line_style(data_info, idx)

            iterations = data['iterations']
            hydro_errors = data['hydro_errors']

            if style['highlight']:
                # Subtle glow underlay for highlighted method.
                ax_hydro.plot(
                    iterations, hydro_errors,
                    linestyle='-',
                    color=style['color'],
                    linewidth=style['linewidth'] + 2.2,
                    alpha=0.14,
                    zorder=style['zorder'] - 1,
                    solid_capstyle='round',
                )

            ax_hydro.plot(
                iterations, hydro_errors,
                linestyle=style['linestyle'],
                color=style['color'],
                linewidth=style['linewidth'],
                alpha=style['alpha'],
                zorder=style['zorder'],
                label=label,
                solid_capstyle='round',
            )

        ax_hydro.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax_hydro.set_ylabel('Hydrostatic Error',
                            fontsize=12, fontweight='bold')
        # ax_hydro.set_title(f'Hydrostatic Convergence Comparison{title_suffix}',
        #                    fontsize=13, fontweight='bold')
        apply_axis_style(ax_hydro)

        if log_scale:
            ax_hydro.set_yscale('log')
            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((0, 0))
            ax_hydro.yaxis.set_major_formatter(formatter)
        else:
            # Linear scale with more ticks
            ax_hydro.yaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
            formatter = ticker.ScalarFormatter(
                useMathText=True, useOffset=False)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-3, 3))
            ax_hydro.yaxis.set_major_formatter(formatter)

        if show_legend:
            ax_hydro.legend(loc='best', fontsize=9, framealpha=0.9,
                            edgecolor='black', fancybox=True)

    # Plot deviatoric errors
    if plot_type in ['dev', 'both']:
        ax_dev = axes[1] if plot_type == 'both' else axes[0]

        for idx, data_info in enumerate(data_list):
            data = data_info['data']
            label = data_info['label']
            style = resolve_line_style(data_info, idx)

            iterations = data['iterations']
            dev_errors = data['dev_errors']

            if style['highlight']:
                # Subtle glow underlay for highlighted method.
                ax_dev.plot(
                    iterations, dev_errors,
                    linestyle='-',
                    color=style['color'],
                    linewidth=style['linewidth'] + 2.2,
                    alpha=0.14,
                    zorder=style['zorder'] - 1,
                    solid_capstyle='round',
                )

            ax_dev.plot(
                iterations, dev_errors,
                linestyle=style['linestyle'],
                color=style['color'],
                linewidth=style['linewidth'],
                alpha=style['alpha'],
                zorder=style['zorder'],
                label=label,
                solid_capstyle='round',
            )

        ax_dev.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax_dev.set_ylabel('Deviatoric Error',
                          fontsize=12, fontweight='bold')
        # ax_dev.set_title(f'Deviatoric Convergence Comparison{title_suffix}',
        #                  fontsize=13, fontweight='bold')
        apply_axis_style(ax_dev)

        if log_scale:
            ax_dev.set_yscale('log')
            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((0, 0))
            ax_dev.yaxis.set_major_formatter(formatter)
        else:
            # Linear scale with more ticks
            ax_dev.yaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
            formatter = ticker.ScalarFormatter(
                useMathText=True, useOffset=False)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-3, 3))
            ax_dev.yaxis.set_major_formatter(formatter)

        if show_legend:
            ax_dev.legend(loc='best', fontsize=9, framealpha=0.9,
                          edgecolor='black', fancybox=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_path}")
    plt.close()


def print_comparison_stats(data_list):
    """Print statistics comparing different solvers"""
    print("\n" + "="*80)
    print("CONVERGENCE COMPARISON STATISTICS")
    print("="*80)

    for data_info in data_list:
        data = data_info['data']
        label = data_info['label']

        hydro_errors = data['hydro_errors']
        dev_errors = data['dev_errors']

        print(f"\n{label}:")
        print(f"  Frame: {data['frame']}, Substep: {data['substep']}")
        print(f"  Solver: {data.get('solver_type', 'Unknown')}")

        if 'config' in data:
            config = data['config']
            print(f"  Configuration:")
            for key, value in config.items():
                print(f"    {key}: {value}")

        print(f"\n  Hydrostatic Error:")
        print(f"    Initial: {hydro_errors[0]:.6e}")
        print(f"    Final:   {hydro_errors[-1]:.6e}")
        reduction_h = hydro_errors[0] / \
            hydro_errors[-1] if hydro_errors[-1] > 0 else float('inf')
        print(f"    Reduction: {reduction_h:.2f}x")

        print(f"\n  Deviatoric Error:")
        print(f"    Initial: {dev_errors[0]:.6e}")
        print(f"    Final:   {dev_errors[-1]:.6e}")
        reduction_d = dev_errors[0] / \
            dev_errors[-1] if dev_errors[-1] > 0 else float('inf')
        print(f"    Reduction: {reduction_d:.2f}x")

    print("\n" + "="*80 + "\n")


def auto_generate_label(filepath, data):
    """
    Automatically generate a descriptive label from filepath and data.

    Args:
        filepath: Path to the data file
        data: Loaded convergence data

    Returns:
        String label for the plot legend
    """
    # Try to extract meaningful info from filename
    path = Path(filepath)
    filename = path.stem  # filename without extension

    # Get solver info from data
    solver_type = data.get('solver_type', 'Unknown')
    frame = data.get('frame', '?')
    substep = data.get('substep', '?')

    # Prefer explicit JSON label/name if provided
    custom_label = data.get('name', data.get('label', None))
    if isinstance(custom_label, str) and custom_label.strip():
        return custom_label.strip()
    elif 'config' in data and data['config'].get('cheb_enable'):
        label = f"{solver_type} (dynamic Cheby)"
    else:
        label = f"{solver_type}"

    # Add frame/substep info if comparing different frames/substeps
    label += f" [F{frame}S{substep}]"

    return label


def main():
    parser = argparse.ArgumentParser(
        description='Compare convergence curves from multiple solvers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two specific files
  python compare_convergence.py solver1_data.json solver2_data.json

  # Compare all JSON files in a directory
  python compare_convergence.py outputs/*.json

  # Use a glob pattern and specify output
  python compare_convergence.py --pattern "outputs/frame_0_*.json" -o comparison.png

  # Compare only hydrostatic errors with custom labels
  python compare_convergence.py data1.json data2.json --type hydro --labels "Solver A" "Solver B"
        """
    )

    parser.add_argument('files', nargs='*',
                        help='Convergence data JSON files to compare')
    parser.add_argument('--pattern', '-p', type=str,
                        help='Glob pattern for input files (alternative to listing files)')
    parser.add_argument('--output', '-o', type=str, default='convergence_comparison.png',
                        help='Output path for comparison plot (default: convergence_comparison.png)')
    parser.add_argument('--labels', '-l', nargs='+',
                        help='Custom labels for each solver (must match number of input files)')
    parser.add_argument('--type', '-t', choices=['hydro', 'dev', 'both'], default='both',
                        help='Which errors to plot: hydro, dev, or both (default: both)')
    parser.add_argument('--log', action='store_true',
                        help='Use logarithmic scale instead of linear (default: linear)')
    parser.add_argument('--no-legend', action='store_true',
                        help='Hide legend from plot')
    parser.add_argument('--no-stats', action='store_true',
                        help='Do not print comparison statistics')
    parser.add_argument('--title', type=str, default='',
                        help='Additional text to append to plot title')

    args = parser.parse_args()

    # Collect input files
    input_files = []
    if args.pattern:
        input_files.extend(glob.glob(args.pattern))
    if args.files:
        input_files.extend(args.files)

    if not input_files:
        parser.error(
            "No input files specified. Use positional arguments or --pattern.")

    # Remove duplicates while preserving order
    input_files = list(dict.fromkeys(input_files))

    print(f"Loading {len(input_files)} convergence data file(s)...")

    # Load all data
    data_list = []
    for idx, filepath in enumerate(input_files):
        try:
            data = load_convergence_data(filepath)

            # Generate or use provided label
            if args.labels and idx < len(args.labels):
                label = args.labels[idx]
            else:
                label = auto_generate_label(filepath, data)

            data_list.append({
                'filepath': filepath,
                'data': data,
                'label': label
            })
            print(f"  [OK] Loaded: {filepath}")
        except Exception as e:
            print(f"  [ERR] Error loading {filepath}: {e}")

    if not data_list:
        print("No valid data files loaded. Exiting.")
        return

    # Print statistics
    if not args.no_stats:
        print_comparison_stats(data_list)

    # Generate comparison plot
    title_suffix = f"\n{args.title}" if args.title else ""
    plot_comparison(data_list, args.output,
                    log_scale=args.log,
                    show_legend=not args.no_legend,
                    plot_type=args.type,
                    title_suffix=title_suffix)

    print(f"\nDone! Comparison plot saved to: {args.output}")


if __name__ == '__main__':
    main()
