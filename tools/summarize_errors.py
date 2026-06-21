#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
summarize_errors.py

Recursively scan JSON files and summarize:
- mean/final Deviatoric error  (from dev_errors)
- mean/final Hydrostatic error (from hydro_errors)
- mean/final Aggregate error: E_agg = a * E_dev + b * E_hyd
  (default a=b=1 -> plain sum)

- Relative improvement (%) w.r.t. a baseline:
    (E_base - E) / E_base * 100

Output formats:
- markdown / latex / csv / json
Optional:
- render a paper-like PNG table with automatic header wrapping and best-value highlighting.

Best-value highlighting rule:
- Error columns: lower is better (min)
- Improvement columns: higher is better (max)
- Baseline improvement cells shown as "—" and excluded from best selection

Expected JSON fields:
- "dev_errors": list[float]
- "hydro_errors": list[float]
Optional:
- "name": str  (method name; fallback to filename stem)
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

from xpbd_gpu.constants import EPSILON


# ----------------------------
# Basic helpers
# ----------------------------

def safe_mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def rel_improvement_pct(baseline: float, value: float) -> float:
    """Relative improvement (%) w.r.t baseline: (baseline - value)/baseline * 100."""
    if baseline is None or value is None:
        return float("nan")
    if math.isnan(baseline) or math.isnan(value):
        return float("nan")
    if baseline == 0:
        return float("nan")
    return (baseline - value) / baseline * 100.0


def fmt_float(x: float, nd: int = 6) -> str:
    return "nan" if (x is None or math.isnan(x)) else f"{x:.{nd}f}"


def fmt_pct(x: float, nd: int = 1) -> str:
    return "nan" if (x is None or math.isnan(x)) else f"{x:.{nd}f}"


def header_label(base: str, arrow: Optional[str], arrows: bool) -> str:
    return f"{base} {arrow}" if (arrows and arrow) else base


def build_headers(arrows: bool, for_markdown: bool) -> List[str]:
    br = "<br>" if for_markdown else "\n"

    return [
        "Method",
        header_label(f"Deviatoric error{br}(mean)", "↓", arrows),
        header_label(f"Hydrostatic error{br}(mean)", "↓", arrows),
        header_label(f"Aggregate error{br}(mean)", "↓", arrows),
        header_label(f"Relative improvement{br}(mean) %", "↑", arrows),

        header_label(f"Deviatoric error{br}(final)", "↓", arrows),
        header_label(f"Hydrostatic error{br}(final)", "↓", arrows),
        header_label(f"Aggregate error{br}(final)", "↓", arrows),
        header_label(f"Relative improvement{br}(final) %", "↑", arrows),
    ]


# ----------------------------
# Data model
# ----------------------------

@dataclass
class MethodMetrics:
    method: str
    file: str

    dev_mean: float
    hyd_mean: float
    agg_mean: float

    dev_final: float
    hyd_final: float
    agg_final: float

    # relative improvements vs baseline (filled later)
    dev_mean_impr_pct: float = float("nan")
    hyd_mean_impr_pct: float = float("nan")
    agg_mean_impr_pct: float = float("nan")

    dev_final_impr_pct: float = float("nan")
    hyd_final_impr_pct: float = float("nan")
    agg_final_impr_pct: float = float("nan")


# ----------------------------
# IO / parsing
# ----------------------------

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_method_name(data: Dict[str, Any], path: Path) -> str:
    name = data.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return path.stem


def extract_errors(data: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    dev = data.get("dev_errors")
    hyd = data.get("hydro_errors")
    if not isinstance(dev, list) or not isinstance(hyd, list):
        raise ValueError(
            "Missing or invalid 'dev_errors'/'hydro_errors' lists.")

    dev_f = [float(x) for x in dev if isinstance(x, (int, float))]
    hyd_f = [float(x) for x in hyd if isinstance(x, (int, float))]
    if not dev_f or not hyd_f:
        raise ValueError(
            "'dev_errors'/'hydro_errors' are empty after filtering numerics.")

    n = min(len(dev_f), len(hyd_f))
    return dev_f[:n], hyd_f[:n]


def compute_metrics(path: Path, a_dev: float, b_hyd: float) -> MethodMetrics:
    data = load_json(path)
    method = parse_method_name(data, path)
    dev, hyd = extract_errors(data)

    agg = [a_dev * d + b_hyd * h for d, h in zip(dev, hyd)]

    return MethodMetrics(
        method=method,
        file=str(path),

        dev_mean=safe_mean(dev),
        hyd_mean=safe_mean(hyd),
        agg_mean=safe_mean(agg),

        dev_final=dev[-1],
        hyd_final=hyd[-1],
        agg_final=agg[-1],
    )


def find_json_files(inputs: List[str]) -> List[Path]:
    files: List[Path] = []
    for p in inputs:
        path = Path(p).expanduser().resolve()
        if path.is_file() and path.suffix.lower() == ".json":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.json")))

    # de-dup preserve order
    seen = set()
    uniq: List[Path] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq


# ----------------------------
# Baseline / improvements / sorting
# ----------------------------

def select_baseline(
    metrics: List[MethodMetrics],
    baseline_file: Optional[str],
    baseline_method: Optional[str],
) -> MethodMetrics:
    if baseline_file:
        bf = str(Path(baseline_file).expanduser().resolve())
        for m in metrics:
            if str(Path(m.file).resolve()) == bf:
                return m
        raise SystemExit(
            f"[ERROR] baseline_file not found among loaded JSONs: {bf}")

    if baseline_method:
        key = baseline_method.strip().lower()
        for m in metrics:
            if m.method.strip().lower() == key:
                return m
        raise SystemExit(
            f"[ERROR] baseline_method not found among methods: {baseline_method}")

    raise SystemExit(
        "[ERROR] You must set either --baseline-file or --baseline-method.")


def attach_improvements(metrics: List[MethodMetrics], baseline: MethodMetrics) -> None:
    for m in metrics:
        m.dev_mean_impr_pct = rel_improvement_pct(
            baseline.dev_mean, m.dev_mean)
        m.hyd_mean_impr_pct = rel_improvement_pct(
            baseline.hyd_mean, m.hyd_mean)
        m.agg_mean_impr_pct = rel_improvement_pct(
            baseline.agg_mean, m.agg_mean)

        m.dev_final_impr_pct = rel_improvement_pct(
            baseline.dev_final, m.dev_final)
        m.hyd_final_impr_pct = rel_improvement_pct(
            baseline.hyd_final, m.hyd_final)
        m.agg_final_impr_pct = rel_improvement_pct(
            baseline.agg_final, m.agg_final)


def sort_methods(metrics: List[MethodMetrics], sort_by: str) -> List[MethodMetrics]:
    key_map = {
        "agg_mean": lambda m: m.agg_mean,
        "agg_final": lambda m: m.agg_final,
        "dev_mean": lambda m: m.dev_mean,
        "hyd_mean": lambda m: m.hyd_mean,
        "method": lambda m: m.method.lower(),
    }
    if sort_by not in key_map:
        raise SystemExit(
            f"[ERROR] Unknown --sort-by: {sort_by}. Choices: {', '.join(key_map.keys())}")
    return sorted(metrics, key=key_map[sort_by])


# ----------------------------
# Table building + best highlighting
# ----------------------------

def build_table_rows(
    metrics: List[MethodMetrics],
    baseline: MethodMetrics,
    nd_err: int,
    nd_pct: int,
    show_baseline_impr_as_dash: bool = True,
) -> Tuple[List[List[str]], List[List[float]]]:
    """
    Returns:
      rows_str: display rows
      rows_raw: numeric rows aligned with display columns (Method column = nan)
    """
    base_file = str(Path(baseline.file).resolve())
    base_method = baseline.method

    def is_baseline(m: MethodMetrics) -> bool:
        return m.method == base_method and str(Path(m.file).resolve()) == base_file

    rows_str: List[List[str]] = []
    rows_raw: List[List[float]] = []

    for m in metrics:
        base_row = is_baseline(m)

        def impr_cell(v: float) -> str:
            if show_baseline_impr_as_dash and base_row:
                return "—"
            return f"{fmt_pct(v, nd_pct)}%"

        rows_str.append([
            m.method,
            fmt_float(m.dev_mean, nd_err),
            fmt_float(m.hyd_mean, nd_err),
            fmt_float(m.agg_mean, nd_err),
            impr_cell(m.agg_mean_impr_pct),

            fmt_float(m.dev_final, nd_err),
            fmt_float(m.hyd_final, nd_err),
            fmt_float(m.agg_final, nd_err),
            impr_cell(m.agg_final_impr_pct),
        ])

        rows_raw.append([
            float("nan"),
            m.dev_mean,
            m.hyd_mean,
            m.agg_mean,
            (float("nan") if (show_baseline_impr_as_dash and base_row)
             else m.agg_mean_impr_pct),

            m.dev_final,
            m.hyd_final,
            m.agg_final,
            (float("nan") if (show_baseline_impr_as_dash and base_row)
             else m.agg_final_impr_pct),
        ])

    return rows_str, rows_raw


def best_cells(rows_raw: List[List[float]]) -> Set[Tuple[int, int]]:
    """
    Return set of (row_idx, col_idx) among DATA rows / DISPLAY columns.
    row_idx: 0..len(rows)-1
    col_idx: 0..len(headers)-1
    """
    if not rows_raw:
        return set()

    n_rows = len(rows_raw)
    n_cols = len(rows_raw[0])

    # col indices: 0 Method
    # 1 dev_mean,2 hyd_mean,3 agg_mean,4 impr_mean,
    # 5 dev_final,6 hyd_final,7 agg_final,8 impr_final
    lower_better = {1, 2, 3, 5, 6, 7}
    higher_better = {4, 8}

    best: Set[Tuple[int, int]] = set()

    for c in range(1, n_cols):
        finite = [(r, rows_raw[r][c]) for r in range(n_rows)
                  if rows_raw[r][c] is not None and not math.isnan(rows_raw[r][c])]
        if not finite:
            continue

        if c in lower_better:
            target = min(v for _, v in finite)
        elif c in higher_better:
            target = max(v for _, v in finite)
        else:
            continue

        for r, v in finite:
            if abs(v - target) <= EPSILON:
                best.add((r, c))

    return best


def apply_best_markdown(rows: List[List[str]], best: Set[Tuple[int, int]]) -> List[List[str]]:
    out = []
    for r, row in enumerate(rows):
        new_row = row[:]
        for c in range(len(row)):
            if (r, c) in best:
                new_row[c] = f"**{new_row[c]}**"
        out.append(new_row)
    return out


def apply_best_latex(rows: List[List[str]], best: Set[Tuple[int, int]]) -> List[List[str]]:
    out = []
    for r, row in enumerate(rows):
        new_row = row[:]
        for c in range(len(row)):
            if (r, c) in best:
                new_row[c] = f"\\textbf{{{new_row[c]}}}"
        out.append(new_row)
    return out


# ----------------------------
# Output renderers
# ----------------------------

def to_markdown(headers: List[str], rows: List[List[str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def to_latex(headers: List[str], rows: List[List[str]]) -> str:
    # booktabs-friendly; user should include \usepackage{booktabs}
    cols = "l" + "r" * (len(headers) - 1)
    lines = []
    lines.append("\\begin{tabular}{" + cols + "}")
    lines.append("\\toprule")
    lines.append(" & ".join(headers) + " \\\\")
    lines.append("\\midrule")
    for r in rows:
        lines.append(" & ".join(r) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def to_csv_text(metrics: List[MethodMetrics]) -> str:
    if not metrics:
        return ""
    fieldnames = list(asdict(metrics[0]).keys())
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fieldnames)
    w.writeheader()
    for m in metrics:
        w.writerow(asdict(m))
    return buf.getvalue()


def render_table_png(
    headers: List[str],
    rows: List[List[str]],
    best: Set[Tuple[int, int]],
    out_path: str,
    title: Optional[str] = None,
    font_size: int = 10,
    dpi: int = 250,
) -> None:
    """
    Render a paper-like PNG table:
    - wrapped headers (use \n in headers)
    - column width heuristic based on text length
    - subtle header background
    - highlight best cells with light gray background + bold
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit(
            "[ERROR] matplotlib is required for --plot. Install it via:\n"
            "  pip install matplotlib\n"
            f"Original error: {e}"
        )

    n_rows = len(rows) + 1  # includes header row
    n_cols = len(headers)

    # Wider figure to avoid squishing; height scales with rows
    fig_w = max(12.0, 1.55 * n_cols)
    fig_h = max(3.5, 0.55 * n_rows)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font_size)
    tbl.scale(1.0, 1.35)  # increase row height

    # Estimate column widths from header/body lengths
    def text_len(s: str) -> int:
        s = "" if s is None else str(s)
        return max((len(part) for part in s.split("\n")), default=len(s))

    col_lens = []
    for c in range(n_cols):
        header_len = text_len(headers[c])
        body_len = max(text_len(rows[r][c])
                       for r in range(len(rows))) if rows else 0
        col_lens.append(max(header_len, body_len, 6))

    total = sum(col_lens)
    rel_widths = []
    for c, L in enumerate(col_lens):
        w = L / total
        if c == 0:  # Method column slightly wider
            w *= 1.25
        rel_widths.append(w)

    s = sum(rel_widths)
    rel_widths = [w / s for w in rel_widths]

    # Apply widths to each cell
    for c in range(n_cols):
        for r in range(n_rows):
            tbl[(r, c)].set_width(rel_widths[c])

    # Header styling
    for c in range(n_cols):
        cell = tbl[(0, c)]
        cell.set_text_props(weight="bold")
        cell.set_facecolor((0.95, 0.95, 0.95))

    # Highlight best cells (data rows start at 1)
    for (r0, c) in best:
        r = r0 + 1
        cell = tbl[(r, c)]
        cell.set_facecolor((0.90, 0.90, 0.90))
        cell.set_text_props(weight="bold")

    if title:
        ax.set_title(title, pad=14)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+",
                    help="JSON file(s) and/or folder(s) (folders scanned recursively).")

    ap.add_argument("--baseline-file", default=None,
                    help="Path to baseline JSON file.")
    ap.add_argument("--baseline-method", default=None,
                    help="Baseline method name (matches JSON['name']).")

    # coefficients for aggregate error (default: plain sum)
    ap.add_argument("--a-dev", type=float, default=1.0,
                    help="Coefficient for Deviatoric error in Aggregate error.")
    ap.add_argument("--b-hyd", type=float, default=1.0,
                    help="Coefficient for Hydrostatic error in Aggregate error.")

    ap.add_argument("--sort-by", default="agg_mean",
                    help="Sort key: agg_mean (default), agg_final, dev_mean, hyd_mean, method")

    ap.add_argument("--format", default="markdown",
                    choices=["markdown", "latex", "csv", "json"])
    ap.add_argument("--output", default=None,
                    help="Write printed output to file (optional).")

    ap.add_argument("--nd-err", type=int, default=6,
                    help="Decimals for error values.")
    ap.add_argument("--nd-pct", type=int, default=1,
                    help="Decimals for improvement percentage.")

    ap.add_argument("--arrows", action="store_true",
                    help="Add ↑/↓ arrows to column headers.")
    ap.add_argument("--plot", default=None,
                    help="If set, render a PNG table to this path (requires matplotlib).")
    ap.add_argument("--plot-title", default=None,
                    help="Optional title for the PNG table.")
    ap.add_argument("--font-size", type=int, default=10,
                    help="Font size for PNG table.")
    ap.add_argument("--dpi", type=int, default=250, help="DPI for PNG table.")

    ap.add_argument("--no-best-bold", action="store_true",
                    help="Disable bolding best values in markdown/latex output (PNG still highlights).")

    args = ap.parse_args()

    if args.a_dev < 0 or args.b_hyd < 0:
        raise SystemExit("[ERROR] coefficients must be non-negative.")
    if args.a_dev == 0 and args.b_hyd == 0:
        raise SystemExit("[ERROR] at least one coefficient must be > 0.")

    json_files = find_json_files(args.inputs)
    if not json_files:
        raise SystemExit("[ERROR] No JSON files found from given inputs.")

    metrics: List[MethodMetrics] = []
    for f in json_files:
        try:
            metrics.append(compute_metrics(
                f, a_dev=args.a_dev, b_hyd=args.b_hyd))
        except Exception as e:
            print(f"[WARN] skip {f}: {e}")

    if not metrics:
        raise SystemExit(
            "[ERROR] No valid JSON files with required fields were loaded.")

    baseline = select_baseline(
        metrics, args.baseline_file, args.baseline_method)
    attach_improvements(metrics, baseline)
    metrics = sort_methods(metrics, args.sort_by)

    headers_md = build_headers(arrows=args.arrows, for_markdown=True)
    headers_png = build_headers(arrows=args.arrows, for_markdown=False)
    headers = headers_md if args.format == "markdown" else headers_png

    rows_str, rows_raw = build_table_rows(
        metrics=metrics,
        baseline=baseline,
        nd_err=args.nd_err,
        nd_pct=args.nd_pct,
        show_baseline_impr_as_dash=True,
    )

    best = best_cells(rows_raw)

    # Print output (optionally bold best cells)
    if args.format == "markdown":
        out_rows = rows_str if args.no_best_bold else apply_best_markdown(
            rows_str, best)
        text = to_markdown(headers, out_rows)
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
        else:
            print(text)

    elif args.format == "latex":
        # latex headers should not contain arrows as unicode sometimes; but you asked optional arrows
        # It's typically okay; if you prefer, disable --arrows for latex.
        out_rows = rows_str if args.no_best_bold else apply_best_latex(
            rows_str, best)
        text = to_latex(headers, out_rows)
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
        else:
            print(text)

    elif args.format == "json":
        payload = [asdict(m) for m in metrics]
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
        else:
            print(text)

    elif args.format == "csv":
        text = to_csv_text(metrics)
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
        else:
            print(text)

    # Plot PNG
    if args.plot:
        render_table_png(
            headers=headers_png,
            rows=rows_str,
            best=best,
            out_path=args.plot,
            title=args.plot_title,
            font_size=args.font_size,
            dpi=args.dpi,
        )
        print(f"[OK] Table figure saved to: {args.plot}")


if __name__ == "__main__":
    main()
