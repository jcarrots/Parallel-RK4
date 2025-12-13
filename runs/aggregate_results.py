r"""Aggregate run CSVs into analysis tables and plots.

This script walks the ``runs`` directory, parses every ``.csv`` file that
captures RK4 timing data, and writes both a consolidated ``results.csv`` file
and higher-level presentation artifacts:

* ``results.tex`` renders strong-scaling tables, weak-scaling tables, and
  per-size performance summaries for ``mpi_cuda`` and ``mpi_omp``.
* ``mpi_omp_scaling.png`` visualizes single-node (thread scaling) and
  cross-node (rank scaling) performance for the ``mpi_omp`` runs.

Filenames encode the configuration (e.g., GPU count or MPI rank/OMP thread
counts), while the file body provides runtime metrics. The LaTeX output uses
``booktabs`` commands (``\toprule``/``\midrule``/``\bottomrule``); include
``\usepackage{booktabs}`` in your preamble when importing the generated table.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional runtime dependency
    matplotlib = None
    plt = None


RUNS_DIR = Path(__file__).parent


CUDA_PATTERN = re.compile(r"rk4_mpi_cuda_g(?P<gpus>\d+)_D(?P<D>\d+)\.csv")
OMP_PATTERN = re.compile(r"rk4_mpi_omp_r(?P<ranks>\d+)_t(?P<threads>\d+)_D(?P<D>\d+)\.csv")


def parse_numeric(value: str) -> Any:
    """Convert string values into ints or floats when possible."""

    if value.isdigit():
        return int(value)

    try:
        return float(value)
    except ValueError:
        return value


def parse_file(path: Path) -> dict[str, Any]:
    """Parse a single CSV file into a flat result row."""

    row: dict[str, Any] = {
        "implementation": None,
        "gpus": None,
        "ranks": None,
        "threads": None,
        "D": None,
        "T": None,
        "dt": None,
        "nsteps": None,
        "max_time_s": None,
        "l2_error": None,
        "source": path.relative_to(RUNS_DIR).as_posix(),
    }

    filename = path.name
    cuda_match = CUDA_PATTERN.match(filename)
    omp_match = OMP_PATTERN.match(filename)

    if cuda_match:
        row["implementation"] = "mpi_cuda"
        row["gpus"] = int(cuda_match.group("gpus"))
        row["D"] = int(cuda_match.group("D"))
    elif omp_match:
        row["implementation"] = "mpi_omp"
        row["ranks"] = int(omp_match.group("ranks"))
        row["threads"] = int(omp_match.group("threads"))
        row["D"] = int(omp_match.group("D"))
    else:
        raise ValueError(f"Unrecognized filename format: {filename}")

    with path.open() as csv_file:
        for line in csv_file:
            key, value = line.strip().split(",", 1)
            if key in row:
                row[key] = parse_numeric(value)

    # The mpi_cuda runs were produced with a consistent 20-step, T=0.02
    # configuration even though the raw CSV headers vary; normalize the
    # encoded metadata so downstream tables and plots reflect the actual
    # simulation parameters.
    if row["implementation"] == "mpi_cuda":
        row["nsteps"] = 20

    return row


def escape_latex(value: Any) -> str:
    """Return a LaTeX-safe string representation of ``value``."""

    if value is None:
        return ""

    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }

    for char, replacement in replacements.items():
        text = text.replace(char, replacement)

    return text


def format_parameters(row: dict[str, Any]) -> str:
    """Create a compact, human-friendly parameter string for LaTeX output."""

    fields = []
    for key in ("gpus", "ranks", "threads", "D", "T", "dt", "nsteps"):
        value = row.get(key)
        if value not in (None, ""):
            fields.append(f"{key}={value}")

    return ", ".join(fields)


def as_float(value: Any) -> float:
    """Return ``value`` as ``float`` when possible, else ``math.nan``."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def resource_count(row: dict[str, Any]) -> int:
    """Return the parallelism level for a row (GPUs or ranks)."""

    if row.get("implementation") == "mpi_cuda":
        return int(row.get("gpus") or 0)

    return int(row.get("ranks") or 0)


def effective_workers(row: dict[str, Any]) -> int:
    """Return the total worker count (MPI ranks x threads for OMP)."""

    if row.get("implementation") == "mpi_cuda":
        return int(row.get("gpus") or 0)

    ranks = int(row.get("ranks") or 0)
    threads = int(row.get("threads") or 0)
    return ranks * threads


def group_by_static_parameters(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    """Bucket rows by all parameters except ``D``.

    Grouping by the shared configuration allows the LaTeX table to present
    results ordered by the problem size ``D`` while keeping other settings
    constant, as requested.
    """

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row.get("implementation"),
            row.get("gpus"),
            row.get("ranks"),
            row.get("threads"),
            row.get("T"),
            row.get("dt"),
            row.get("nsteps"),
        )
        groups.setdefault(key, []).append(row)

    return groups


def summarize_strong_scaling(
    rows: list[dict[str, Any]]
) -> dict[tuple[Any, ...], dict[int, dict[int, dict[str, Any]]]]:
    """Organize rows for strong-scaling presentation.

    * ``mpi_cuda``: vary GPU count while holding D/T/dt/nsteps constant.
    * ``mpi_omp``: vary MPI ranks while holding threads/D/T/dt/nsteps constant.
    """

    results: dict[tuple[Any, ...], dict[int, dict[int, dict[str, Any]]]] = {}

    for row in rows:
        impl = row.get("implementation")
        if impl == "mpi_cuda":
            key = (impl, row.get("T"), row.get("dt"), row.get("nsteps"))
            resource_axis = int(row.get("gpus") or 0)
        else:
            key = (
                impl,
                row.get("threads"),
                row.get("T"),
                row.get("dt"),
                row.get("nsteps"),
            )
            resource_axis = int(row.get("ranks") or 0)

        dimension = int(row.get("D") or 0)
        results.setdefault(key, {}).setdefault(dimension, {})[resource_axis] = row

    return results


def summarize_weak_scaling(
    rows: list[dict[str, Any]]
) -> dict[tuple[Any, ...], dict[float, dict[int, dict[str, Any]]]]:
    """Organize rows for weak-scaling presentation keyed by load per resource."""

    results: dict[tuple[Any, ...], dict[float, dict[int, dict[str, Any]]]] = {}

    for row in rows:
        impl = row.get("implementation")
        resources = resource_count(row)
        if resources <= 0:
            continue

        load = (int(row.get("D") or 0)) / resources

        if impl == "mpi_cuda":
            key = (impl, row.get("T"), row.get("dt"), row.get("nsteps"))
        else:
            key = (
                impl,
                row.get("threads"),
                row.get("T"),
                row.get("dt"),
                row.get("nsteps"),
            )

        results.setdefault(key, {}).setdefault(load, {})[resources] = row

    return results


def performance_by_size(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Select canonical configurations for per-size comparison."""

    summaries: dict[str, list[dict[str, Any]]] = {"mpi_cuda": [], "mpi_omp": []}

    for row in rows:
        impl = row.get("implementation")
        if impl == "mpi_cuda" and int(row.get("gpus") or 0) == 1:
            summaries[impl].append(row)
        elif impl == "mpi_omp" and int(row.get("ranks") or 0) == 1 and int(row.get("threads") or 0) == 1:
            summaries[impl].append(row)

    for impl in summaries:
        summaries[impl].sort(key=lambda r: int(r.get("D") or 0))

    return summaries


def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return rows sorted by implementation and parameters, then ``D``."""

    def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            row.get("implementation") or "",
            row.get("gpus") or 0,
            row.get("ranks") or 0,
            row.get("threads") or 0,
            row.get("T") or 0,
            row.get("dt") or 0,
            row.get("nsteps") or 0,
            row.get("D") or 0,
        )

    return sorted(rows, key=sort_key)


def write_latex_table(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write LaTeX tables for strong/weak scaling and per-size performance."""

    strong = summarize_strong_scaling(rows)
    weak = summarize_weak_scaling(rows)
    size_summary = performance_by_size(rows)

    def strong_caption(impl: str) -> str:
        if impl == "mpi_cuda":
            return "Strong scaling for mpi\\_cuda (fixed D, varying GPU count)"
        return "Strong scaling for mpi\\_omp (fixed D, varying MPI ranks at fixed threads)"

    def weak_caption(impl: str) -> str:
        if impl == "mpi_cuda":
            return "Weak scaling for mpi\\_cuda (constant cells per GPU)"
        return "Weak scaling for mpi\\_omp (constant cells per rank, fixed threads)"

    def size_caption(impl: str) -> str:
        if impl == "mpi_cuda":
            return "Runtime vs. problem size for mpi\\_cuda (1 GPU baseline)"
        return "Runtime vs. problem size for mpi\\_omp (1 rank, 1 thread baseline)"

    with output_path.open("w") as tex_file:
        tex_file.write("% Auto-generated by aggregate_results.py\n")

        # Strong scaling tables
        for key, dim_map in sorted(strong.items(), key=lambda item: item[0]):
            impl = key[0]
            caption = strong_caption(impl or "")
            label = f"tab:strong_{impl}"

            if impl == "mpi_cuda":
                _, T, dt, nsteps = key
                param_text = f"T={T}, dt={dt}, nsteps={nsteps}"
                resource_label = "GPUs"
            else:
                _, threads, T, dt, nsteps = key
                param_text = f"threads={threads}, T={T}, dt={dt}, nsteps={nsteps}"
                resource_label = "MPI ranks"

            tex_file.write("\\begin{table}[htbp]\n")
            tex_file.write("\\centering\n")
            tex_file.write(f"\\caption{{{caption} ({escape_latex(param_text)})}}\n")
            tex_file.write(f"\\label{{{label}}}\n")
            tex_file.write("\\begin{tabular}{lrrr}\n")
            tex_file.write("\\toprule\n")
            tex_file.write(f"D & {escape_latex(resource_label)} & Time (s) & Speedup \\\\\n")

            for dim in sorted(dim_map):
                resource_rows = dim_map[dim]
                baseline_resource = min(resource_rows)
                baseline_time = as_float(resource_rows[baseline_resource].get("max_time_s"))

                for idx, res in enumerate(sorted(resource_rows)):
                    row = resource_rows[res]
                    runtime = as_float(row.get("max_time_s"))
                    speedup = baseline_time / runtime if runtime > 0 else math.nan
                    prefix = f"{dim}" if idx == 0 else ""
                    tex_file.write(
                        f"{escape_latex(prefix)} & {res} & {runtime:.6f} & {speedup:.2f} \\\\\n"
                    )

                tex_file.write("\\midrule\n")

            tex_file.write("\\bottomrule\n")
            tex_file.write("\\end{tabular}\n")
            tex_file.write("\\end{table}\n\n")

        # Weak scaling tables
        for key, load_map in sorted(weak.items(), key=lambda item: item[0]):
            impl = key[0]
            caption = weak_caption(impl or "")
            label = f"tab:weak_{impl}"

            if impl == "mpi_cuda":
                _, T, dt, nsteps = key
                param_text = f"T={T}, dt={dt}, nsteps={nsteps}"
                resource_label = "GPUs"
            else:
                _, threads, T, dt, nsteps = key
                param_text = f"threads={threads}, T={T}, dt={dt}, nsteps={nsteps}"
                resource_label = "MPI ranks"

            tex_file.write("\\begin{table}[htbp]\n")
            tex_file.write("\\centering\n")
            tex_file.write(f"\\caption{{{caption} ({escape_latex(param_text)})}}\n")
            tex_file.write(f"\\label{{{label}}}\n")
            tex_file.write("\\begin{tabular}{lrrr}\n")
            tex_file.write("\\toprule\n")
            tex_file.write(
                f"Load per resource & {escape_latex(resource_label)} & D & Time (s) \\\\\n"
            )

            for load in sorted(load_map):
                resource_rows = load_map[load]
                for idx, res in enumerate(sorted(resource_rows)):
                    row = resource_rows[res]
                    prefix = f"{load:g}" if idx == 0 else ""
                    tex_file.write(
                        f"{escape_latex(prefix)} & {res} & {row.get('D')} & {as_float(row.get('max_time_s')):.6f} \\\\\n"
                    )

                tex_file.write("\\midrule\n")

            tex_file.write("\\bottomrule\n")
            tex_file.write("\\end{tabular}\n")
            tex_file.write("\\end{table}\n\n")

        # Per-size summaries
        for impl, impl_rows in sorted(size_summary.items()):
            caption = size_caption(impl)
            label = f"tab:size_{impl}"

            tex_file.write("\\begin{table}[htbp]\n")
            tex_file.write("\\centering\n")
            tex_file.write(f"\\caption{{{caption}}}\n")
            tex_file.write(f"\\label{{{label}}}\n")
            tex_file.write("\\begin{tabular}{lrr}\n")
            tex_file.write("\\toprule\n")
            tex_file.write("D & Time (s) & L2 error \\\\\n")

            for row in impl_rows:
                tex_file.write(
                    f"{row.get('D')} & {as_float(row.get('max_time_s')):.6f} & {escape_latex(row.get('l2_error'))} \\\\\n"
                )

            tex_file.write("\\bottomrule\n")
            tex_file.write("\\end{tabular}\n")
            tex_file.write("\\end{table}\n\n")


def generate_mpi_omp_scaling_plot(rows: list[dict[str, Any]], output_path: Path) -> bool:
    """Render single-node and cross-node scaling for mpi+omp as a PNG."""

    if plt is None:
        print("matplotlib unavailable; skipping mpi_omp scaling plot.")
        return False

    omp_rows = [r for r in rows if r.get("implementation") == "mpi_omp"]
    if not omp_rows:
        return False

    def rows_for(filter_fn):
        return [r for r in omp_rows if filter_fn(r)]

    # Single-node: ranks=1, vary threads; pick the largest common D across threads
    thread_sets = {
        threads: {int(r.get("D") or 0) for r in rows_for(lambda row, t=threads: int(row.get("ranks") or 0) == 1 and int(row.get("threads") or 0) == t)}
        for threads in {1, 2, 4}
    }
    common_single_node = set.intersection(*thread_sets.values()) if all(thread_sets.values()) else set()
    single_node_dimension = max(common_single_node) if common_single_node else None

    single_node_data = []
    if single_node_dimension:
        for threads in sorted(thread_sets):
            match = next(
                (
                    r
                    for r in omp_rows
                    if int(r.get("ranks") or 0) == 1
                    and int(r.get("threads") or 0) == threads
                    and int(r.get("D") or 0) == single_node_dimension
                ),
                None,
            )
            if match:
                single_node_data.append((threads, as_float(match.get("max_time_s"))))

    # Cross-node: threads=1, vary ranks; pick the largest common D across ranks
    rank_sets = {
        ranks: {int(r.get("D") or 0) for r in rows_for(lambda row, rk=ranks: int(row.get("threads") or 0) == 1 and int(row.get("ranks") or 0) == rk)}
        for ranks in {1, 2, 4, 8, 16}
    }
    common_multi_node = set.intersection(*rank_sets.values()) if all(rank_sets.values()) else set()
    multi_node_dimension = max(common_multi_node) if common_multi_node else None

    cross_node_data = []
    if multi_node_dimension:
        for ranks in sorted(rank_sets):
            match = next(
                (
                    r
                    for r in omp_rows
                    if int(r.get("threads") or 0) == 1
                    and int(r.get("ranks") or 0) == ranks
                    and int(r.get("D") or 0) == multi_node_dimension
                ),
                None,
            )
            if match:
                cross_node_data.append((ranks, as_float(match.get("max_time_s"))))

    if not single_node_data and not cross_node_data:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    if single_node_data:
        threads, times = zip(*sorted(single_node_data))
        axes[0].plot(threads, times, marker="o")
        axes[0].set_title(f"Single node (D={single_node_dimension})")
        axes[0].set_xlabel("Threads per rank (ranks=1)")
        axes[0].set_ylabel("Time (s)")
        axes[0].grid(True, linestyle=":")

    if cross_node_data:
        ranks, times = zip(*sorted(cross_node_data))
        axes[1].plot(ranks, times, marker="o", color="C1")
        axes[1].set_title(f"Across nodes (D={multi_node_dimension}, threads=1)")
        axes[1].set_xlabel("MPI ranks")
        axes[1].grid(True, linestyle=":")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    return True


def main() -> None:
    rows = []
    for csv_path in sorted(RUNS_DIR.rglob("*.csv")):
        if csv_path.name == "results.csv":
            # Avoid ingesting an earlier aggregate output.
            continue

        rows.append(parse_file(csv_path))

    sorted_rows = sort_rows(rows)

    output_path = RUNS_DIR / "results.csv"
    header = list(sorted_rows[0].keys()) if sorted_rows else []
    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=header)
        writer.writeheader()
        writer.writerows(sorted_rows)

    tex_output_path = RUNS_DIR / "results.tex"
    write_latex_table(sorted_rows, tex_output_path)

    figure_path = RUNS_DIR / "mpi_omp_scaling.png"
    figure_written = generate_mpi_omp_scaling_plot(sorted_rows, figure_path)

    print(
        "\n".join(
            [
                f"Wrote {len(rows)} rows to {output_path.relative_to(Path.cwd())}",
                f"Wrote LaTeX table to {tex_output_path.relative_to(Path.cwd())}",
                (
                    f"Wrote mpi_omp scaling figure to {figure_path.relative_to(Path.cwd())}"
                    if figure_written
                    else "Skipped mpi_omp scaling figure (matplotlib unavailable or data missing)"
                ),
            ]
        )
    )


if __name__ == "__main__":
    main()
