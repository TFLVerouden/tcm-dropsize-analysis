"""new_analysis.py

Quick/dirty loaders to get *compatible* size distributions from:

- Morgan PDA (.h5) in Morgan_data/PDA/<case>
- Abe Spraytec averages (.txt) in spraytec/Averages/Unweighted/<case>

Compatibility here means: both are returned on the same diameter bin edges (in µm).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, cast
import json

import numpy as np
import pandas as pd
import h5py

import matplotlib.pyplot as plt


def add_broken_xaxis_marks(
    fig: Any,
    ax_left: Any,
    ax_right: Any,
    *,
    size: float = 0.008,
    linewidth: float | None = None,
    color: str = "k",
) -> None:
    """Draw diagonal break marks between two horizontally adjacent axes.

    Important: Call this *after* final layout (tight_layout/subplots_adjust),
    otherwise the axes positions move and the marks end up offset.
    """

    from matplotlib.lines import Line2D

    if linewidth is None:
        linewidth = float(plt.rcParams.get("axes.linewidth", 1.0))

    b0 = ax_left.get_position()
    b1 = ax_right.get_position()

    x_left_edge = float(b0.x1)
    x_right_edge = float(b1.x0)
    y_lo = float(b0.y0)
    y_hi = float(b0.y1)

    def _diag(x: float, y: float) -> tuple[list[float], list[float]]:
        return [x - size, x + size], [y - size, y + size]

    for x in (x_left_edge, x_right_edge):
        for y in (y_lo, y_hi):
            xs, ys = _diag(x, y)
            fig.add_artist(
                Line2D(
                    xs,
                    ys,
                    transform=fig.transFigure,
                    color=color,
                    linewidth=linewidth,
                    clip_on=False,
                )
            )


@dataclass
class BinnedDistribution:
    bin_edges_um: np.ndarray  # shape (n_bins+1,)
    bin_centers_um: np.ndarray  # shape (n_bins,)
    bin_widths_um: np.ndarray  # shape (n_bins,)
    n_percent: np.ndarray  # percent per bin, sums to ~100
    v_percent: np.ndarray  # percent per bin, sums to ~100
    n_pdf: np.ndarray  # PDF (1/um), integrates to ~1
    v_pdf: np.ndarray  # PDF (1/um), integrates to ~1
    meta: dict[str, Any]


_VOLUME_COL_RE = re.compile(r"^%\s*V\s*\((?P<lo>[\d.]+)\-(?P<hi>[\d.]+)")


def _extract_spraytec_bins_from_columns(columns: list[str]) -> tuple[np.ndarray, list[str]]:
    """Return (bin_edges_um, ordered_volume_columns).

    Spraytec average files have columns like:
      '% V (0.100-0.117µm)'  (µ character may be mangled in latin1)
    """
    pairs: list[tuple[float, float, str]] = []
    for col in columns:
        m = _VOLUME_COL_RE.search(col)
        if not m:
            continue
        lo = float(m.group("lo"))
        hi = float(m.group("hi"))
        pairs.append((lo, hi, col))

    if not pairs:
        raise ValueError(
            "Could not find any Spraytec '% V (lo-hi...)' columns")

    pairs.sort(key=lambda t: t[0])
    ordered_cols = [p[2] for p in pairs]

    # Build edges: [lo0, hi0, hi1, hi2, ...]
    edges = [pairs[0][0]]
    for lo, hi, _ in pairs:
        # Loose continuity check; don't be strict because formatting/rounding happens.
        if edges and lo < edges[-1] * 0.999:
            # Non-monotonic; still append hi to keep going.
            pass
        edges.append(hi)

    bin_edges_um = np.asarray(edges, dtype=float)
    if not np.all(np.isfinite(bin_edges_um)):
        raise ValueError("Non-finite bin edges parsed from Spraytec columns")
    if not np.all(np.diff(bin_edges_um) > 0):
        raise ValueError("Parsed bin edges are not strictly increasing")

    return bin_edges_um, ordered_cols


def _bin_centers_and_widths(bin_edges_um: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    widths = np.diff(bin_edges_um)
    centers = (bin_edges_um[:-1] + bin_edges_um[1:]) / 2.0
    return centers, widths


def _percent_to_pdf(percent: np.ndarray, bin_widths_um: np.ndarray) -> np.ndarray:
    """Convert percent-per-bin to a PDF in 1/um."""
    p = np.asarray(percent, dtype=float) / 100.0
    pdf = p / bin_widths_um
    # Normalize defensively
    area = float(np.sum(pdf * bin_widths_um))
    if area > 0:
        pdf = pdf / area
    return pdf


def load_abe_average_txt(file_path: Path, *, expected_bin_edges_um: np.ndarray | None = None) -> BinnedDistribution:
    """Parse one Spraytec 'average_*.txt' into number+volume distributions."""
    df = pd.read_csv(file_path, delimiter=",", encoding="latin1")
    df = df.replace("-", 0)

    bin_edges_um, volume_cols = _extract_spraytec_bins_from_columns(
        list(df.columns))
    if expected_bin_edges_um is not None:
        if len(bin_edges_um) != len(expected_bin_edges_um) or not np.allclose(
                bin_edges_um, expected_bin_edges_um, rtol=0, atol=1e-9
        ):
            raise ValueError(f"Bin edges mismatch in {file_path}")

    bin_centers_um, bin_widths_um = _bin_centers_and_widths(bin_edges_um)

    # First row contains the averaged distribution in your generated files.
    row = df.iloc[0]
    v_percent = pd.to_numeric(
        row[volume_cols], errors="coerce").to_numpy(dtype=float)
    v_percent = np.nan_to_num(v_percent, nan=0.0, posinf=0.0, neginf=0.0)
    if v_percent.sum() > 0:
        v_percent = v_percent / v_percent.sum() * 100.0

    # Convert volume% to number% via d^3 weighting (same as your scripts).
    # Use µm^3; constant factors cancel when normalizing.
    n_raw = v_percent / np.maximum(bin_centers_um, 1e-12) ** 3
    if n_raw.sum() > 0:
        n_percent = n_raw / n_raw.sum() * 100.0
    else:
        n_percent = np.zeros_like(v_percent)

    v_pdf = _percent_to_pdf(v_percent, bin_widths_um)
    n_pdf = _percent_to_pdf(n_percent, bin_widths_um)

    meta = {
        "source": "abe",
        "file": str(file_path),
        "filename": file_path.name,
        "date_time": row.get("Date-Time", None),
        "transmission": row.get("Transmission", None),
        "duration": row.get("Duration", None),
        "time_relative": row.get("Time (relative)", None),
        "num_records": row.get("Number of records in average ", None),
    }

    return BinnedDistribution(
        bin_edges_um=bin_edges_um,
        bin_centers_um=bin_centers_um,
        bin_widths_um=bin_widths_um,
        n_percent=n_percent,
        v_percent=v_percent,
        n_pdf=n_pdf,
        v_pdf=v_pdf,
        meta=meta,
    )


def load_abe_folder(folder: Path, *, expected_bin_edges_um: np.ndarray | None = None) -> list[BinnedDistribution]:
    """Load all Spraytec average .txt files in a folder."""
    files = sorted(folder.glob("average_*.txt"))
    dists: list[BinnedDistribution] = []
    for file_path in files:
        dists.append(load_abe_average_txt(
            file_path, expected_bin_edges_um=expected_bin_edges_um))
    return dists


def _ensure_um_scale(d: np.ndarray) -> np.ndarray:
    """Heuristic: Morgan d might be stored as µm or meters depending on export."""
    d = np.asarray(d, dtype=float)
    finite = d[np.isfinite(d)]
    if finite.size == 0:
        return d
    med = float(np.median(finite))
    # Typical µm droplet sizes: ~1..1000; typical meters: ~1e-6..1e-3.
    if med < 1e-2:  # almost certainly meters
        return d * 1e6
    return d


def _binned_from_samples(
        d_um: np.ndarray,
        bin_edges_um: np.ndarray,
        *,
        meta: dict[str, Any],
) -> BinnedDistribution:
    d_um = _ensure_um_scale(d_um)
    d_um = d_um[np.isfinite(d_um)]

    bin_centers_um, bin_widths_um = _bin_centers_and_widths(bin_edges_um)

    # Number histogram (counts)
    counts, _ = np.histogram(d_um, bins=bin_edges_um)
    total = counts.sum()
    n_percent = (counts / total *
                 100.0) if total > 0 else np.zeros_like(bin_centers_um)

    # Approx volume per droplet ~ d^3 -> volume histogram
    # Digitize to bin index; ignore out-of-range.
    idx = np.digitize(d_um, bin_edges_um) - 1
    valid = (idx >= 0) & (idx < len(bin_centers_um))
    v_sums = np.bincount(idx[valid], weights=(
        d_um[valid] ** 3), minlength=len(bin_centers_um))
    v_total = float(v_sums.sum())
    v_percent = (v_sums / v_total *
                 100.0) if v_total > 0 else np.zeros_like(bin_centers_um)

    n_pdf = _percent_to_pdf(n_percent, bin_widths_um)
    v_pdf = _percent_to_pdf(v_percent, bin_widths_um)

    return BinnedDistribution(
        bin_edges_um=np.asarray(bin_edges_um, dtype=float),
        bin_centers_um=bin_centers_um,
        bin_widths_um=bin_widths_um,
        n_percent=np.asarray(n_percent, dtype=float),
        v_percent=np.asarray(v_percent, dtype=float),
        n_pdf=n_pdf,
        v_pdf=v_pdf,
        meta=meta,
    )


def load_morgan_h5(file_path: Path, *, bin_edges_um: np.ndarray) -> BinnedDistribution:
    with h5py.File(file_path, "r") as f:
        # Based on Morgan_data/Morgantonpz.py
        dset = cast(h5py.Dataset, f["/BSA/Diameter"])
        d = np.asarray(dset[()], dtype=float)

        t_obj = f.get("/BSA/Arrival_Time")
        if isinstance(t_obj, h5py.Dataset):
            t = np.asarray(t_obj[()], dtype=float)
        else:
            t = None

    meta = {
        "source": "morgan",
        "file": str(file_path),
        "filename": file_path.name,
        "arrival_time": t,
    }
    return _binned_from_samples(d, bin_edges_um, meta=meta)


def load_morgan_folder(folder: Path, *, bin_edges_um: np.ndarray, case_name: str | None = None) -> dict[str, Any]:
    """Load all PDA .h5 files in folder.

    Returns:
      {
            'files': [BinnedDistribution, ...],
            'all': BinnedDistribution  # concatenated diameters across files
      }
    """
    files = sorted(folder.glob("*.h5"))
    file_dists: list[BinnedDistribution] = []
    all_d: list[np.ndarray] = []

    for file_path in files:
        dist = load_morgan_h5(file_path, bin_edges_um=bin_edges_um)
        if case_name is not None:
            dist.meta["case"] = case_name
        file_dists.append(dist)

        with h5py.File(file_path, "r") as f:
            dset = cast(h5py.Dataset, f["/BSA/Diameter"])
            all_d.append(np.asarray(dset[()], dtype=float))

    if all_d:
        d_concat = np.concatenate(all_d)
    else:
        d_concat = np.array([])

    all_dist = _binned_from_samples(
        d_concat,
        bin_edges_um,
        meta={
            "source": "morgan",
            "file": str(folder),
            "filename": folder.name,
            "aggregate": True,
            **({"case": case_name} if case_name is not None else {}),
        },
    )

    return {"files": file_dists, "all": all_dist}


def _parse_measurement_index_abe(filename: str) -> int | None:
    # average_<keyphrase>_<n>.txt
    m = re.search(r"_(\d+)(?:_.*)?\.txt$", filename)
    return int(m.group(1)) if m else None


def _parse_measurement_index_morgan(filename: str) -> int | None:
    # ..._e<n>.000001.h5
    m = re.search(r"_e(\d+)\.[\d]+\.h5$", filename)
    return int(m.group(1)) if m else None


def _slug(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_.-]", "", s)
    return s


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, (np.generic,)):
        return x.item()
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    if isinstance(x, (Path,)):
        return str(x)
    return x


def _save_distribution_npz(out_path: Path, dist: BinnedDistribution, meta: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_json = json.dumps({k: _to_jsonable(v)
                           for k, v in meta.items()}, ensure_ascii=False)
    np.savez(
        out_path,
        bin_edges_um=dist.bin_edges_um,
        n_percent=dist.n_percent,
        v_percent=dist.v_percent,
        n_pdf=dist.n_pdf,
        v_pdf=dist.v_pdf,
        meta_json=np.array(meta_json, dtype=object),
    )


def _plot_overlay_pdf(out_path: Path, dists: list[BinnedDistribution], title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from tcm_utils.plot_style import use_tcm_poster_style

    use_tcm_poster_style()

    plt.figure(figsize=(6, 4))
    for dist in dists:
        x = dist.bin_edges_um[:-1]
        y = dist.n_percent
        meas_idx = dist.meta.get("measurement_index", None)
        label = f"m{int(meas_idx)}" if meas_idx is not None else None
        plt.step(x, y, where="post", alpha=0.8, linewidth=1.0, label=label)
    plt.xscale("log")
    plt.xlabel(r"Diameter ($\mathrm{\mu}$m)")
    plt.ylabel("Number distribution (%)")
    plt.grid(which="major")
    plt.title(title)
    plt.legend(title="Measurement", fontsize=8,
               title_fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()


def _canonical_bin_edges_from_abe(abe_folders: list[Path]) -> np.ndarray:
    for folder in abe_folders:
        files = sorted(folder.glob("average_*.txt"))
        if not files:
            continue
        df = pd.read_csv(files[0], delimiter=",", encoding="latin1")
        edges, _ = _extract_spraytec_bins_from_columns(list(df.columns))
        return edges
    raise FileNotFoundError(
        "Could not find any Abe average_*.txt files to derive bin edges")


def export_all_to_new_analysis_data() -> None:
    """Exports *all individual distributions* and one overlay PDF per series.

    Output:
      new_analysis/data/Abe/*.npz
      new_analysis/data/Morgan/*.npz
      new_analysis/data/plots/series_<id>_<dataset>_overlay.pdf
    """
    repo_root = Path(__file__).resolve().parent
    meta_csv = repo_root / "new_analysis" / "metadata.csv"
    out_root = repo_root / "new_analysis" / "data"
    (out_root / "Abe").mkdir(parents=True, exist_ok=True)
    (out_root / "Morgan").mkdir(parents=True, exist_ok=True)
    (out_root / "plots").mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(meta_csv)

    abe_dirs = {
        "1percent": repo_root / "spraytec" / "Averages" / "Unweighted" / "1percent",
        "0dot03": repo_root / "spraytec" / "Averages" / "Unweighted" / "0dot03",
        "0dot25": repo_root / "spraytec" / "Averages" / "Unweighted" / "0dot25",
        "water": repo_root / "spraytec" / "Averages" / "Unweighted" / "water",
        "600k_0dot2": repo_root / "spraytec" / "Averages" / "Unweighted" / "600k_0dot2",
    }
    canonical_edges = _canonical_bin_edges_from_abe(list(abe_dirs.values()))

    def abe_key_from_row(row: pd.Series) -> str:
        liquid = str(row["liquid"])
        conc = float(row["concentration_v_perc"])
        if liquid.lower().startswith("water"):
            return "water"
        if "600" in liquid.lower():
            return "600k_0dot2"
        # PEO2M concentrations
        if abs(conc - 0.03) < 1e-9:
            return "0dot03"
        if abs(conc - 0.25) < 1e-9:
            return "0dot25"
        if abs(conc - 1.0) < 1e-9:
            return "1percent"
        raise ValueError(
            f"Unrecognized Abe row mapping: liquid={liquid}, conc={conc}")

    def morgan_case_from_row(row: pd.Series) -> str:
        liquid = str(row["liquid"])
        conc = float(row["concentration_v_perc"])
        if liquid.lower().startswith("water"):
            return "050B_water"
        # PEO600k concentrations
        if abs(conc - 0.05) < 1e-9:
            return "050B_0pt05wt"
        if abs(conc - 0.1) < 1e-9:
            return "050B_0pt1wt"
        if abs(conc - 0.2) < 1e-9:
            return "050B_0pt2wt"
        if abs(conc - 0.5) < 1e-9:
            return "050B_0pt5wt"
        if abs(conc - 1.0) < 1e-9:
            return "050B_1wt"
        raise ValueError(
            f"Unrecognized Morgan row mapping: liquid={liquid}, conc={conc}")

    # Export each series separately, and build plots
    for _, row in metadata.iterrows():
        series_id = int(row["series"])
        dataset = str(row["dataset"])

        common_meta = {
            "series": series_id,
            "dataset": dataset,
            "liquid": row["liquid"],
            "concentration_v_perc": float(row["concentration_v_perc"]),
            "airflow_m_s": float(row["airflow_m_s"]),
            "relaxation_s": float(row["relaxation_s"]),
            "Deborah": float(row["Deborah"]),
        }

        if dataset.lower() == "abe":
            key = abe_key_from_row(row)
            folder = abe_dirs[key]
            files = sorted(folder.glob("average_*.txt"))
            dists: list[BinnedDistribution] = []
            for f in files:
                dist = load_abe_average_txt(
                    f, expected_bin_edges_um=canonical_edges)
                meas_idx = _parse_measurement_index_abe(f.name)
                dist.meta["measurement_index"] = meas_idx
                meta_out = {**common_meta, **dist.meta,
                            "measurement_index": meas_idx, "folder_key": key}

                out_name = (
                    f"series{series_id:02d}_Abe_"
                    f"{_slug(str(row['liquid']))}_c{row['concentration_v_perc']}_"
                    f"air{row['airflow_m_s']}_m{(meas_idx if meas_idx is not None else 0):03d}.npz"
                )
                _save_distribution_npz(
                    out_root / "Abe" / out_name, dist, meta_out)
                dists.append(dist)

            plot_name = f"series{series_id:02d}_Abe_overlay.pdf"
            title = f"Series {series_id} (Abe) {row['liquid']} {row['concentration_v_perc']}%"
            _plot_overlay_pdf(out_root / "plots" / plot_name, dists, title)

        elif dataset.lower() == "morgan":
            case = morgan_case_from_row(row)
            folder = repo_root / "Morgan_data" / "PDA" / case
            files = sorted(folder.glob("*.h5"))
            dists: list[BinnedDistribution] = []
            for f in files:
                dist = load_morgan_h5(f, bin_edges_um=canonical_edges)
                meas_idx = _parse_measurement_index_morgan(f.name)
                dist.meta["case"] = case
                dist.meta["measurement_index"] = meas_idx
                meta_out = {**common_meta, **dist.meta,
                            "measurement_index": meas_idx, "case": case}

                out_name = (
                    f"series{series_id:02d}_Morgan_"
                    f"{_slug(str(row['liquid']))}_c{row['concentration_v_perc']}_"
                    f"air{row['airflow_m_s']}_m{(meas_idx if meas_idx is not None else 0):03d}.npz"
                )
                _save_distribution_npz(
                    out_root / "Morgan" / out_name, dist, meta_out)
                dists.append(dist)

            plot_name = f"series{series_id:02d}_Morgan_overlay.pdf"
            title = f"Series {series_id} (Morgan) {row['liquid']} {row['concentration_v_perc']}%"
            _plot_overlay_pdf(out_root / "plots" / plot_name, dists, title)
        else:
            raise ValueError(f"Unknown dataset in metadata.csv: {dataset}")


def _load_meta_json(npz: np.lib.npyio.NpzFile) -> dict[str, Any]:
    raw = npz["meta_json"]
    # Stored as 0-d object array containing a JSON string.
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    if not isinstance(raw, str):
        raw = str(raw)
    return json.loads(raw)


def _mode_from_npz(npz_path: Path) -> tuple[float, float]:
    """Return (mode_um, max_bin_percent)."""
    data = np.load(npz_path, allow_pickle=True)
    edges = np.asarray(data["bin_edges_um"], dtype=float)
    n_percent = np.asarray(data["n_percent"], dtype=float)
    centers = (edges[:-1] + edges[1:]) / 2.0
    idx = int(np.nanargmax(n_percent))
    mode_um = float(centers[idx])
    max_bin = float(np.nanmax(n_percent))
    return mode_um, max_bin


def _write_peaky_skip_log(
    *,
    data_root: Path,
    metadata_csv: Path,
    out_dir: Path,
    peaky_threshold_percent: float,
) -> Path:
    """Write a CSV listing which measurements would be skipped by the peaky rule."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(metadata_csv)

    entries: list[dict[str, Any]] = []
    for _, r in meta.iterrows():
        series_id = int(r["series"])
        dataset = str(r["dataset"])

        if dataset.lower() == "abe":
            folder = data_root / "Abe"
        elif dataset.lower() == "morgan":
            folder = data_root / "Morgan"
        else:
            continue

        for npz_path in sorted(folder.glob(f"series{series_id:02d}_*.npz")):
            mode_um, max_bin = _mode_from_npz(npz_path)
            if max_bin <= peaky_threshold_percent:
                continue

            data = np.load(npz_path, allow_pickle=True)
            meta_json = _load_meta_json(data)
            entries.append(
                {
                    "series": series_id,
                    "dataset": dataset,
                    "npz": str(npz_path),
                    "measurement_index": meta_json.get("measurement_index", None),
                    "source_file": meta_json.get("file", None),
                    "source_filename": meta_json.get("filename", None),
                    "max_bin_percent": max_bin,
                    "mode_um": mode_um,
                    "threshold_percent": peaky_threshold_percent,
                }
            )

    df = pd.DataFrame(entries).sort_values(
        ["dataset", "series", "measurement_index"], na_position="last")
    out_path = out_dir / \
        f"skipped_measurements_maxBinGt{int(peaky_threshold_percent)}pct.csv"
    df.to_csv(out_path, index=False)
    return out_path


def _series_mode_stats(
    *,
    data_root: Path,
    metadata_csv: Path,
    skip_peaky: bool,
    peaky_threshold_percent: float = 50.0,
) -> pd.DataFrame:
    """Compute mean/std of per-measurement modes for each series."""
    meta = pd.read_csv(metadata_csv)
    rows: list[dict[str, Any]] = []
    for _, r in meta.iterrows():
        series_id = int(r["series"])
        dataset = str(r["dataset"])
        deborah = float(r["Deborah"])
        relaxation_s = float(r["relaxation_s"])

        if dataset.lower() == "abe":
            folder = data_root / "Abe"
        elif dataset.lower() == "morgan":
            folder = data_root / "Morgan"
        else:
            continue

        npz_files = sorted(folder.glob(f"series{series_id:02d}_*.npz"))
        modes: list[float] = []
        skipped = 0
        for f in npz_files:
            mode_um, max_bin = _mode_from_npz(f)
            if skip_peaky and max_bin > peaky_threshold_percent:
                skipped += 1
                continue
            modes.append(mode_um)

        if len(modes) == 0:
            mean_mode = np.nan
            std_mode = np.nan
        elif len(modes) == 1:
            mean_mode = float(modes[0])
            std_mode = 0.0
        else:
            mean_mode = float(np.mean(modes))
            std_mode = float(np.std(modes, ddof=1))

        rows.append(
            {
                "series": series_id,
                "dataset": dataset,
                "Deborah": deborah,
                "relaxation_s": relaxation_s,
                "mode_mean_um": mean_mode,
                "mode_std_um": std_mode,
                "n_measurements": len(npz_files),
                "n_used": len(modes),
                "n_skipped": skipped,
                "skip_peaky": skip_peaky,
                "peaky_threshold_percent": peaky_threshold_percent,
            }
        )

    return pd.DataFrame(rows)


def _plot_mode_vs_x(
    *,
    x_col: str,
    x_label: str,
    data_root: Path,
    metadata_csv: Path,
    out_dir: Path,
    peaky_threshold_percent: float = 40.0,
    loglog: bool,
) -> None:
    """Helper to plot mean±std mode vs a chosen x variable."""
    from matplotlib.ticker import FuncFormatter, NullFormatter
    from tcm_utils.plot_style import append_unit_to_last_ticklabel, use_tcm_poster_style

    for skip in (False, True):
        df = _series_mode_stats(
            data_root=data_root,
            metadata_csv=metadata_csv,
            skip_peaky=skip,
            peaky_threshold_percent=peaky_threshold_percent,
        )

        use_tcm_poster_style()

        title = f"Mode vs {x_label}"
        title += (
            f" (skip max-bin > {peaky_threshold_percent:.0f}%)" if skip else " (no skipping)"
        )

        suffix = "skipOn" if skip else "skipOff"
        ll = "_loglog" if loglog else ""
        out_path = out_dir / f"mode_vs_{x_col}{ll}_{suffix}.pdf"

        if not loglog:
            plt.figure(figsize=(6, 4.6))
            for dataset, marker in (("Abe", "o"), ("Morgan", "s")):
                sub = df[df["dataset"].str.lower() == dataset.lower()].copy()
                sub = sub[np.isfinite(sub["mode_mean_um"])].sort_values(x_col)
                if sub.empty:
                    continue
                base_lw = float(plt.rcParams.get("lines.linewidth", 2.0))
                base_ms = float(plt.rcParams.get("lines.markersize", 6.0))
                lw = 0.7 * base_lw
                ms = 0.85 * base_ms
                plt.errorbar(
                    sub[x_col],
                    sub["mode_mean_um"],
                    yerr=sub["mode_std_um"],
                    fmt=marker,
                    capsize=3,
                    capthick=lw,
                    elinewidth=lw,
                    linewidth=lw,
                    markersize=ms,
                )

            plt.xlabel(x_label)
            plt.ylabel(r"Mode diameter ($\mathrm{\mu}$m)")
            plt.grid(which="major")
            plt.tight_layout()
            plt.savefig(out_path, format="pdf")
            plt.close()
            continue

        # Broken-axis version for "loglog": left axis shows x=0 points (linear),
        # right axis shows x>0 points on log scale.
        fig, (ax0, ax1) = plt.subplots(
            1,
            2,
            sharey=True,
            # Narrower overall; make the x=0 panel much thinner.
            figsize=(5.4, 4.8),
            gridspec_kw={"width_ratios": [0.35, 4.65]},
        )

        # Determine positive range for the log axis.
        x_all = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
        x_all = x_all[np.isfinite(x_all)]
        x_pos = x_all[x_all > 0]
        min_pos = float(np.min(x_pos)) if x_pos.size else 1.0
        max_pos = float(np.max(x_pos)) if x_pos.size else 10.0

        for dataset, marker in (("Abe", "o"), ("Morgan", "s")):
            sub = df[df["dataset"].str.lower() == dataset.lower()].copy()
            sub = sub[np.isfinite(sub["mode_mean_um"]) &
                      np.isfinite(sub[x_col])]
            if sub.empty:
                continue

            sub0 = sub[sub[x_col] <= 0]
            sub1 = sub[sub[x_col] > 0]

            base_lw = float(plt.rcParams.get("lines.linewidth", 2.0))
            base_ms = float(plt.rcParams.get("lines.markersize", 6.0))
            lw = 0.7 * base_lw
            ms = 0.85 * base_ms

            label = dataset
            if not sub1.empty:
                x1_vals = sub1[x_col]
                if x_col == "relaxation_s":
                    x1_vals = x1_vals * 1000.0  # seconds -> ms
                ax1.errorbar(
                    x1_vals,
                    sub1["mode_mean_um"],
                    yerr=sub1["mode_std_um"],
                    fmt=marker,
                    capsize=3,
                    capthick=lw,
                    elinewidth=lw,
                    linewidth=lw,
                    markersize=ms,
                )

            if not sub0.empty:
                x0_vals = sub0[x_col]
                if x_col == "relaxation_s":
                    x0_vals = x0_vals * 1000.0  # seconds -> ms
                ax0.errorbar(
                    x0_vals,
                    sub0["mode_mean_um"],
                    yerr=sub0["mode_std_um"],
                    fmt=marker,
                    capsize=3,
                    capthick=lw,
                    elinewidth=lw,
                    linewidth=lw,
                    markersize=ms,
                )

        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax0.set_yscale("log")

        # Left axis: show only the 0 tick.
        ax0.set_xticks([0.0])
        ax0.set_xticklabels(["0"])
        ax0.minorticks_off()

        # Axis limits
        if x_col == "relaxation_s":
            # Display relaxation time in milliseconds on the x-axis (right panel).
            # Range: 0.2 ms .. 80 ms
            ax1.set_xlim(0.2, 80.0)
            ax1.set_xticks([0.2, 1.0, 10.0])
            ax1.xaxis.set_major_formatter(
                FuncFormatter(lambda v, pos: f"{v:g}"))
            ax1.xaxis.set_minor_formatter(NullFormatter())
            append_unit_to_last_ticklabel(
                ax1, axis="x", unit="ms", fmt="{x:g}")

            # Keep the left panel focused tightly around 0 ms.
            ax0.set_xlim(-0.02, 0.02)
            x_label_used = "Relaxation time"
        else:
            # Left axis: zoom around 0 to make the point(s) visible.
            left_halfwidth = min(0.5, max(min_pos * 0.2, 1e-12))
            ax0.set_xlim(-left_halfwidth, left_halfwidth)
            # Right axis: normal log span.
            ax1.set_xlim(min_pos * 0.9, max_pos * 1.1)
            x_label_used = x_label

            ax1.xaxis.set_major_formatter(
                FuncFormatter(lambda v, pos: f"{v:g}"))
            ax1.xaxis.set_minor_formatter(NullFormatter())

        # Avoid scientific notation for y ticks.
        ax0.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:g}"))
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:g}"))

        ax0.grid(which="major")
        ax1.grid(which="major")
        grid_lw = float(plt.rcParams.get("grid.linewidth", 1.0))
        for gl in ax0.get_xgridlines() + ax0.get_ygridlines():
            gl.set_linewidth(grid_lw)
        for gl in ax1.get_xgridlines() + ax1.get_ygridlines():
            gl.set_linewidth(grid_lw)

        # Cosmetic: hide the touching spines.
        ax0.spines["right"].set_visible(False)
        ax1.spines["left"].set_visible(False)
        ax0.yaxis.tick_left()

        # No y ticks on the right subplot.
        ax1.tick_params(axis="y", which="both", left=False, right=False,
                        labelleft=False, labelright=False)

        fig.supxlabel(x_label_used)
        ax0.set_ylabel(r"Mode diameter ($\mathrm{\mu}$m)")

        fig.tight_layout()
        # Move the two halves further apart.
        fig.subplots_adjust(wspace=0.22)

        # Diagonal break marks: draw after layout so they stay aligned.
        add_broken_xaxis_marks(fig, ax0, ax1, size=0.008)
        fig.savefig(out_path, format="pdf")
        plt.close(fig)


def plot_mode_summaries(
    *,
    data_root: Path | None = None,
    metadata_csv: Path | None = None,
    out_dir: Path | None = None,
    peaky_threshold_percent: float = 40.0,
) -> None:
    """Generate mode summary plots:

    - vs Deborah (linear + log-log), skip off/on
    - vs relaxation time (linear + log-log), skip off/on
    """
    repo_root = Path(__file__).resolve().parent
    data_root = data_root or (repo_root / "new_analysis" / "data")
    metadata_csv = metadata_csv or (
        repo_root / "new_analysis" / "metadata.csv")
    out_dir = out_dir or (repo_root / "new_analysis" / "data" / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Log which measurements are excluded by the peaky-threshold rule.
    log_dir = data_root / "logs"
    _write_peaky_skip_log(
        data_root=data_root,
        metadata_csv=metadata_csv,
        out_dir=log_dir,
        peaky_threshold_percent=peaky_threshold_percent,
    )

    for loglog in (False, True):
        _plot_mode_vs_x(
            x_col="Deborah",
            x_label="Deborah",
            data_root=data_root,
            metadata_csv=metadata_csv,
            out_dir=out_dir,
            peaky_threshold_percent=peaky_threshold_percent,
            loglog=loglog,
        )
        _plot_mode_vs_x(
            x_col="relaxation_s",
            x_label="Relaxation time (s)",
            data_root=data_root,
            metadata_csv=metadata_csv,
            out_dir=out_dir,
            peaky_threshold_percent=peaky_threshold_percent,
            loglog=loglog,
        )


def load_everything() -> tuple[dict[str, list[BinnedDistribution]], dict[str, dict[str, Any]]]:
    """Convenience: loads the exact folders you listed."""
    root = Path(__file__).resolve().parent

    abe_dirs = {
        "1percent": root / "spraytec" / "Averages" / "Unweighted" / "1percent",
        "0dot03": root / "spraytec" / "Averages" / "Unweighted" / "0dot03",
        "0dot25": root / "spraytec" / "Averages" / "Unweighted" / "0dot25",
        "water": root / "spraytec" / "Averages" / "Unweighted" / "water",
        "600k_0dot2": root / "spraytec" / "Averages" / "Unweighted" / "600k_0dot2",
    }

    # Determine canonical bin edges from the first Abe file we can find.
    canonical_edges: np.ndarray | None = None
    for folder in abe_dirs.values():
        files = sorted(folder.glob("average_*.txt"))
        if files:
            canonical_edges, _ = _extract_spraytec_bins_from_columns(
                list(pd.read_csv(files[0], delimiter=",", encoding="latin1").columns))
            break
    if canonical_edges is None:
        raise FileNotFoundError(
            "Could not find any Abe average_*.txt files to derive bin edges")

    abe: dict[str, list[BinnedDistribution]] = {}
    for name, folder in abe_dirs.items():
        abe[name] = load_abe_folder(
            folder, expected_bin_edges_um=canonical_edges)

    morgan_dirs = {
        "050B_0pt1wt": root / "Morgan_data" / "PDA" / "050B_0pt1wt",
        "050B_0pt2wt": root / "Morgan_data" / "PDA" / "050B_0pt2wt",
        "050B_0pt5wt": root / "Morgan_data" / "PDA" / "050B_0pt5wt",
        "050B_0pt05wt": root / "Morgan_data" / "PDA" / "050B_0pt05wt",
        "050B_1wt": root / "Morgan_data" / "PDA" / "050B_1wt",
        "050B_water": root / "Morgan_data" / "PDA" / "050B_water",
    }

    morgan: dict[str, dict[str, Any]] = {}
    for name, folder in morgan_dirs.items():
        morgan[name] = load_morgan_folder(folder, bin_edges_um=canonical_edges)

    return abe, morgan


if __name__ == "__main__":
    export_all_to_new_analysis_data()
    print("Export complete -> new_analysis/data")
    plot_mode_summaries(peaky_threshold_percent=40.0)
    print("Mode plots complete -> new_analysis/data/plots")
