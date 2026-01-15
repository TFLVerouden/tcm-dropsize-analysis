"""new_analysis.py

One-off analysis script.

Goal
----
Get directly comparable drop-size distributions from:
- Abe (Spraytec averaged .txt files)
- Morgan (PDA .h5 files)

We force both datasets onto the *same* diameter bins (µm) by taking the Spraytec
bin edges as canonical and binning Morgan samples onto those.

This is intentionally a read-through linear script, not a reusable library.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, cast
import json

import numpy as np
import pandas as pd
import h5py

import matplotlib.pyplot as plt

from natsort import natsorted

from tcm_utils.plot_style import (
    add_broken_xaxis_marks,
    append_unit_to_last_ticklabel,
    plot_binned_area,
    raise_axis_frame,
    set_grid,
    set_log_axes,
    use_tcm_poster_style,
    set_ticks_every
)
###############################################################################
# Small utilities (kept tiny on purpose)
###############################################################################

_VOLUME_COL_RE = re.compile(r"^%\s*V\s*\((?P<lo>[\d.]+)\-(?P<hi>[\d.]+)")


def _spraytec_bin_edges_and_volume_cols(columns: list[str]) -> tuple[np.ndarray, list[str]]:
    """Parse Spraytec columns -> (bin_edges_um, volume_cols_sorted).

    Spraytec average files have columns like:
        % V (0.100-0.117µm)
    (the µ sometimes gets mangled; we ignore everything after the numbers).
    """

    pairs: list[tuple[float, float, str]] = []
    for col in columns:
        m = _VOLUME_COL_RE.search(col)
        if m:
            pairs.append((float(m.group("lo")), float(m.group("hi")), col))

    pairs.sort(key=lambda t: t[0])
    volume_cols = [p[2] for p in pairs]
    edges = np.asarray([pairs[0][0]] + [p[1] for p in pairs], dtype=float)
    return edges, volume_cols


def _ensure_um(d: np.ndarray) -> np.ndarray:
    """Morgan diameters are either in meters or in µm; make them µm."""

    d = np.asarray(d, dtype=float)
    finite = d[np.isfinite(d)]
    if finite.size == 0:
        return d
    return d * 1e6 if float(np.median(finite)) < 1e-2 else d


def _slug(s: Any) -> str:
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


def _save_npz(out_path: Path, *, bin_edges_um: np.ndarray, n_percent: np.ndarray, v_percent: np.ndarray, n_pdf: np.ndarray, v_pdf: np.ndarray, meta: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_json = json.dumps({k: _to_jsonable(v)
                           for k, v in meta.items()}, ensure_ascii=False)
    np.savez(
        out_path,
        bin_edges_um=np.asarray(bin_edges_um, dtype=float),
        n_percent=np.asarray(n_percent, dtype=float),
        v_percent=np.asarray(v_percent, dtype=float),
        n_pdf=np.asarray(n_pdf, dtype=float),
        v_pdf=np.asarray(v_pdf, dtype=float),
        meta_json=np.array(meta_json, dtype=object),
    )


###############################################################################
# Script starts here
###############################################################################

# 1) File/folder layout
# --------------------
repo_root = Path(__file__).resolve().parent

metadata_csv = repo_root / "new_analysis" / "metadata.csv"
out_root = repo_root / "new_analysis" / "data"
out_abe = out_root / "Abe"
out_morgan = out_root / "Morgan"
out_plots = out_root / "plots"
out_plots_all = out_plots / "all_distributions"
out_logs = out_root / "logs"

out_abe.mkdir(parents=True, exist_ok=True)
out_morgan.mkdir(parents=True, exist_ok=True)
out_plots.mkdir(parents=True, exist_ok=True)
out_plots_all.mkdir(parents=True, exist_ok=True)
out_logs.mkdir(parents=True, exist_ok=True)

# Input folders (hard-coded because this is one-off work)
abe_dirs = {
    "1percent": repo_root / "spraytec" / "Averages" / "Unweighted" / "1percent",
    "0dot03": repo_root / "spraytec" / "Averages" / "Unweighted" / "0dot03",
    "0dot25": repo_root / "spraytec" / "Averages" / "Unweighted" / "0dot25",
    "water": repo_root / "spraytec" / "Averages" / "Unweighted" / "water",
    "600k_0dot2": repo_root / "spraytec" / "Averages" / "Unweighted" / "600k_0dot2",
}

morgan_root = repo_root / "Morgan_data" / "PDA"

# 2) Choose canonical diameter bins
# --------------------------------
# We take the Spraytec bin edges as the shared axis between the two datasets.
first_abe_file: Path | None = None
for folder in abe_dirs.values():
    files = natsorted(folder.glob("average_*.txt"), key=lambda p: p.name)
    if files:
        first_abe_file = files[0]
        break

if first_abe_file is None:
    raise FileNotFoundError("No Spraytec average_*.txt files found")

first_df = pd.read_csv(first_abe_file, delimiter=",",
                       encoding="latin1").replace("-", 0)
bin_edges_um, spraytec_volume_cols = _spraytec_bin_edges_and_volume_cols(
    list(first_df.columns))
bin_centers_um = (bin_edges_um[:-1] + bin_edges_um[1:]) / 2.0
bin_widths_um = np.diff(bin_edges_um)

# 3) Read metadata
# ---------------
metadata = pd.read_csv(metadata_csv)

# 4) Export per-measurement distributions (NPZ) + per-series overlay plots
# ----------------------------------------------------------------------
# What we store in each .npz:
#   - bin_edges_um
#   - n_percent and v_percent per bin
#   - n_pdf and v_pdf (1/µm)
#   - meta_json with the series metadata and the source filename

use_tcm_poster_style()

for _, row in metadata.iterrows():
    series_id = int(row["series"])
    dataset = str(row["dataset"]).strip()

    common_meta = {
        "series": series_id,
        "dataset": dataset,
        "liquid": row["liquid"],
        "concentration_v_perc": float(row["concentration_v_perc"]),
        "airflow_m_s": float(row["airflow_m_s"]),
        "relaxation_s": float(row["relaxation_s"]),
        "Deborah": float(row["Deborah"]),
    }

    # -----------------
    # Abe (Spraytec)
    # -----------------
    if dataset.lower() == "abe":
        liquid = str(row["liquid"])
        conc = float(row["concentration_v_perc"])

        if liquid.lower().startswith("water"):
            abe_key = "water"
        elif "600" in liquid.lower():
            abe_key = "600k_0dot2"
        elif abs(conc - 0.03) < 1e-9:
            abe_key = "0dot03"
        elif abs(conc - 0.25) < 1e-9:
            abe_key = "0dot25"
        else:
            abe_key = "1percent"

        folder = abe_dirs[abe_key]
        files = natsorted(folder.glob("average_*.txt"), key=lambda p: p.name)

        # Overlay plot: one distribution per measurement (area plots).
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        set_grid(ax, mode="horizontal", on=True)
        set_log_axes(ax, x=True)

        color_cycle = plt.rcParams.get(
            "axes.prop_cycle").by_key().get("color", ["C0"])

        for i, f in enumerate(files):
            m = re.search(r"_(\d+)(?:_.*)?\.txt$", f.name)
            measurement_index = int(m.group(1)) if m else None

            df = pd.read_csv(f, delimiter=",",
                             encoding="latin1").replace("-", 0)
            row0 = df.iloc[0]

            v_percent = pd.to_numeric(
                row0[spraytec_volume_cols], errors="coerce").to_numpy(dtype=float)
            v_percent = np.nan_to_num(
                v_percent, nan=0.0, posinf=0.0, neginf=0.0)
            v_percent = v_percent / v_percent.sum() * 100.0

            # Convert volume-% to number-% with d^3 weighting.
            n_raw = v_percent / (bin_centers_um ** 3)
            n_percent = n_raw / n_raw.sum() * 100.0

            v_pdf = (v_percent / 100.0) / bin_widths_um
            n_pdf = (n_percent / 100.0) / bin_widths_um

            meta_out = {
                **common_meta,
                "source": "abe",
                "folder_key": abe_key,
                "file": str(f),
                "filename": f.name,
                "measurement_index": measurement_index,
                "date_time": row0.get("Date-Time", None),
                "transmission": row0.get("Transmission", None),
                "duration": row0.get("Duration", None),
                "time_relative": row0.get("Time (relative)", None),
                "num_records": row0.get("Number of records in average ", None),
            }

            out_name = (
                f"series{series_id:02d}_Abe_"
                f"{_slug(row['liquid'])}_c{row['concentration_v_perc']}_"
                f"air{row['airflow_m_s']}_m{(measurement_index if measurement_index is not None else 0):03d}.npz"
            )
            _save_npz(
                out_abe / out_name,
                bin_edges_um=bin_edges_um,
                n_percent=n_percent,
                v_percent=v_percent,
                n_pdf=n_pdf,
                v_pdf=v_pdf,
                meta=meta_out,
            )

            color = color_cycle[i % len(color_cycle)]
            stairs = plot_binned_area(
                ax,
                bin_edges_um,
                n_percent,
                x_mode="edges",
                color=color,
                alpha=0.15,
                outline=True,
                outline_linewidth=2,
                outline_color=color,
                white_underlay=True,
                zorder_fill=6,
                zorder_outline=7,
            )
            if stairs is not None:
                label = f"m{measurement_index}" if measurement_index is not None else f.name
                stairs.set_label(label)

        ax.set_xlabel(r"Diameter (μm)")
        ax.set_ylabel("Number distribution (%)")
        ax.set_title(
            f"Series {series_id} (Abe) {row['liquid']} {row['concentration_v_perc']}%")
        ax.legend(frameon=False)
        raise_axis_frame(ax)
        fig.tight_layout()
        fig.savefig(
            out_plots_all / f"series{series_id:02d}_Abe_overlay.pdf", format="pdf", bbox_inches="tight")
        plt.close(fig)

        # Special series00 example plot (extra output; does not replace normal overlay).
        if series_id == 0:
            fig_ex, ax_ex = plt.subplots(1, 1, figsize=(4, 3.5))
            set_grid(ax_ex, mode="none", on=False)
            set_log_axes(ax_ex, x=True)

            for f in files:
                df = pd.read_csv(f, delimiter=",",
                                 encoding="latin1").replace("-", 0)
                row0 = df.iloc[0]
                v_percent = pd.to_numeric(
                    row0[spraytec_volume_cols], errors="coerce").to_numpy(dtype=float)
                v_percent = np.nan_to_num(
                    v_percent, nan=0.0, posinf=0.0, neginf=0.0)
                v_percent = v_percent / v_percent.sum() * 100.0
                n_raw = v_percent / (bin_centers_um ** 3)
                n_percent = n_raw / n_raw.sum() * 100.0
                plot_binned_area(
                    ax_ex,
                    bin_edges_um,
                    n_percent,
                    x_mode="edges",
                    color="C0",
                    alpha=0.3,
                    outline=False,
                    white_underlay=True,
                    zorder_fill=6,
                    zorder_outline=7,
                )

            ax_ex.set_xlim(0.1, 1000)
            ax_ex.set_ylim(0, 20)
            ax_ex.set_xlabel(r"Diameter (μm)")
            ax_ex.set_ylabel("Nr. distr. (%)")

            set_ticks_every(ax_ex, axis="y", step=5)

            raise_axis_frame(ax_ex)
            fig_ex.tight_layout()
            fig_ex.savefig(
                out_plots /
                f"example_distribution_series{series_id:02d}_Abe_{_slug(row['liquid'])}_c{row['concentration_v_perc']}_air{row['airflow_m_s']}.pdf",
                format="pdf",
                bbox_inches="tight",
            )

            # Change the xlims to zoomed in view and save again with different name
            ax_ex.set_xlim(4, 200)
            fig_ex.tight_layout()
            fig_ex.savefig(
                out_plots /
                f"example_distribution_series{series_id:02d}_Abe_{_slug(row['liquid'])}_c{row['concentration_v_perc']}_air{row['airflow_m_s']}_zoomed.pdf",
                format="pdf",
                bbox_inches="tight",
            )

            plt.close(fig_ex)

        continue

    # -----------------
    # Morgan (PDA)
    # -----------------
    if dataset.lower() == "morgan":
        liquid = str(row["liquid"])
        conc = float(row["concentration_v_perc"])

        if liquid.lower().startswith("water"):
            case = "050B_water"
        elif abs(conc - 0.05) < 1e-9:
            case = "050B_0pt05wt"
        elif abs(conc - 0.1) < 1e-9:
            case = "050B_0pt1wt"
        elif abs(conc - 0.2) < 1e-9:
            case = "050B_0pt2wt"
        elif abs(conc - 0.5) < 1e-9:
            case = "050B_0pt5wt"
        else:
            case = "050B_1wt"

        folder = morgan_root / case
        files = natsorted(folder.glob("*.h5"), key=lambda p: p.name)

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        set_grid(ax, mode="horizontal", on=True)
        set_log_axes(ax, x=True)

        color_cycle = plt.rcParams.get(
            "axes.prop_cycle").by_key().get("color", ["C0"])

        for i, f in enumerate(files):
            m = re.search(r"_e(\d+)\.[\d]+\.h5$", f.name)
            measurement_index = int(m.group(1)) if m else None

            with h5py.File(f, "r") as h5:
                dset = cast(h5py.Dataset, h5["/BSA/Diameter"])
                d_um = _ensure_um(np.asarray(dset[()], dtype=float))
                d_um = d_um[np.isfinite(d_um)]

                t_obj = h5.get("/BSA/Arrival_Time")
                arrival_time = np.asarray(t_obj[()], dtype=float) if isinstance(
                    t_obj, h5py.Dataset) else None

            # Number histogram (% per bin)
            counts, _ = np.histogram(d_um, bins=bin_edges_um)
            n_percent = counts / counts.sum() * 100.0

            # Volume histogram (% per bin): weight each droplet by d^3
            idx = np.digitize(d_um, bin_edges_um) - 1
            valid = (idx >= 0) & (idx < len(bin_centers_um))
            v_sums = np.bincount(idx[valid], weights=(
                d_um[valid] ** 3), minlength=len(bin_centers_um))
            v_percent = v_sums / v_sums.sum() * 100.0

            v_pdf = (v_percent / 100.0) / bin_widths_um
            n_pdf = (n_percent / 100.0) / bin_widths_um

            meta_out = {
                **common_meta,
                "source": "morgan",
                "case": case,
                "file": str(f),
                "filename": f.name,
                "measurement_index": measurement_index,
                "arrival_time": arrival_time,
            }

            out_name = (
                f"series{series_id:02d}_Morgan_"
                f"{_slug(row['liquid'])}_c{row['concentration_v_perc']}_"
                f"air{row['airflow_m_s']}_m{(measurement_index if measurement_index is not None else 0):03d}.npz"
            )
            _save_npz(
                out_morgan / out_name,
                bin_edges_um=bin_edges_um,
                n_percent=n_percent,
                v_percent=v_percent,
                n_pdf=n_pdf,
                v_pdf=v_pdf,
                meta=meta_out,
            )

            color = color_cycle[i % len(color_cycle)]

            stairs = plot_binned_area(
                ax,
                bin_edges_um,
                n_percent,
                x_mode="edges",
                color=color,
                alpha=0.15,
                outline=True,
                outline_linewidth=2,
                outline_color=color,
                white_underlay=True,
                zorder_fill=6,
                zorder_outline=7,
            )
            if stairs is not None:
                label = f"m{measurement_index}" if measurement_index is not None else f.name
                stairs.set_label(label)

        ax.set_xlim(max(0.1, float(bin_edges_um[0])), float(bin_edges_um[-1]))
        ax.set_xlabel(r"Diameter (μm)")
        ax.set_ylabel("Number distribution (%)")
        ax.set_title(
            f"Series {series_id} (Morgan) {row['liquid']} {row['concentration_v_perc']}%")
        ax.legend(frameon=False)
        raise_axis_frame(ax)
        fig.tight_layout()
        fig.savefig(
            out_plots_all / f"series{series_id:02d}_Morgan_overlay.pdf",
            format="pdf",
            bbox_inches="tight",
        )
        plt.close(fig)
        continue

    raise ValueError(f"Unknown dataset in metadata.csv: {dataset}")


# 5) Mode summaries (mean ± std) vs Deborah / relaxation time
# ----------------------------------------------------------
# This uses the exported NPZ files so we don't re-import raw data here.

def _load_meta_json(npz: np.lib.npyio.NpzFile) -> dict[str, Any]:
    raw = npz["meta_json"]
    raw = raw.item() if isinstance(raw, np.ndarray) else raw
    return json.loads(str(raw))


def _mode_and_peak_percent(*, bin_edges_um: np.ndarray, n_percent: np.ndarray) -> tuple[float, float]:
    centers = (bin_edges_um[:-1] + bin_edges_um[1:]) / 2.0
    i = int(np.nanargmax(n_percent))
    return float(centers[i]), float(np.nanmax(n_percent))


peaky_threshold_percent = 40.0

# Per-series aggregate: compute mode stats from the per-measurement NPZ files.
rows: list[dict[str, Any]] = []
skipped_rows: list[dict[str, Any]] = []

for _, r in metadata.iterrows():
    series_id = int(r["series"])
    dataset = str(r["dataset"]).strip()

    folder = out_abe if dataset.lower() == "abe" else out_morgan
    npz_files = sorted(folder.glob(f"series{series_id:02d}_*.npz"))

    # Two passes: skip-off and skip-on.
    for skip_peaky in (False, True):
        modes: list[float] = []
        n_skipped = 0

        for p in npz_files:
            data = np.load(p, allow_pickle=True)
            edges = np.asarray(data["bin_edges_um"], dtype=float)
            n_percent = np.asarray(data["n_percent"], dtype=float)
            mode_um, peak = _mode_and_peak_percent(
                bin_edges_um=edges, n_percent=n_percent)

            if skip_peaky and peak > peaky_threshold_percent:
                n_skipped += 1
                meta_json = _load_meta_json(data)
                skipped_rows.append(
                    {
                        "series": series_id,
                        "dataset": dataset,
                        "npz": str(p),
                        "measurement_index": meta_json.get("measurement_index", None),
                        "source_filename": meta_json.get("filename", None),
                        "max_bin_percent": peak,
                        "mode_um": mode_um,
                        "threshold_percent": peaky_threshold_percent,
                    }
                )
                continue

            modes.append(mode_um)

        mode_mean_um = float(np.mean(modes)) if modes else np.nan
        mode_std_um = float(np.std(modes, ddof=1)) if len(
            modes) > 1 else (0.0 if len(modes) == 1 else np.nan)

        rows.append(
            {
                "series": series_id,
                "dataset": dataset,
                "Deborah": float(r["Deborah"]),
                "relaxation_s": float(r["relaxation_s"]),
                "mode_mean_um": mode_mean_um,
                "mode_std_um": mode_std_um,
                "n_measurements": len(npz_files),
                "n_used": len(modes),
                "n_skipped": n_skipped,
                "skip_peaky": skip_peaky,
                "peaky_threshold_percent": peaky_threshold_percent,
            }
        )

mode_df = pd.DataFrame(rows)
skip_df = pd.DataFrame(skipped_rows)
if not skip_df.empty:
    skip_df.sort_values(["dataset", "series", "measurement_index"], na_position="last").to_csv(
        out_logs /
        f"skipped_measurements_maxBinGt{int(peaky_threshold_percent)}pct.csv",
        index=False,
    )


def _plot_mode_vs(x_col: str, *, x_label: str, loglog: bool) -> None:
    """Plot mode mean±std vs x, for Abe and Morgan, skip-off + skip-on."""
    from matplotlib.ticker import FuncFormatter, NullFormatter, NullLocator

    for skip_peaky in (False, True):
        df = mode_df[mode_df["skip_peaky"] == skip_peaky].copy()

        use_tcm_poster_style()

        # Explicit dataset colors (swap Abe/Morgan).
        cycle = plt.rcParams.get("axes.prop_cycle").by_key().get(
            "color", ["C0", "C1"])
        c0 = cycle[0] if len(cycle) > 0 else "C0"
        c1 = cycle[1] if len(cycle) > 1 else "C1"
        ds_color = {"Abe": c0, "Morgan": c1}

        suffix = "skipOn" if skip_peaky else "skipOff"
        ll = "_loglog" if loglog else ""
        out_path = out_plots / f"mode_vs_{x_col}{ll}_{suffix}.pdf"

        # Simple (non-broken) version
        if not loglog:
            plt.figure(figsize=(6, 4.6))
            # Plot Morgan first, then Abe, so Abe sits on top.
            for ds, marker, z in (("Morgan", "s", 2), ("Abe", "o", 3)):
                sub = df[df["dataset"].str.lower() == ds.lower()].copy()
                sub = sub[np.isfinite(sub["mode_mean_um"])].sort_values(x_col)
                if sub.empty:
                    continue

                clr = ds_color.get(ds, None)

                base_lw = float(plt.rcParams.get("lines.linewidth", 2.0))
                base_ms = float(plt.rcParams.get("lines.markersize", 6.0))
                lw = 0.7 * base_lw
                ms = 0.85 * base_ms

                plt.errorbar(
                    sub[x_col],
                    sub["mode_mean_um"],
                    yerr=sub["mode_std_um"],
                    fmt=marker,
                    color=clr,
                    ecolor=clr,
                    capsize=3,
                    capthick=lw,
                    elinewidth=lw,
                    linewidth=lw,
                    markersize=ms,
                    zorder=z,
                )

            plt.xlabel(x_label)
            plt.ylabel(r"Mode diameter (μm)")
            plt.grid(which="major")
            plt.tight_layout()
            plt.savefig(out_path, format="pdf", bbox_inches="tight")
            plt.close()
            continue

        # Broken-axis log-log version (x=0 gets its own tiny panel)
        fig, (ax0, ax1) = plt.subplots(
            1,
            2,
            sharey=True,
            figsize=(5.4, 4.8),
            gridspec_kw={"width_ratios": [0.35, 4.65]},
        )

        x_all = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
        x_all = x_all[np.isfinite(x_all)]
        x_pos = x_all[x_all > 0]
        min_pos = float(np.min(x_pos)) if x_pos.size else 1.0
        max_pos = float(np.max(x_pos)) if x_pos.size else 10.0

        # Plot Morgan first, then Abe, so Abe sits on top.
        for ds, marker, z in (("Morgan", "s", 2), ("Abe", "o", 3)):
            sub = df[df["dataset"].str.lower() == ds.lower()].copy()
            sub = sub[np.isfinite(sub["mode_mean_um"]) &
                      np.isfinite(sub[x_col])]
            if sub.empty:
                continue

            clr = ds_color.get(ds, None)

            sub0 = sub[sub[x_col] <= 0]
            sub1 = sub[sub[x_col] > 0]

            base_lw = float(plt.rcParams.get("lines.linewidth", 2.0))
            base_ms = float(plt.rcParams.get("lines.markersize", 6.0))
            lw = 0.7 * base_lw
            ms = 0.85 * base_ms

            if not sub1.empty:
                x1 = sub1[x_col]
                if x_col == "relaxation_s":
                    x1 = x1 * 1000.0  # s -> ms
                ax1.errorbar(
                    x1,
                    sub1["mode_mean_um"],
                    yerr=sub1["mode_std_um"],
                    fmt=marker,
                    color=clr,
                    ecolor=clr,
                    capsize=3,
                    capthick=lw,
                    elinewidth=lw,
                    linewidth=lw,
                    markersize=ms,
                    zorder=z,
                )

            if not sub0.empty:
                x0 = sub0[x_col]
                if x_col == "relaxation_s":
                    x0 = x0 * 1000.0  # s -> ms
                ax0.errorbar(
                    x0,
                    sub0["mode_mean_um"],
                    yerr=sub0["mode_std_um"],
                    fmt=marker,
                    color=clr,
                    ecolor=clr,
                    capsize=3,
                    capthick=lw,
                    elinewidth=lw,
                    linewidth=lw,
                    markersize=ms,
                    zorder=z,
                )

        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax0.set_yscale("log")

        frame_lw = float(plt.rcParams.get("axes.linewidth", 1.0))

        ax0.set_xticks([0.0])
        ax0.set_xticklabels(["0"])
        # Keep y minor ticks on the left panel (log y); only disable x minors here.
        ax0.xaxis.set_minor_locator(NullLocator())

        # Right panel is log-x: keep minor ticks, but do not label them.
        ax1.xaxis.set_minor_formatter(NullFormatter())

        if x_col == "relaxation_s":
            ax1.set_xlim(0.21, 80.0)
            ax1.set_xticks([1.0, 10.0])
            ax1.xaxis.set_major_formatter(
                FuncFormatter(lambda v, pos: f"{v:g}"))
            ax1.xaxis.set_minor_formatter(NullFormatter())
            append_unit_to_last_ticklabel(
                ax1, axis="x", unit="ms", fmt="{x:g}")
            ax1.set_ylim(1.0, 1000)
            ax0.set_xlim(-0.02, 0.02)
            ax0.set_ylim(1.0, 1000)
            x_label_used = "Relaxation time"
        else:
            left_halfwidth = min(0.5, max(min_pos * 0.2, 1e-12))
            ax0.set_xlim(-left_halfwidth, left_halfwidth)
            ax1.set_xlim(0.91, 150)
            ax1.xaxis.set_major_formatter(
                FuncFormatter(lambda v, pos: f"{v:g}"))
            ax1.xaxis.set_minor_formatter(NullFormatter())
            ax1.set_ylim(1.0, 1000)
            ax0.set_ylim(1.0, 1000)
            x_label_used = "Deborah number"

        # ax0.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:g}"))
        # ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:g}"))

        # Grid:
        # - Always keep y-gridlines.
        # - Hide only the specific unwanted x-gridlines (x=0 and x=0.2ms).
        ax0.grid(which="major", axis="y")
        ax1.grid(which="major")
        if x_col == "relaxation_s":
            # Force grid artists to be realized so get_xdata() is reliable.
            fig.canvas.draw()
            for gl in ax1.get_xgridlines():
                xs = gl.get_xdata()
                if len(xs) and abs(float(xs[0]) - 0.2) < 1e-6:
                    gl.set_visible(False)

        # Match gridline + tick widths to the spine width.
        for gl in ax0.get_xgridlines() + ax0.get_ygridlines():
            gl.set_linewidth(frame_lw)
        for gl in ax1.get_xgridlines() + ax1.get_ygridlines():
            gl.set_linewidth(frame_lw)
        ax0.tick_params(axis="both", which="both", width=frame_lw)
        ax1.tick_params(axis="x", which="both", width=frame_lw)

        ax0.spines["right"].set_visible(False)
        ax1.spines["left"].set_visible(False)
        ax0.yaxis.tick_left()
        ax1.tick_params(axis="y", which="both", left=False,
                        right=False, labelleft=False, labelright=False)

        # White plot background (inside axes) + transparent outer figure.
        # Also fill the gap between broken axes with white.
        fig.patch.set_alpha(0.0)
        ax0.set_facecolor("white")
        ax1.set_facecolor("white")

        # We'll place a centered xlabel at the same height as a normal axis xlabel.
        # Strategy: temporarily set an ax1 xlabel, measure its figure-space y,
        # then replace it with a centered fig.text.
        ax1.set_xlabel(x_label_used)
        ax0.set_ylabel(r"Mode diameter (μm)")

        fig.tight_layout()
        # Broken-axis gap: keep it a touch smaller (adjust here if needed).
        wspace = 0.1
        fig.subplots_adjust(wspace=wspace)

        from matplotlib.patches import Rectangle

        b0 = ax0.get_position()
        b1 = ax1.get_position()
        x0f = float(b0.x0)
        x1f = float(b1.x1)
        y0f = float(min(b0.y0, b1.y0))
        y1f = float(max(b0.y1, b1.y1))

        fig.add_artist(
            Rectangle(
                (x0f, y0f),
                x1f - x0f,
                y1f - y0f,
                transform=fig.transFigure,
                facecolor="white",
                edgecolor="none",
                zorder=-1,
            )
        )

        # Center the xlabel between the two panels, but keep the *height*
        # identical to a normal axis xlabel.
        fig.canvas.draw()
        label = ax1.xaxis.label
        pos = label.get_position()
        trans = label.get_transform()
        x_disp, y_disp = trans.transform(pos)
        _, y_label_fig = fig.transFigure.inverted().transform((x_disp, y_disp))
        ax1.set_xlabel("")
        fig.text(
            0.5 * (x0f + x1f),
            float(y_label_fig),
            x_label_used,
            ha="center",
            va=label.get_verticalalignment(),
        )

        # Nudge the break marks slightly inward so they overlap the spine edges.
        add_broken_xaxis_marks(
            fig,
            ax0,
            ax1,
            length_points=12.0,
            inset_points=0.5 * frame_lw,
            angle_deg=65,
        )
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
        plt.close(fig)


for loglog in (False, True):
    _plot_mode_vs("Deborah", x_label="Deborah", loglog=loglog)
    _plot_mode_vs("relaxation_s", x_label="Relaxation time (s)", loglog=loglog)


print("Export complete -> new_analysis/data")
print("Mode plots complete -> new_analysis/data/plots")
