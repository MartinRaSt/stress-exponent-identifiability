# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-09
# License: see the LICENSE file in the repository root
"""
An embedding gallery (one panel per successful solver/device combination)
for `src/experiments/exp_sammon_demo.py`. Vector PDF (pdf.fonttype=42),
the Okabe-Ito palette, rasterized scatter layers @300 dpi, a CSV with the
underlying coordinates alongside each figure (spec section 14a).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Okabe-Ito colorblind-safe palette (Okabe & Ito, 2008)
OKABE_ITO = [
    "#000000", "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
]

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def make_gallery_figure(
    dataset_name: str,
    panels: list[dict],
    out_pdf: Path,
    out_csv: Path,
) -> None:
    """Plot a gallery of panels (one per solver/device combination) and
    save the PDF + a CSV with the underlying coordinates.

    `panels`: a list of dicts {'solver':str, 'device':str, 'Y': (n,2) ndarray,
    'labels': (n,) ndarray|None, 'stress': float, 'wall_time_sec': float}.
    """
    n_panels = len(panels)
    if n_panels == 0:
        raise ValueError(f"No successful panel for dataset '{dataset_name}' - nothing to plot.")

    n_cols = min(3, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.3 * n_cols / 2, 5.2 * n_rows / 2), squeeze=False)

    rows_csv = []
    for idx, panel in enumerate(panels):
        ax = axes[idx // n_cols][idx % n_cols]
        Y = panel["Y"]
        labels = panel["labels"]
        if labels is not None:
            uniq = np.unique(labels)
            colors = [OKABE_ITO[i % len(OKABE_ITO)] for i in range(len(uniq))]
            for lab, col in zip(uniq, colors):
                mask = labels == lab
                ax.scatter(Y[mask, 0], Y[mask, 1], s=8, c=col, label=str(lab), rasterized=True, linewidths=0)
        else:
            ax.scatter(Y[:, 0], Y[:, 1], s=8, c=OKABE_ITO[0], rasterized=True, linewidths=0)
        ax.set_title(f"{panel['solver']} ({panel['device']})\nstress={panel['stress']:.4f}, t={panel['wall_time_sec']:.2f}s", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        for i in range(Y.shape[0]):
            rows_csv.append({
                "dataset": dataset_name, "solver": panel["solver"], "device": panel["device"],
                "point_index": i, "y0": Y[i, 0], "y1": Y[i, 1],
                "label": labels[i] if labels is not None else "",
            })

    for idx in range(n_panels, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.suptitle(f"alpha-Sammon solver gallery: {dataset_name}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", dpi=300)
    plt.close(fig)

    pd.DataFrame(rows_csv).to_csv(out_csv, index=False)
