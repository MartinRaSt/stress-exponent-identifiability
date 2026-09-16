# Project: Sammon mapping revisited - adaptive weighting, a scalable solver
#          and a temporal extension (alpha-Sammon)
# Authors: Martin Radvansky <martin.radvansky@vsb.cz>
#          Martin Radvansky, Jr. <martin.radvansky1@vsb.cz>
# Affiliation: Department of Computer Science, Faculty of Electrical Engineering
#              and Computer Science, VSB - Technical University of Ostrava,
#              17. listopadu 2172/15, 708 00 Ostrava-Poruba, Czech Republic
# Created: 2026-09-14
# License: see the LICENSE file in the repository root
"""
Q1-step3 (reserse/2026-09-14_specifikace_rozsireni_q1.md, B.7 item 6) - box
plots of the primary scenario metric (S1 centroid_lre, S2
cophenetic_pearson, S3 geodesic_stress_si) over R replicates, method x
scenario (one panel per scenario), with a horizontal `oracle_truth` line
(S1/S3: the true embedding; S2: mds_upper_bound_cpcc - see B.2).

Input: results/data/[<mode>/]exp9_metric_fidelity_results.csv (status=='ok').
Output: results/figures/[<mode>/]fig_metric_fidelity_boxes.pdf + .csv.

Run: venv\\python.exe -m src.figures.fig_metric_fidelity_boxes [--quick|--full|--smoke]
or: src\\run_fig_metric_fidelity_boxes.bat [quick|full|smoke]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiments.config_experiments import load_experiments_config, resolve_experiment_config
from src.figures.fig_common import OKABE_ITO, add_quick_arg, parse_fig_mode, require_experiment_csv, save_csv_alongside, save_figure

FIG_NAME = "fig_metric_fidelity_boxes"
BASE_EXPERIMENT_NAME = "exp9_metric_fidelity"

_METRIC_DIR_LABEL = {"min": "lower is better", "max": "higher is better"}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Box plots of the primary metric fidelity metric (Q1-step3) over replicates, method x scenario.")
    add_quick_arg(parser)
    args = parser.parse_args()
    mode = parse_fig_mode(args)

    from src.common.logging_utils import get_logger

    logger = get_logger(FIG_NAME, mode=mode)

    exp_cfg = resolve_experiment_config(BASE_EXPERIMENT_NAME, mode)
    fig_cfg = load_experiments_config()["fig_metric_fidelity_boxes"]
    methods: list[str] = list(fig_cfg["methods"])
    primary_metric: dict[str, str] = exp_cfg["primary_metric"]
    primary_direction: dict[str, str] = exp_cfg["primary_direction"]
    scenarios = list(exp_cfg["active_scenarios"])

    df = require_experiment_csv(BASE_EXPERIMENT_NAME, mode, f"venv\\python.exe -m src.experiments.exp9_metric_fidelity --{mode}")
    ok = df[df["status"] == "ok"]

    fig, axes = plt.subplots(1, len(scenarios), figsize=(3.2 * len(scenarios), 3.2))
    axes = np.atleast_1d(axes)
    csv_rows: list[dict] = []

    for ax, scenario in zip(axes, scenarios):
        metric = primary_metric[scenario]
        direction = primary_direction[scenario]
        sub = ok[ok["scenario"] == scenario]

        box_data, labels, colors = [], [], []
        for k, method in enumerate(methods):
            vals = sub[(sub["method"] == method) & sub[metric].notna()][metric].to_numpy(dtype=np.float64)
            if vals.size == 0:
                continue
            box_data.append(vals)
            labels.append(method)
            colors.append(OKABE_ITO[k % len(OKABE_ITO)])
            for v in vals:
                csv_rows.append({"scenario": scenario, "metric": metric, "method": method, "value": float(v)})

        if not box_data:
            logger.warning("fig_metric_fidelity_boxes: scenario %s has no 'ok' data, the panel remains empty.", scenario)
            continue

        bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        oracle_vals = sub[(sub["method"] == "oracle_truth") & sub[metric].notna()][metric]
        if not oracle_vals.empty:
            oracle_median = float(oracle_vals.median())
            ax.axhline(oracle_median, color="black", linestyle="--", linewidth=0.8, label="oracle_truth")
            csv_rows.append({"scenario": scenario, "metric": metric, "method": "oracle_truth_median_line", "value": oracle_median})

        ax.set_title(f"{scenario}: {metric} ({_METRIC_DIR_LABEL.get(direction, direction)})", fontsize=8)
        ax.tick_params(axis="x", rotation=90, labelsize=6)
        ax.tick_params(axis="y", labelsize=7)

    fig.tight_layout()
    save_figure(fig, FIG_NAME)
    logger.info("Saved figure %s.pdf (mode=%s).", FIG_NAME, mode)

    csv_df = pd.DataFrame(csv_rows)
    save_csv_alongside(csv_df, FIG_NAME)
    logger.info("Saved CSV %s.csv (%d rows).", FIG_NAME, len(csv_df))


if __name__ == "__main__":
    main()
