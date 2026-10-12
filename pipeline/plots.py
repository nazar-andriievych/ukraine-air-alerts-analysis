"""Charts for the model report, saved as PNG files.

- skill_by_horizon.png  how much lower the error is than the 28-day rate, per horizon
- reliability.png       predicted probability vs how often alerts happened (calibration)
- examples.png          probability vs actual alerts for three raions over the last validation week
- importance.png        LightGBM feature importance (share of total gain)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

# Reference palette (light mode), categorical slots in fixed order: one color per forecast.
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLORS = {"persistence": "#2a78d6", "rate_28d": "#eb6834", "hour_of_day": "#1baf7a",
          "logreg": "#eda100", "lightgbm": "#e87ba4"}
LABELS = {"persistence": "Persistence", "rate_28d": "28-day rate", "hour_of_day": "Hour-of-day rate",
          "logreg": "Logistic regression", "lightgbm": "LightGBM"}
ALERT_FILL = "#cde2fb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "legend.frameon": False, "legend.labelcolor": INK_2, "font.size": 10, "lines.linewidth": 2,
})


def _dodge(values: list[float], gap: float) -> list[float]:
    """Spread label positions apart by at least `gap`, keeping their order."""
    order = np.argsort(values)
    placed = np.array(values, dtype=float)[order]
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + gap)
    out = np.empty_like(placed)
    out[order] = placed
    return list(out)


def skill_by_horizon(by_horizon: pd.DataFrame, reference: str, path: Path) -> None:
    """Skill against the reference per horizon; persistence is left out (its skill falls below -0.5)."""
    skill = 1 - by_horizon.drop(columns="persistence").div(by_horizon[reference], axis=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(0, color=AXIS, linewidth=1)
    for name in skill.columns:
        ax.plot(skill.index, skill[name], color=COLORS[name], label=LABELS[name])
    ends = skill.iloc[-1]
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    for name, y in zip(ends.index, _dodge(list(ends), span * 0.05)):
        ax.annotate(LABELS[name], (skill.index[-1], y), xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK_2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set(xlabel="Hours ahead", ylabel=f"Lower error than the {LABELS[reference].lower()}",
           xlim=(1, 24), xticks=[1, 3, 6, 12, 18, 24])
    ax.set_title("Skill against the 28-day rate by horizon, validation (2026-05-18 to 07-12)")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncol=4)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def reliability(scored: pd.DataFrame, forecasts: list[str], path: Path, bins: int = 10) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="Perfect calibration")
    edges = np.linspace(0, 1, bins + 1)
    for name in forecasts:
        bucket = np.clip(np.digitize(scored[name], edges) - 1, 0, bins - 1)
        table = scored.groupby(bucket).agg(predicted=(name, "mean"), observed=("alert", "mean"))
        ax.plot(table["predicted"], table["observed"], color=COLORS[name], marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=LABELS[name])
    ax.set(xlabel="Predicted probability", ylabel="Share of hours with an alert", xlim=(0, 1), ylim=(0, 1))
    ax.set_title("Calibration, validation")
    ax.legend(loc="upper left")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def examples(scored: pd.DataFrame, names: pd.Series, path: Path, forecast: str = "lightgbm",
             horizons: tuple[int, ...] = (1, 12), days: int = 7) -> None:
    """For each target hour: the forecast made h hours ahead, with actual alert hours shaded."""
    target = scored["origin"] + (scored["horizon"] - 1) * pd.Timedelta(hours=1)
    end = target.max().floor("D")
    window = scored[(target >= end - pd.Timedelta(days=days)) & (target < end)].assign(target=target)
    rate = scored.groupby("raion_uid")["alert"].mean()
    picks = [(rate - level).abs().idxmin() for level in (0.1, 0.35, 0.7)]

    fig, axes = plt.subplots(len(picks), 1, figsize=(11, 2.4 * len(picks)), sharex=True)
    shades = {1: COLORS["lightgbm"], 12: "#b8577a"}  # same hue: lighter = 1 h ahead, darker = 12 h ahead
    for ax, uid in zip(axes, picks):
        rows = window[window["raion_uid"] == uid]
        actual = rows[rows["horizon"] == 1].set_index("target")["alert"].sort_index()
        ax.fill_between(actual.index, 0, actual, step="post", color=ALERT_FILL, linewidth=0, label="Alert (actual)")
        for h in horizons:
            line = rows[rows["horizon"] == h].set_index("target")[forecast].sort_index()
            ax.plot(line.index, line, color=shades.get(h, INK_2), linewidth=1.5, label=f"Forecast {h} h ahead")
        ax.set(ylim=(0, 1), yticks=[0, 0.5, 1], ylabel="Probability")
        ax.set_title(f"{names.get(uid, uid)} (alert {rate[uid]:.0%} of validation hours)", fontsize=10)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 1.45), ncol=3)
    fig.suptitle("LightGBM forecasts vs actual alerts, last validation week (UTC)", x=0.125, ha="left",
                 fontweight="bold", color=INK, y=1.04)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def importance(gain: pd.Series, path: Path) -> None:
    share = (gain / gain.sum()).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(share.index, share, color=COLORS["lightgbm"], height=0.6)
    for y, value in enumerate(share):
        ax.annotate(f"{value:.1%}", (value, y), xytext=(4, 0), textcoords="offset points", va="center",
                    color=INK_2, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="Share of total gain")
    ax.set_title("LightGBM: what the model relies on")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def all_charts(scored: pd.DataFrame, by_horizon: pd.DataFrame, gain: pd.Series, names: pd.Series,
               folder: Path) -> None:
    skill_by_horizon(by_horizon, "rate_28d", folder / "skill_by_horizon.png")
    reliability(scored, ["rate_28d", "logreg", "lightgbm"], folder / "reliability.png")
    examples(scored, names, folder / "examples.png")
    importance(gain, folder / "importance.png")
