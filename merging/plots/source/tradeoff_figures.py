"""Per-model accuracy-vs-memory trade-off, one panel per pool.

The "Sandhi at inference time" view: one curve PER MODEL showing how that
model's accuracy moves as the pool gives up memory, against the user-set
accuracy bound. Companion to `pareto_figures.py`, which shows the same sweep
collapsed to the pool's worst model plus its Pareto frontier.

Curves come from the `<model>__drop` columns of `results/<set>/sweep.csv`,
negated so the y axis is each model's accuracy change against its own unmerged
baseline (0 = unchanged, down = worse). Marker groups are the named operating
points in `results/<set>/report.csv`: at one operating point every model shares
the pool's memory saving, so a group is a vertical cluster -- one marker per
model -- ringed by a dashed ellipse.

Only operating points with a GLOBAL cutoff are marked (`A`/`B`/`C`/`P`), so the
markers land exactly on the curves. The per-model points (`Bpm`/`Cpm`/`Kpm`,
whose `cutoff` field is the literal `pm`) give each model its own cutoff and so
do not lie on a single-cutoff sweep -- see pareto_figures.py for those.

Run (from this directory):
    ~/miniforge3/envs/plots/bin/python tradeoff_figures.py

Outputs `../figures/tradeoff_<pool>.pdf`, 600 DPI, Type-42 fonts.
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

import pandas as pd

# =============================================================================
# Configuration
# =============================================================================

RESULTS_DIR = "../../results"
FIGURES_DIR = "../figures"

# User-set accuracy bound (pp). Every run's `drop_tolerance`.
DROP_TOLERANCE = 2.0

# Memory axis: 'gb' (absolute freed) or 'pct' (share of the pool's full size).
X_UNIT = 'gb'

# See pareto_figures.MB_TO_GB -- `savings_mb` is already decimal MB, so /1000.
MB_TO_GB = 1e-3

ACC_YLIM = (-10.0, 10.0)

POOLS = {
    'fig5a': (r'3$\times$ Qwen3-32B', (7.0, 2.8)),
    'fig5b': (r'5$\times$ Llama-3.1-8B', (7.0, 2.8)),
    'llama5_deepseek2': (r'5$\times$ Llama + 2$\times$ DeepSeek', (7.0, 2.8)),
    'deepseek2': (r'2$\times$ DeepSeek-7B', (7.0, 2.8)),
    'qwen25_2': (r'2$\times$ Qwen2.5-7B', (7.0, 2.8)),
}

# ColorBrewer Set1 without red -- red is reserved for the accuracy bound.
MODEL_COLORS = ['#ff7f00', '#4daf4a', '#984ea3', '#377eb8',
                '#a65628', '#f781bf', '#999999']

THRESHOLD_COLOR = '#e41a1c'
BASELINE_COLOR = '#636363'
ELLIPSE_COLOR = '#252525'

# Long HF repo names -> the domain labels the paper's other figures use.
DISPLAY_NAMES = {
    'Llama-3.1-8B-UltraMedical': 'Medical',
    'Llama-3.1-8B-Instruct-multi-truth-judge': 'Truth',
    'Llama-3.1-Hawkish-8B': 'Finance',
    'Llama-SafetyGuard-Content-Binary': 'Safety',
    'calme-2.3-legalkit-8b': 'Legal',
    'Qwen2.5-Math-7B-Instruct': 'Math',
    'Qwen2.5-Coder-7B-Instruct': 'Coder',
    'deepseek-math-7b-instruct': 'DS-Math',
    'deepseek-coder-7b-instruct-v1.5': 'DS-Coder',
    'Light-IF-32B': 'Light-IF',
    'T-pro-it-2.0': 'T-pro-it',
    'MedGo': 'MedGo',
}

PLOT_STYLE = {
    'font_size': 13,
    'legend_size': 9,
    'line_width': 1.1,
    'line_alpha': 0.5,
    'marker_size': 4.0,
    'cross_width': 1.3,
    'grid_color': 'lightgrey',
    'span_color': '#f0f0f0',
}


# =============================================================================
# Data
# =============================================================================

def display_name(model):
    return DISPLAY_NAMES.get(model, model)


def to_x(series_mb, series_pct):
    """Memory axis values in the configured unit."""
    return series_mb * MB_TO_GB if X_UNIT == 'gb' else series_pct


def load_pool(pool):
    """Return (models, sweep_df, points_df) for one pool.

    The sweep gets an `x` column in the configured memory unit and one
    `acc__<model>` column per model (accuracy delta = -drop).
    """
    sweep = pd.read_csv(f"{RESULTS_DIR}/{pool}/sweep.csv")
    points = pd.read_csv(f"{RESULTS_DIR}/{pool}/report.csv").set_index('point')

    models = [c[:-len('__drop')] for c in sweep.columns if c.endswith('__drop')]
    sweep = sweep.assign(x=to_x(sweep['savings_mb'], sweep['savings_pct']))
    for m in models:
        sweep[f'acc__{m}'] = -sweep[f'{m}__drop']
    return models, sweep.sort_values('x'), points


def merge_cutoffs(pool, model, sweep):
    """Cutoffs at which THIS model had a merge accepted.

    Taken from `results/<set>/steps/<model>_steps.csv`, the MICR search log:
    one row per attempted merge, `decision == 'accepted'` for the ones kept.
    Those step indices are exactly the sweep's cutoffs, so a marker lands on
    the curve at every configuration where this model actually changed.

    Only the step *indices* are used. The `score` column in that file is on the
    MICR M split (run_figures.py passes `--eval_split M`), a different split
    from the sweep's accuracies, so mixing the two scales would be wrong -- the
    y value always comes from the sweep.

    Falls back to the cutoffs where the model's sweep score moves when a pool
    has no `steps/` directory. That is a subset: a merge can be accepted and
    leave the score unchanged, so those panels under-count merges.
    """
    cutoffs = set(sweep['cutoff'].astype(int))
    path = f"{RESULTS_DIR}/{pool}/steps/{model}_steps.csv"
    if os.path.exists(path):
        st = pd.read_csv(path)
        idx = st.loc[st['decision'] == 'accepted', 'step_idx'].astype(int)
        return sorted(cutoffs & set(idx)), True

    col = f'{model}__score'
    ordered = sweep.sort_values('cutoff')
    moved = ordered['cutoff'][ordered[col].diff().fillna(0) != 0].astype(int)
    return sorted(set(moved)), False


# =============================================================================
# Plotting
# =============================================================================

def create_tradeoff_figure(pool, title, figsize):
    models, sweep, points = load_pool(pool)
    by_cutoff = sweep.set_index(sweep['cutoff'].astype(int))
    merges = {m: merge_cutoffs(pool, m, sweep) for m in models}

    with plt.style.context('paper.mplstyle'):
        plt.rcParams.update({'font.size': PLOT_STYLE['font_size']})
        fig, ax = plt.subplots(figsize=figsize)

        # -- rejected region + user-set bound --------------------------------
        lo = ACC_YLIM[0] if ACC_YLIM else sweep[
            [f'acc__{m}' for m in models]].min().min() * 1.5
        ax.axhspan(lo, -DROP_TOLERANCE, color=PLOT_STYLE['span_color'],
                   zorder=0)
        ax.axhline(0, color=BASELINE_COLOR, lw=1.0, zorder=1)
        ax.axhline(-DROP_TOLERANCE, color=THRESHOLD_COLOR, lw=1.4, ls='--',
                   zorder=2)

        # -- one curve per model, marked at every accepted merge --------------
        for i, m in enumerate(models):
            color = MODEL_COLORS[i % len(MODEL_COLORS)]
            # The line only interpolates between measured configurations, so
            # it is drawn back: the markers are the data.
            ax.plot(sweep['x'], sweep[f'acc__{m}'], color=color,
                    lw=PLOT_STYLE['line_width'], label=display_name(m),
                    alpha=PLOT_STYLE['line_alpha'], zorder=3,
                    solid_capstyle='round')

            cuts, _exact = merges[m]
            rows = by_cutoff.loc[cuts]
            # Merges that push this model past the user-set bound are crossed
            # out: same configuration, but not one the rule would accept.
            below = rows[f'acc__{m}'] < -DROP_TOLERANCE
            ax.plot(rows['x'][~below], rows[f'acc__{m}'][~below],
                    marker='o', ls='none', color=color,
                    ms=PLOT_STYLE['marker_size'], mec='#ffffff', mew=0.4,
                    zorder=4)
            ax.plot(rows['x'][below], rows[f'acc__{m}'][below],
                    marker='x', ls='none', color=color,
                    ms=PLOT_STYLE['marker_size'] + 1.4,
                    mew=PLOT_STYLE['cross_width'], zorder=5)

        # -- axes --------------------------------------------------------------
        ax.set_xlabel('Memory Savings (GB)' if X_UNIT == 'gb'
                      else 'Memory Savings (%)')
        ax.set_ylabel('Accuracy $\\Delta$')
        ax.set_title(title, fontsize=PLOT_STYLE['font_size'])
        ax.set_xlim(left=0)
        if ACC_YLIM:
            ax.set_ylim(*ACC_YLIM)
        ax.grid(True, color=PLOT_STYLE['grid_color'], ls='--', lw=0.5,
                alpha=0.7)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

        # Outside the axes: model curves span most of the panel, so any inset
        # box clips the very trajectories the figure is about.
        # Build handles explicitly: the plotted lines are semi-transparent,
        # and a faded legend swatch is much harder to match to its curve.
        model_handles = [
            Line2D([], [], color=MODEL_COLORS[i % len(MODEL_COLORS)],
                   lw=2.0, marker='o', ms=PLOT_STYLE['marker_size'],
                   label=display_name(m))
            for i, m in enumerate(models)]
        ax.legend(handles=model_handles, title='Models',
                  fontsize=PLOT_STYLE['legend_size'],
                  title_fontsize=PLOT_STYLE['legend_size'],
                  loc='center left', bbox_to_anchor=(1.01, 0.5), ncol=1,
                  frameon=False, handlelength=1.6, labelspacing=0.35,
                  alignment='left')

        fig.savefig(f'{FIGURES_DIR}/tradeoff_{pool}.pdf')
        exact = all(e for _c, e in merges.values())
        counts = ', '.join(f'{display_name(m)}={len(merges[m][0])}'
                           for m in models)
        print(f"Saved {FIGURES_DIR}/tradeoff_{pool}.pdf  "
              f"[merges from {'steps.csv' if exact else 'score changes (no '
              'steps/ dir -- UNDERCOUNTS)'}]  {counts}")
        plt.close(fig)


def create_legend_figure():
    """Standalone legend for the non-model elements."""
    handles = [
        Line2D([], [], ls='--', color=THRESHOLD_COLOR, lw=1.4,
               label='User-set accuracy $\\Delta$ bound'),
        Line2D([], [], color=BASELINE_COLOR, lw=1.0, label='Unmerged baseline'),
        Line2D([], [], marker='o', ls='none', color=ELLIPSE_COLOR,
               ms=PLOT_STYLE['marker_size'], label='Accepted merge'),
        Line2D([], [], marker='x', ls='none', color=ELLIPSE_COLOR,
               ms=PLOT_STYLE['marker_size'] + 1.4,
               mew=PLOT_STYLE['cross_width'],
               label=f'Merge past the bound ($<${-DROP_TOLERANCE:.0f})'),
    ]
    with plt.style.context('paper.mplstyle'):
        plt.rcParams.update({'font.size': PLOT_STYLE['font_size']})
        fig = plt.figure(figsize=(9, 0.6))
        fig.legend(handles=handles, loc='center', ncol=len(handles),
                   frameon=False, columnspacing=1.8)
        fig.savefig(f'{FIGURES_DIR}/tradeoff_legend.pdf', bbox_inches='tight')
        print(f"Saved {FIGURES_DIR}/tradeoff_legend.pdf")
        plt.close(fig)


if __name__ == "__main__":
    for pool, (title, figsize) in POOLS.items():
        create_tradeoff_figure(pool, title, figsize)
    create_legend_figure()
    print("\nAll figures generated!")
