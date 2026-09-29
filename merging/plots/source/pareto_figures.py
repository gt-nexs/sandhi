"""Accuracy-vs-memory (Pareto) figures, one per model pool.

Renders the publication version of each pool's memory/accuracy trade-off from
the pipeline's own outputs -- `results/<set>/sweep.csv` (every operating point,
with the `frontier` flag) and `results/<set>/report.csv` (the named operating
points). Replaces the diagnostic `pareto.png` / `pareto_models.png` that
`build_operating_points.py` emits during a run.

Both axes follow the acceptance rule: a point is admissible only when EVERY
model stays within DROP_TOLERANCE pp of its own unmerged baseline, so the y axis
is the WORST per-model accuracy delta and the shaded band is the rejected
region. sweep.csv's own `accuracy`/`frontier` columns are built on the *mean*
drop instead and are deliberately not used -- see load_sweep().

The two series are also different search spaces: `sweep.csv` sweeps a *single*
cutoff applied uniformly to every model, while the `Bpm`/`Cpm`/`Kpm` rows of
`report.csv` pick a cutoff *per model* (`cutoff == "pm"`). The starred point is
`Cpm`: the most memory saved with every model still inside the rule.

Run (from this directory):
    ~/miniforge3/envs/plots/bin/python pareto_figures.py

Outputs `../figures/pareto_<pool>.pdf` plus `pareto_legend.pdf`, 600 DPI with
Type-42 fonts, per `paper.mplstyle`.
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================

RESULTS_DIR = "../../results"
FIGURES_DIR = "../figures"

# Acceptance rule: an operating point is admissible when EVERY model stays
# within this many percentage points of its own unmerged baseline -- i.e.
# worst-case per-model drop <= DROP_TOLERANCE. Matches every run's
# `drop_tolerance` and the budget B/C/Bpm/Cpm are selected against.
DROP_TOLERANCE = 2.0

# Fixed accuracy axis, shared by every panel so they read on one common scale.
# Set to None to fit each panel to its own data instead.
ACC_YLIM = (-10.0, 10.0)

# The `_avg` companion panels use their own, tighter fixed axis (mean deltas
# stay within a couple of pp) and a log memory axis so the small pools'
# structure isn't crushed against the left edge. The frontier's 0-GB anchor
# (point A) is omitted on a log axis -- zero has no log position.
AVG_ACC_YLIM = (-5.0, 5.0)
AVG_LOG_X = True

# X axis unit(s). The FIRST is primary and takes the bare `pareto_<pool>.pdf`
# filename; any others are rendered alongside with a `_<unit>` suffix. 'gb' is
# absolute memory freed; 'pct' normalises each pool against its own size, which
# makes panels comparable across pools -- add it back as ('gb', 'pct') to get
# both.
X_UNITS = ('gb',)

# build_operating_points.py keeps its internal math in MiB but converts every
# user-facing figure to DECIMAL MB on the way out (`value_MB = value_MiB *
# 1.048576`), so the `savings_mb` CSV column is already decimal MB and the step
# to GB is /1000. Dividing by 1024 here would re-apply a binary conversion to an
# already-decimal quantity and understate every pool by 2.4%.
MB_TO_GB = 1e-3

# Pools to render: results dir -> display config.
#   title     : panel title (None to omit)
#   figsize   : (width, height) inches, matching the acc/mem panel geometry
#   highlight : report.csv operating point to star (Cpm = max savings
#               subject to every model within DROP_TOLERANCE)
# `llama5` is omitted -- its sweep/report are byte-identical to `fig5b`.
POOLS = {
    'fig5a': {
        'title': r'3$\times$ Qwen3-32B',
        'figsize': (5.5, 2.5),
        'highlight': 'Cpm',
    },
    'fig5b': {
        'title': r'5$\times$ Llama-3.1-8B',
        'figsize': (5.5, 2.5),
        'highlight': 'Cpm',
    },
    'llama5_deepseek2': {
        'title': r'5$\times$ Llama + 2$\times$ DeepSeek',
        'figsize': (5.5, 2.5),
        'highlight': 'Cpm',
    },
    'deepseek2': {
        'title': r'2$\times$ DeepSeek-7B',
        'figsize': (5.5, 2.5),
        'highlight': 'Cpm',
    },
    'qwen25_2': {
        'title': r'2$\times$ Qwen2.5-7B',
        'figsize': (5.5, 2.5),
        'highlight': 'Cpm',
    },
}

PLOT_STYLE = {
    'font_size': 14,
    'title_size': 14,
    'annot_size': 11,
    'scatter_color': '#bdbdbd',      # colorbrewer greys
    'scatter_size': 20,
    'frontier_width': 1.8,
    'frontier_marker_size': 4.5,
    'highlight_size': 190,
    'baseline_color': '#636363',
    # The acceptance threshold is a rule, not data -- red (house Sandhi red)
    # is reserved for it alone, so "red = the bound / rejected region".
    'threshold_color': '#e41a1c',
    'grid_color': 'lightgrey',
    'span_color': '#f0f0f0',
}

# Frontier: dark grey, reading as the structured edge of the lighter
# scatter; the operating point gets dark red (ColorBrewer reds) so the
# single most important mark carries the only saturated color.
FRONTIER_COLOR = '#404040'
STAR_COLOR = '#a50f15'

# Draw the tolerance line only when the sweep actually approaches it; pools
# with a narrow accuracy range (2-model pools) would otherwise be squashed.
TOLERANCE_VISIBLE_MARGIN = 1.0

# Composed pools (the 9- and 12-model figures) have no sweep of their own:
# per results/FIGURE_COMPOSITIONS.md they are additive compositions of the
# atom runs. Any combination of one swept configuration per atom is a real
# deployable state -- cross-pool tensors never merge, so pool savings is the
# SUM of the atoms' savings and the worst per-model drop is the MAX across
# atoms. Domination is preserved under (sum, max), so the composed Pareto
# frontier is exact when built from the atoms' frontiers; the grey cloud is a
# deterministic stride-sample of the full combination product.
COMPOSED_POOLS = {
    # Titles stay short -- the "(N GB pool)" suffix is appended by the
    # renderer, and the atom composition is documented in the README table.
    # The paper's 7-5-llama-2-qwen pool (the OTHER 7-model pool -- the
    # llama+deepseek 7 is a real joint run, this one is composed).
    '7model_qwen': {
        'atoms': ['fig5b', 'qwen25_2'],
        'title': '7 models (Llama+Qwen)',
        'figsize': (5.5, 2.5),
        'highlight': 'Cpm',
    },
    '9model': {
        'atoms': ['llama5_deepseek2', 'qwen25_2'],
        'title': '9 models',
        'figsize': (5.5, 2.5),
        'highlight': 'Cpm',
    },
    '12model': {
        'atoms': ['llama5_deepseek2', 'qwen25_2', 'fig5a'],
        'title': '12 models',
        'figsize': (5.5, 2.5),
        'highlight': 'Cpm',
    },
}

# Cap on grey cloud points per composed panel (the full 12-model product is
# ~100k combinations; a strided subset keeps the PDF light while the true
# frontier is guaranteed by the atom-frontier product rows).
COMPOSED_CLOUD_POINTS = 4000

# The `_avg` companion panels are laid out FOUR ACROSS in the paper
# (0.24\textwidth subfigures, ~1.7 in printed), so they render at a compact
# size whose scale-down keeps the effective font at ~8.5 pt -- the same as
# the worst-drop panels at column width. The worst panels keep each pool's
# own `figsize`.
AVG_FIGSIZE = (2.7, 1.6)


# =============================================================================
# Data Loading
# =============================================================================

def load_sweep(pool, metric='worst'):
    """Load a pool's sweep, on the chosen accuracy metric.

    metric='worst' (the primary panels): the WORST per-model drop. The
    `accuracy` / `frontier` columns in sweep.csv are built on the *mean*
    per-model drop (run_figures.py passes `--accuracy-metric mean`), which lets
    one model's gain mask another's loss: on fig5b every mean-frontier point is
    carried by the truth-judge model improving 1.9-5.2 pp while other members
    degrade. The acceptance rule is per-model, so both the axis and the
    frontier are rebuilt here from the `worst_drop` column.

    metric='mean' (the `_avg` companion panels): the mean per-model drop, from
    the same sweep's `mean_drop` column, with the frontier recomputed on it.
    """
    df = pd.read_csv(f"{RESULTS_DIR}/{pool}/sweep.csv")
    df = df.assign(acc=-df['worst_drop' if metric == 'worst' else 'mean_drop'])
    return df.assign(front=frontier_mask(df, 'acc', 'savings_mb'))


def frontier_mask(df, acc_col, mem_col):
    """Non-domination on (accuracy, memory): the same rule as
    build_operating_points.mark_pareto_frontier, recomputed on our metric.

    `p` is dominated when some `q` is >= on both axes and > on at least one, so
    duplicate points do not knock each other off the frontier.
    """
    acc, mem = df[acc_col].to_numpy(), df[mem_col].to_numpy()
    keep = []
    for i in range(len(df)):
        dominated = ((acc >= acc[i]) & (mem >= mem[i])
                     & ((acc > acc[i]) | (mem > mem[i]))).any()
        keep.append(not dominated)
    return keep


def load_operating_points(pool):
    """Load a pool's named operating points (A/B/C/Bpm/Cpm/Kpm/P)."""
    df = pd.read_csv(f"{RESULTS_DIR}/{pool}/report.csv")
    return df.set_index('point')


def pool_total_gb(points):
    """Pool denominator in GB, recovered as savings_mb / (savings_pct / 100).

    This is `sum(full_model_mib)` over the pool members, in decimal MB -- the
    same denominator build_operating_points.py divides by.
    """
    rows = points[points['savings_pct'] > 0]
    row = rows.iloc[0]
    return float(row['savings_mb']) / (float(row['savings_pct']) / 100) * MB_TO_GB


def operating_point_accuracy(points, name, metric='worst'):
    """Aggregate accuracy delta for a named operating point.

    Taken across the per-model `<model>__drop` columns and negated -- the same
    quantity the sweep's axis carries under the same metric, so the star is
    positioned by the axis's own aggregation ('worst' = the acceptance rule;
    'mean' = the `_avg` companion view).
    """
    drops = points.loc[name,
                       [c for c in points.columns if c.endswith('__drop')]
                       ].astype(float)
    return -float(drops.max() if metric == 'worst' else drops.mean())


def compose_pool(atoms, metric='worst'):
    """Build a composed pool's (sweep, points) from its atoms' CSVs.

    Returns frames with the same columns the per-pool loaders produce, so
    create_pareto_figure renders composed panels unchanged. The sweep holds
    the atom-frontier product (which contains the exact composed frontier:
    domination is preserved under sum-savings with either max- or
    weighted-mean-drop aggregation) plus a strided sample of the full
    combination product as the grey cloud; the points frame composes A and the
    per-model-cutoff points (Bpm/Cpm/Kpm) by summing savings and unioning the
    per-model drop columns.

    metric='worst': composed drop = max over atoms of the atom's worst drop.
    metric='mean': composed drop = model-count-weighted mean of atom means.
    """
    import itertools
    import math

    sweeps = [load_sweep(a, metric) for a in atoms]
    reports = [load_operating_points(a) for a in atoms]
    total_mb = sum(pool_total_gb(p) for p in reports) / MB_TO_GB
    n_models = [sum(c.endswith('__drop') for c in s.columns) for s in sweeps]
    n_total = sum(n_models)

    def combo(rows):
        sav = sum(r.savings_mb for r in rows)
        if metric == 'worst':
            return (sav, min(r.acc for r in rows))
        return (sav, sum(n * r.acc for n, r in zip(n_models, rows)) / n_total)

    rows = [combo(rs) for rs in itertools.product(
        *[list(s[s['front']].itertuples()) for s in sweeps])]

    sizes = [len(s) for s in sweeps]
    n_all = math.prod(sizes)
    stride = max(1, n_all // COMPOSED_CLOUD_POINTS)
    for flat in range(0, n_all, stride):
        idxs, rem = [], flat
        for n in sizes:
            idxs.append(rem % n)
            rem //= n
        rows.append(combo([s.iloc[j] for s, j in zip(sweeps, idxs)]))

    sweep = pd.DataFrame(rows, columns=['savings_mb', 'acc'])
    sweep['savings_pct'] = 100 * sweep['savings_mb'] / total_mb
    sweep = sweep.assign(front=frontier_mask(sweep, 'acc', 'savings_mb'))

    pts = {}
    for name in ('A', 'Bpm', 'Cpm', 'Kpm'):
        if not all(name in r.index for r in reports):
            continue
        sav = sum(float(r.loc[name, 'savings_mb']) for r in reports)
        row = {'savings_mb': sav, 'savings_pct': 100 * sav / total_mb}
        for r in reports:
            for c in r.columns:
                if c.endswith('__drop'):
                    row[c] = float(r.loc[name, c])
        pts[name] = row
    points = pd.DataFrame(pts).T
    points.index.name = 'point'
    return sweep, points


def annotation_side(frontier, x, y, x_range):
    """Place the operating-point label clear of the frontier line.

    Prefers above, then below; when the frontier passes on both sides of the
    marker (sparse pools, where the frontier is steep near its right end) the
    label goes beside it instead.
    """
    near = frontier[(frontier['x'] - x).abs() < 0.15 * x_range]
    if not (near['acc'] > y).any():
        return 'above'
    if not (near['acc'] < y).any():
        return 'below'
    return 'left'


# =============================================================================
# Plotting
# =============================================================================

def create_pareto_figure(pool, config, x_unit='pct', sweep=None, points=None,
                         metric='worst'):
    """Render one pool's accuracy-vs-memory figure.

    `x_unit` selects the memory axis: 'pct' of the pool's full size, or 'gb'
    of memory actually freed. Pass preloaded `sweep`/`points` frames to render
    a composed pool (see compose_pool); by default both load from the pool's
    own results directory. metric='mean' renders the `_avg` companion panel
    (mean per-model drop; no threshold band -- the acceptance rule is
    per-model, so a mean within tolerance proves nothing).
    """
    if sweep is None:
        sweep = load_sweep(pool, metric)
    if points is None:
        points = load_operating_points(pool)

    if x_unit == 'gb':
        sweep = sweep.assign(x=sweep['savings_mb'] * MB_TO_GB)
        points = points.assign(x=points['savings_mb'] * MB_TO_GB)
        x_label = 'Memory Savings (GB)'
    else:
        sweep = sweep.assign(x=sweep['savings_pct'])
        points = points.assign(x=points['savings_pct'])
        x_label = 'Memory Savings (%)'
    # The primary unit takes the bare filename; any extra unit is suffixed,
    # as is the mean-drop companion view.
    suffix = '' if x_unit == X_UNITS[0] else f'_{x_unit}'
    if metric == 'mean':
        suffix += '_avg'

    frontier = sweep[sweep['front']].sort_values('x')
    off_frontier = sweep[~sweep['front']]

    highlight = config.get('highlight')
    hl_row = points.loc[highlight] if highlight in points.index else None

    with plt.style.context('paper.mplstyle'):
        plt.rcParams.update({'font.size': PLOT_STYLE['font_size']})
        fig, ax = plt.subplots(figsize=config['figsize'] if metric == 'worst'
                               else AVG_FIGSIZE)

        # -- all operating points explored by the sweep ----------------------
        ax.scatter(off_frontier['x'], off_frontier['acc'],
                   s=PLOT_STYLE['scatter_size'],
                   color=PLOT_STYLE['scatter_color'],
                   edgecolors='none', zorder=2)

        # -- the Pareto frontier ---------------------------------------------
        # Anchored at the unmerged reference (report.csv point A: nothing
        # merged, 0 saved, every model at its baseline), so the trade-off
        # visibly starts from the origin. The sweep itself has no such point:
        # its shallowest cutoff (0) already applies each model's first
        # accepted merge.
        log_x = metric == 'mean' and AVG_LOG_X
        fx, fy = frontier['x'].tolist(), frontier['acc'].tolist()
        if 'A' in points.index and not log_x:
            fx = [float(points.loc['A', 'x'])] + fx
            fy = [operating_point_accuracy(points, 'A')] + fy
        ax.plot(fx, fy,
                '-o', color=FRONTIER_COLOR,
                lw=PLOT_STYLE['frontier_width'],
                ms=PLOT_STYLE['frontier_marker_size'],
                markeredgecolor=FRONTIER_COLOR, zorder=4)

        # -- unmerged reference ----------------------------------------------
        ax.axhline(0, color=PLOT_STYLE['baseline_color'], lw=1.0, ls='--',
                   alpha=0.8, zorder=1)

        # -- acceptance rule ---------------------------------------------------
        # Everything below the line has some model more than DROP_TOLERANCE pp
        # off its baseline, so it is not an admissible operating point at all.
        # Both views draw the dashed rho line (the user-set accuracy bound);
        # only the worst-drop panels shade the region below it -- there the
        # rule literally rejects those points, while on the mean view the
        # per-model rule can't be read off the axis.
        ymin = min(sweep['acc'].min(), 0)
        show_tolerance = (ACC_YLIM is not None
                          or ymin < -DROP_TOLERANCE + TOLERANCE_VISIBLE_MARGIN)
        if show_tolerance:
            if metric == 'worst':
                lo = ACC_YLIM[0] if ACC_YLIM else ymin - abs(ymin)
                ax.axhspan(lo, -DROP_TOLERANCE,
                           color=PLOT_STYLE['span_color'], zorder=0)
            ax.axhline(-DROP_TOLERANCE, color=PLOT_STYLE['threshold_color'],
                       lw=1.3, ls='--', alpha=0.95, zorder=3)

        # -- selected operating point (per-model cutoffs) ---------------------
        ys = [sweep['acc'].min(), sweep['acc'].max(), 0.0]
        if show_tolerance:
            ys.append(-DROP_TOLERANCE)

        if hl_row is not None:
            hl_x = float(hl_row['x'])
            hl_y = operating_point_accuracy(points, highlight, metric)
            ys.append(hl_y)
            ax.scatter([hl_x], [hl_y], marker='*',
                       s=PLOT_STYLE['highlight_size'], color=STAR_COLOR,
                       edgecolors='#000000', linewidths=0.6, zorder=5)

            x_range = max(sweep['x'].max(), hl_x) or 1.0
            # A fixed axis leaves the whole band above the data empty, so the
            # label always clears there; the dodging heuristic is only needed
            # when each panel is fitted tightly to its own range.
            side = ('above' if ACC_YLIM is not None
                    else annotation_side(frontier, hl_x, hl_y, x_range))
            # Operating points sit at the right end of the sweep; right-align
            # the label there so it grows inward instead of past the axis.
            near_edge = hl_x > 0.85 * x_range
            if side == 'left':
                # Down and to the left, clearing the steep frontier tail.
                dx, dy, ha = -16, -20, 'right'
            else:
                # Right-aligned labels extend left, back over the rising part
                # of the frontier, so 'above' needs to clear its local peak.
                dy = 20 if side == 'above' else -22
                dx = 6 if near_edge else 0
                ha = 'right' if near_edge else 'center'
            # Kept short deliberately: a "34.7 GB (43.2%)" label is wide enough
            # to reach back over the rising frontier in every pool. The pool
            # size is in the title and the percentage is one column over in
            # report.csv, so the axis value alone is enough here.
            text = f"{hl_x:.1f} GB" if x_unit == 'gb' else f"{hl_x:.1f}%"
            ax.annotate(text, (hl_x, hl_y),
                        textcoords='offset points', xytext=(dx, dy), ha=ha,
                        fontsize=PLOT_STYLE['annot_size'],
                        color=STAR_COLOR, zorder=6)
            # Reserve room for the label so it never rides into the title.
            span = (max(ys) - min(ys)) or 1.0
            ys.append(hl_y + 0.26 * span if side == 'above'
                      else hl_y - 0.26 * span)

        # -- axes --------------------------------------------------------------
        ax.set_xlabel(x_label)
        if metric == 'worst':
            # The axis is plotted in accuracy-delta space (up = better) but
            # labeled as the DROP, so the tick labels are negated: the
            # threshold reads "2", a model 2 pp under baseline reads as a
            # drop of 2, and an improvement shows as a negative drop above
            # the baseline line. (+ 0 normalizes -0.0 so the baseline tick
            # reads "0", not "-0".)
            ax.set_ylabel('Worst accuracy drop')
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda v, _: f'{-v + 0:g}'))
        else:
            # Delta convention on the avg panels: ticks stay in delta space
            # (negative = worse), matching the Delta in the label.
            ax.set_ylabel('Avg Accuracy $\\Delta$')
        # Titles only on the worst-drop panels; the avg panels are arranged
        # as LaTeX subfigures whose subcaptions carry the pool identity,
        # consistent with the paper's other (title-less) figures.
        if metric == 'worst' and config.get('title'):
            title = config['title']
            if x_unit == 'gb':
                title += f"  ({pool_total_gb(points):.0f} GB pool)"
            ax.set_title(title, fontsize=PLOT_STYLE['title_size'])
        if log_x:
            ax.set_xscale('log')
            ax.set_xlim(left=sweep.loc[sweep['x'] > 0, 'x'].min() * 0.8)
            # Plain decimal ticks at 1-2-5 steps; matplotlib's default minor
            # labels ("3x10^-1") swamp the sub-decade pools.
            ax.xaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
            ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            ax.set_xlim(left=0)
        ylim = AVG_ACC_YLIM if metric == 'mean' else ACC_YLIM
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            pad = 0.10 * ((max(ys) - min(ys)) or 1.0)
            ax.set_ylim(min(ys) - pad, max(ys) + pad)
        ax.grid(True, color=PLOT_STYLE['grid_color'], ls='--', lw=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

        fig.savefig(f'{FIGURES_DIR}/pareto_{pool}{suffix}.pdf')
        print(f"Saved {FIGURES_DIR}/pareto_{pool}{suffix}.pdf")
        plt.close(fig)


def create_legend_figure(metric='worst'):
    """Standalone legend, per the house convention of keeping it off-panel.

    metric='mean' emits pareto_legend_avg.pdf without the threshold entry --
    the avg panels don't draw the (per-model) acceptance rule.
    """
    handles = [
        Line2D([], [], marker='o', ls='none',
               color=PLOT_STYLE['scatter_color'],
               ms=PLOT_STYLE['frontier_marker_size'],
               label='Explored configuration'),
        Line2D([], [], marker='o', color=FRONTIER_COLOR,
               lw=PLOT_STYLE['frontier_width'],
               ms=PLOT_STYLE['frontier_marker_size'],
               label='Pareto frontier'),
        Line2D([], [], marker='*', ls='none', color=STAR_COLOR,
               markeredgecolor='#000000', markeredgewidth=0.6, ms=13,
               label='Sandhi operating point'),
        Line2D([], [], ls='--', color=PLOT_STYLE['baseline_color'], lw=1.0,
               label='Unmerged baseline'),
        Line2D([], [], ls='--', color=PLOT_STYLE['threshold_color'],
               lw=1.3, label='Accuracy bound $\\rho$'),
    ]
    name = 'pareto_legend.pdf' if metric == 'worst' else 'pareto_legend_avg.pdf'

    with plt.style.context('paper.mplstyle'):
        plt.rcParams.update({'font.size': PLOT_STYLE['font_size']})
        fig = plt.figure(figsize=(12.5, 0.6))
        fig.legend(handles=handles, loc='center', ncol=len(handles),
                   frameon=False, handletextpad=0.5, columnspacing=1.6)
        fig.savefig(f'{FIGURES_DIR}/{name}', bbox_inches='tight')
        print(f"Saved {FIGURES_DIR}/{name}")
        plt.close(fig)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    for metric in ('worst', 'mean'):
        for pool, config in POOLS.items():
            print(f"\nGenerating figure for: {pool} ({metric})")
            for x_unit in X_UNITS:
                create_pareto_figure(pool, config, x_unit, metric=metric)

        for pool, config in COMPOSED_POOLS.items():
            print(f"\nGenerating composed figure for: {pool} ({metric}; "
                  f"atoms: {', '.join(config['atoms'])})")
            sweep, points = compose_pool(config['atoms'], metric)
            print(f"  composed Cpm: "
                  f"{points.loc['Cpm', 'savings_mb'] * MB_TO_GB:.1f} GB "
                  f"({points.loc['Cpm', 'savings_pct']:.1f}%)")
            for x_unit in X_UNITS:
                create_pareto_figure(pool, config, x_unit,
                                     sweep=sweep, points=points, metric=metric)

    create_legend_figure()
    create_legend_figure('mean')
    print("\nAll figures generated!")
