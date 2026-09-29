# Paper figures

Two scripts, both rendering into `figures/*.pdf` at 600 DPI with Type-42 fonts:

- **`accuracy_memory_figures.py`** — the Figure 5 panels (grouped-bar accuracy +
  memory-savings), from the aggregated tables in `data/`.
- **`pareto_figures.py`** — the accuracy-vs-memory trade-off, one panel per model
  pool, straight from each pipeline run's `results/<set>/{sweep,report}.csv`.

# Figure 5 (accuracy + memory-savings) plots

Regenerates the paper's Figure 5 panels: per-configuration grouped-bar accuracy
(SANDHI vs No-merge / Full-merge / LoRA) plus the memory-savings panel.

## Run (in the reference container — no extra deps)
```bash
CODE=/path/to/this/repo
docker run --rm --user "$(id -u):$(id -g)" -e USER="$(id -un)" -e HOME=/tmp \
  -v "$CODE":/workspace/merge_tools -w /workspace/merge_tools/plots/source \
  merge-tools:reference python accuracy_memory_figures.py
```
Outputs land in `plots/figures/*.pdf` (600 DPI, Type-42 fonts). No
dependencies beyond the image's pandas/matplotlib are required.

## Config → figure map
| script config | paper panel | pipeline set |
|---|---|---|
| `3-Qwen-32B` | **Fig 5a** | fig5a (`qwen32b3`) |
| `5-llama` | **Fig 5b** | fig5b (`llama5`) |
| `7-5-llama-2-DS` | **Fig 5c** | fig5c (`llama5`+`deepseek2`) |
| `12-model` | **Fig 5d** | fig5d (all 12) |

The script renders exactly these four panels (plus the legend) — the paper's
only accuracy/memory bar charts. The Figure 6 pools have no bar-chart panel:
their memory numbers are the pipeline's (`results/FIGURE_COMPOSITIONS.md` and
`results/fig6*/report.csv`), and their serving plots are rendered by the
harness in `../../serving/` (recorded in `../../serving/results/`).

## Files
- `source/accuracy_memory_figures.py` — the plotting script.
- `source/pareto_figures.py` — the accuracy-vs-memory panels; reads `results/`
  directly, so it needs no table in `data/`.
- `source/sandhi_colors.py`, `source/paper.mplstyle` — colors + publication style.
- `data/all_models_final.csv` — SANDHI accuracy deltas + memory saved per
  (config, model); `data/memory_savings.csv`, `data/vllm-no-merge.csv` — memory
  and no-merge baselines.
- `data/full_merge/*.csv` — Full-merge (multi-slerp) baselines; `data/lora/*.csv`
  — LoRA-adapter baselines.

## Data sources and operating points
These figures reproduce the paper's *rendered* Figure 5, so the two tables
deliberately sit at the points the paper reports:

- `data/memory_savings.csv` holds the paper's reported memory numbers at the
  paper's operating point and denominators (e.g. 5-llama 26.4 GB / 35.2%,
  3-Qwen-32B 92.6 GB / 49.8% of the paper's 186 GB total).
- `data/all_models_final.csv` holds the artifact runs' per-model accuracy
  deltas (regenerated from each set's `analysis/<set>/report.csv` by
  `scripts/build_plot_data.py`), measured at the artifact's chosen operating
  point — which can free *more* memory than the paper's bar (e.g. 5-llama
  33.9 GB at Cpm vs the paper's 26.4 GB).

The artifact's own memory numbers at every operating point are in each set's
`report.csv` and `results/FIGURE_COMPOSITIONS.md`; the denominator
reconciliation for Fig 5a is in `../README.md` § Recorded references.

# Accuracy-vs-memory (Pareto) plots

One panel per model pool, replacing the diagnostic `pareto.png` /
`pareto_models.png` that `build_operating_points.py` drops next to each run.

```bash
cd source && python pareto_figures.py     # or via the reference container, as above
```

Reads `results/<set>/sweep.csv` (the full cutoff sweep) and
`results/<set>/report.csv` (the named operating points), and writes
`figures/pareto_<set>.pdf` plus a standalone `figures/pareto_legend.pdf`.

Axes: **x = memory savings in GB** (absolute memory freed; the pool's full size
is in each panel title). `X_UNITS` at the top of the script controls this — the
first entry is primary and takes the bare filename, so `('gb', 'pct')` also
emits `pareto_<set>_pct.pdf` with the normalised axis. **y = worst accuracy
drop**: the worst model's change against its own unmerged baseline in
percentage points, labeled as a drop (positive = worse, increasing downward;
an improvement reads as a negative drop), so 0 means "no model lost anything"
and the axis is fixed to ±10 via `ACC_YLIM`. Every panel is also emitted as a
`pareto_<set>_avg.pdf` companion with the **mean** per-model drop on y (the
sweep's `mean_drop` column; for composed pools the model-count-weighted mean
of atom means), frontier recomputed on that metric. The mean panels draw the
dashed ρ line but no shaded band: the acceptance rule is per-model, so a mean
inside the tolerance proves nothing — compare fig5b, where the mean view sits
at or above 0 across the whole sweep while the worst view spends most of it
below −1, the masking the panels exist to expose. The frontier is dark grey
(`FRONTIER_COLOR`, the structured edge of the lighter scatter), the starred
operating point dark red (`STAR_COLOR`), and the dashed red **accuracy bound
ρ** is `DROP_TOLERANCE`, with the rejected region shaded below it on the
worst-drop panels. The frontier line is anchored at the unmerged
reference (`report.csv` point `A`: 0 saved, every model at its baseline) so it
starts from the origin — the sweep itself has no such point, since its
shallowest cutoff (0) already applies each model's first accepted merge.

| script pool | models | paper panel |
|---|---|---|
| `fig5a` | 3× Qwen3-32B | Fig 5a |
| `fig5b` | 5× Llama-3.1-8B | Fig 5b |
| `llama5_deepseek2` | 5× Llama + 2× DeepSeek | Fig 5c |
| `deepseek2` | 2× DeepSeek-7B | Fig 6a |
| `qwen25_2` | 2× Qwen2.5-7B | Fig 6b |

`llama5` is skipped — its `sweep.csv` / `report.csv` are byte-identical to
`fig5b`. The 9- and 12-model pools have no `sweep.csv` of their own; they are
composed additively from atoms (see `results/FIGURE_COMPOSITIONS.md`), and the
script renders them as **composed panels** (`pareto_9model.pdf` = 5 Llama +
2 DS + 2 Qwen2.5; `pareto_12model.pdf` = those + 3 Qwen3-32B): any combination
of one swept configuration per atom is a real deployable state, its savings is
the atoms' sum and its worst drop the atoms' max, and domination is preserved
under (sum, max), so the composed frontier is exact. The grey cloud is a
deterministic stride-sample of the full combination product; the starred
`Cpm` composes to 26.5% (9-model) and 37.7% (12-model), matching
`FIGURE_COMPOSITIONS.md`.

**Acceptance rule.** A point is admissible only when *every* model stays within
`DROP_TOLERANCE` (2.0 pp) of its own unmerged baseline — the dashed red
user-set threshold. Points below it have some model outside the rule, whatever
the pool average looks like.

> **The `accuracy` and `frontier` columns in `sweep.csv` are NOT used.** They are
> built on the *mean* per-model drop (`run_figures.py` passes
> `--accuracy-metric mean`), which lets one model's gain mask another's loss. On
> `fig5b`, all six mean-frontier points are carried by the truth-judge model
> improving 1.9-5.2 pp while other members degrade -- cutoff 24 scores a mean of
> -1.00 ("accuracy improved") while UltraMedical is 1.25 pp *worse*. The
> mean-frontier also advertises points that break the rule: it reaches 33.5% on
> `llama5_deepseek2` where the deepest uniform cutoff with every model inside 2%
> is 17.2%. This script rebuilds both the axis and the frontier from the
> `worst_drop` column, which `sweep.csv` also carries, so no re-run is needed.
> Switching metric changes the frontier almost completely: only 2 of 8 points
> survive on `fig5a`, 3 of 22 on `llama5_deepseek2`.

**The two series are different search spaces, which is the point of the panel.**
`sweep.csv` sweeps one cutoff applied *uniformly* to every model -- the grey
scatter and the dark-grey frontier. `Cpm` in `report.csv` instead picks a cutoff *per
model* (its `cutoff` field is the literal `pm`), which is what buys the headroom:
on `llama5_deepseek2` per-model cutoffs reach 33.5% saved with the worst model at
-1.80 pp, while the uniform frontier is already past -6 pp by that point and
leaves the admissible region at about 17%.

Each panel stars **`Cpm`** -- the most memory saved with every model still inside
the rule. Change `highlight` in `POOLS` for `Bpm` (the same under a 1 pp
budget), `Kpm` (the knee, which gives up some savings for headroom), or `P` (the
paper's memory target). Note `P` **breaks the 2 pp rule** on two pools: its worst
model is 3.74 pp off on `fig5a` and 5.60 pp off on `llama5_deepseek2`.

| pool | pool size | starred `Cpm` | worst model | uniform cutoffs admissible to |
|---|---|---|---|---|
| `fig5a` | 196.6 GB | 89.6 GB (45.6%) | -1.87 pp | 39.1% |
| `fig5b` | 80.3 GB | 34.8 GB (43.3%) | -1.75 pp | 43.3% |
| `llama5_deepseek2` | 107.9 GB | 36.1 GB (33.5%) | -1.80 pp | 17.2% |
| `deepseek2` | 27.7 GB | 1.2 GB (4.4%) | -1.50 pp | 4.4% |
| `qwen25_2` | 30.5 GB | 0.6 GB (1.8%) | -1.21 pp | 1.8% |
