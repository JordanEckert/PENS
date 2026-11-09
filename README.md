# PENS: Persistence-Edited Net Selection

PENS is a supervised prototype-selection method for nearest-neighbor classification. It returns a subset of the training rows for each class together with a radius $\varepsilon_c$ for each class, and the radius bounds how far the persistence diagrams of the selected rows can move away from those of the edited class. This repository holds the implementation (the Python package `pens`, distributed as `pens-tda`), the benchmark scripts and every result file behind the paper

> J. Eckert, H. Schenck and E. Ceyhan, *Prototype Selection for Nearest-Neighbor Classification with a Class-Wise Persistence Certificate*, submitted (2026).

PENS works in three stages.

1. **Edit.** Build the single-linkage dendrogram of all training points pooled across classes; its merge heights are the degree-0 persistence of the Rips filtration, and no higher degree is used. Descend from the root and accept the first node whose class-balanced majority fraction is at least `purity` or whose size is at most `s_min`. The balanced majority class claims each accepted node and the minority members of the node are removed. The stage removes label noise, since a mislabeled point sits in a node claimed by its neighbours' class, and protects minority classes, since the majority is taken after weighting each class by the inverse of its size. The points that remain form the kept set $K_c$ of each class. A class the edit empties receives a single guard point, the row of the class nearest its mean.
2. **Net.** Set one shared radius $\varepsilon = \lambda \times$ the median single-linkage merge height of all kept points (`eps_scale` is $\lambda$), and cover each $K_c$ by greedy farthest-point sampling until its covering radius is at most $\varepsilon$. These points are the core $Q_c$, an $\varepsilon$-net of $K_c$. A class with a single kept point has $\varepsilon_c = 0$.
3. **Completion and pruning.** This stage is aimed at accuracy. Add kept points at the class boundaries until every kept point has a nearest prototype of its own class (`completion="consistent"`), or only where an added point corrects other kept points (`completion="gain"`), and then prune added points whose removal does not lower the weighted number of correctly classified kept points. Core points are never removed.

## The certificate

The prototypes $P_c$ contain the $\varepsilon_c$-net $Q_c$ of $K_c$ and lie in $K_c$, so $d_H(P_c, K_c) \le \varepsilon_c$ and, by stability of Rips persistence, in every homology degree

$$
d_B\bigl(\mathrm{Dgm}(P_c), \mathrm{Dgm}(K_c)\bigr) \le 2\,\varepsilon_c \qquad \text{(Corollary 1).}
$$

Against the observed class $X_c$ the same argument gives $d_B(\mathrm{Dgm}(P_c), \mathrm{Dgm}(X_c)) \le 2(\varepsilon_c + \eta_c)$, where $\eta_c = d_H(K_c, X_c)$ is the edit cost (Theorem 3, `end_to_end_bound`). Both bounds concern the training sample, not the population. No bound is claimed for a class the edit empties (a guard class). `verify_certificate` checks the two inequalities of Corollary 1 numerically with Ripser and GUDHI, `edit_cost` returns $\eta_c$, and `end_to_end_bound` returns the bound of Theorem 3.

## Tie rules

Ties are broken deterministically, as stated in Section 4 of the paper.

- The purity test $\mathrm{pur}(v) \ge \pi$ is evaluated in exact rational arithmetic, so a node of purity exactly $\pi$ is accepted.
- The seed of the farthest-point cover is the kept point nearest the class mean; a tie goes to the smaller row index. Each later point is the kept point farthest from the points chosen so far; a tie goes to the smaller row index. A point at distance exactly $\varepsilon$ counts as covered.
- In the edit, a node whose classes tie after balancing goes to the globally rarer class, then to the smaller class index.
- In completion, a kept point whose nearest prototypes of two classes tie is served by the prototype added first; the core is inserted in increasing row order. In pruning, ties go to the smaller row index.
- The 1-NN prediction on an exact tie depends on the tie rule of the classifier. `KNeighborsClassifier` breaks ties by training order, which is the smaller row index when the prototypes are passed in row order, and that can differ from the rule completion uses.

The fit is deterministic on a given machine (`tests/test_selector.py::test_deterministic`), but distances are computed in floating point. On datasets with many exactly tied distances (duplicate rows or quantised features), two distances that are equal in exact arithmetic can differ in the last bit on another platform or BLAS build, so a tie is resolved as a strict inequality there, which changes which of two equivalent rows is selected; in our reruns this affected a few gain-completion fits (for example on vowel) and no selected configuration. `benchmarks/real/run_components.py` checks every refit at the selected configuration against `real.csv` and records the outcome in `results/components_refit.csv` (no fold differed in the reported run).

## Install

The distribution is called `pens-tda` (the name `pens` on PyPI belongs to an unrelated project); the import name is `pens`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,certificate]"      # the package, pytest, and ripser + gudhi for verify_certificate
```

To also run the benchmarks, install `requirements.txt` instead, which adds scikit-learn, pandas, matplotlib, pillow and six:

```bash
python -m pip install -r requirements.txt
```

Or install only what you need:

```bash
python -m pip install .                    # numpy, scipy
python -m pip install ".[certificate]"     # adds ripser and gudhi
```

The tests run with `python -m pytest` (313 tests, a few seconds). They include the Lemma 2 invariants on every fixture under every completion and weighting option (`tests/test_invariants.py`), the determinism of the fit, and a reduced version of the Remark 1 experiment (`tests/test_remark1.py`).

## Usage

```python
import numpy as np
from pens import PENS, edit_cost, end_to_end_bound, verify_certificate

rng = np.random.default_rng(0)
X = np.vstack([rng.normal(0, 1, (100, 2)), rng.normal(3, 1, (100, 2))])
y = np.repeat([0, 1], 100)

model = PENS().fit(X, y)
X_red, y_red = model.transform(X, y)
print(model.prototypes_, model.radii_)
print(verify_certificate(X, model.kept_, model.prototypes_, model.radii_))
print(edit_cost(X, y, model.kept_))
print(end_to_end_bound(X, y, model.kept_, model.radii_))
```

A fuller script is in [examples/basic_usage.py](examples/basic_usage.py).

## Hyperparameters

The code names, symbols and defaults are those of Table 1 of the paper.

| Name | Symbol | Default | Meaning |
|---|---|---|---|
| `purity` | $\pi$ | `0.9` | Weighted majority-class fraction at which a dendrogram node is accepted, in (0, 1]. |
| `eps_scale` | $\lambda$ | `4.0` | Net radius in units of the median single-linkage merge height of the kept points; larger gives a smaller core and a looser bound. |
| `s_min` | $s_{\min}$ | `5` | Nodes of at most this size are accepted regardless of purity. |
| `min_keep` | $m_{\mathrm{keep}}$ | `1` | Accepted clusters keeping fewer than this many points are dropped. |
| `class_weight` | $w_c$ | `"balanced"` | Weight every class by the inverse of its size in the purity test, the majority claim, completion and pruning; `None` counts points. |
| `completion` | – | `"consistent"` | `"consistent"`, `"gain"` or `None`; see above. |
| `gain` | $g$ | `1.0` | Minimum weighted number of other kept points a point must correct to be added by `completion="gain"`. |
| `prune` | – | `True` | Remove completion points whose removal does not lower the weighted number of correctly classified kept points. |
| `metric` | $\rho$ | `"euclidean"` | Any SciPy `pdist` metric; `manhattan`/`l1` and `l2` are also accepted. |
| `metric_params` | – | `None` | Extra keyword arguments for the metric, e.g. `{"p": 3}` for `minkowski`. |

In the evaluation, `eps_scale` in {1, 2, 4, 8, 16} and `completion` in {`"gain"`, `"consistent"`} are tuned by inner cross-validation for the trade-off between balanced accuracy and reduction (Table 6 of the paper).

Fitted attributes: `classes_`, `prototypes_` ($P_c$), `prototype_indices_`, `kept_` ($K_c$), `core_` ($Q_c$), `radii_` ($\varepsilon_c$), `clusters_`.

## Layout

```
PENS/
  src/pens/
    __init__.py       PENS, bottleneck_bound, edit_cost, end_to_end_bound, hausdorff_distance, verify_certificate
    _metrics.py       metric names and distance helpers
    dendrogram.py     Stage 1: supervised dendrogram descent and edit (exact purity test)
    net.py            Stage 2: shared radius and farthest-point cover
    completion.py     Stage 3: consistent and gain completion, pruning
    selector.py       the PENS estimator
    certificate.py    Hausdorff distance, edit cost, Corollary 1 check, Theorem 3 bound
  tests/              pytest suite
  examples/basic_usage.py
  benchmarks/
    real/             real-data arm: 18 datasets (data/), 11 competitors (baselines.py, vendor/), nested
                      cross-validation, statistical tests, component comparison, certificate on real data,
                      sensitivity, the figure across eps_scale; results/ and figures/
    synthetic/        synthetic arm: 18 generators, certificate sweep, matched-size comparison, noise and
                      imbalance ablations, Remark 1 experiment, Figures 1 and 2; results/ and figures/
    scale/            scalability arm: fit time and memory up to 15000 training rows; data/ and results/
```

Every function in the package and in the benchmark scripts carries a docstring stating its inputs and outputs, and each script's `main()` docstring states how to run it and which files it reads and writes; there is no further documentation beyond this file. `benchmarks/real/vendor/` holds third-party code (SPOTgreedy from InterpretML, IB3 from waashk/instanceselection) and is left as distributed.

## Reproducing the paper

The committed result files are the ones the paper reports. Each table and figure of the paper traces to a script and a committed output as follows; the derived summaries (`*.txt`, `real.csv`, `per_dataset.csv`, `components.csv`, `lambda_tradeoff.csv`) regenerate from the committed CSVs without refitting.

| Paper | Script | Committed output |
|---|---|---|
| Table 1 (defaults) | `src/pens/selector.py` | – |
| Figure 1 (schematic) | `benchmarks/synthetic/plot_schematic.py` | `synthetic/figures/schematic.pdf`, `synthetic/results/schematic.txt` |
| Figure 2 (diagrams) | `benchmarks/synthetic/plot_diagrams.py` | `synthetic/figures/diagrams.pdf`, `synthetic/results/diagrams.txt` |
| Remark 1 counts | `benchmarks/synthetic/run_remark1.py` (`tests/test_remark1.py` is the reduced version) | `synthetic/results/remark1.txt` |
| Section 6.1, Q1, Q2 and Table 2 | `benchmarks/synthetic/run_certificate.py`, `summarize.py` | `synthetic/results/certificate.csv`, `summary.txt` |
| Section 6.1, Q3, Tables 3 and 4 | `benchmarks/synthetic/run_matched.py` | `synthetic/results/matched.csv`, `matched.txt` |
| Section 6.1, Q4 (noise and imbalance) | `benchmarks/synthetic/run_ablation.py`, `summarize.py` | `synthetic/results/ablation.csv`, `summary.txt` |
| Table 5 (datasets) and Table 6 (grids) | `benchmarks/real/common.py`, `real/data/` | – |
| Tables 7 and 8, Figure 4 | `benchmarks/real/run_real.py`, `select_real.py`, `stats.py` | `real/results/real.csv`, `per_dataset.csv`, `summary.txt`, `stats.txt`, `cd_bal_acc_ms.pdf`, `cd_effectiveness_ms.pdf` |
| Table 9 (component comparison) | `benchmarks/real/run_components.py`, `select_components.py`, `fps.py` | `real/results/components.csv`, `components_summary.txt` |
| Section 6.3 (certificate on real data) | `benchmarks/real/certify_real.py` | `real/results/certificate_real.csv`, `certificate_real.txt` |
| Figure 3 (trade-off across λ) | `benchmarks/real/plot_lambda.py` | `real/figures/lambda_tradeoff.pdf`, `real/results/lambda_tradeoff.csv` |
| Section 6.4 (sensitivity) | `benchmarks/real/sensitivity_real.py` | `real/results/sensitivity_real.csv`, `sensitivity_real.txt` |
| Section 6.5 (scalability) | `benchmarks/scale/run_scale.py` | `scale/results/scale.csv`, `summary.txt` |
| Competitor implementations | `benchmarks/real/verify_baselines.py` | `real/results/verify_*.txt` |

To rerun everything, install `requirements.txt` and run the scripts in the order below from their own directories. Run every Python command with `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, one thread per process, as in the reported runs. `NPROC` (default 4) sets the number of worker processes where a script forks. Fit times in `benchmarks/real/results/summary.txt` and in `benchmarks/scale/results/` come from a Mac Mini (Apple M2, 8 GB) and are comparable only with runs on that machine; no other output depends on the machine beyond the tie caveat above.

```bash
# Real-data arm (benchmarks/real). Each run_*.py writes one shard per dataset; the select_*.py scripts
# concatenate the shards into the committed raw files and derive the tables.
printf '%s\n' twonorm ring satimage texture page_blocks phoneme waveform_21 segmentation \
  wine_quality_red yeast vowel vehicle wdbc ionosphere bupa ecoli new_thyroid sonar \
  | xargs -n 1 -P 4 python run_real.py       # results/shards/<dataset>.csv
python select_real.py                        # real_raw.csv, real.csv, per_dataset.csv, summary.txt
python stats.py                              # stats.txt, cd_<metric>.pdf, cd_bal_acc_ms.pdf, cd_effectiveness_ms.pdf
python certify_real.py                       # certificate_real.csv, certificate_real.txt (needs real.csv)
python plot_lambda.py                        # lambda_tradeoff.csv, figures/lambda_tradeoff.pdf (needs real_raw.csv,
                                             #   real.csv, certificate_real.csv)
printf '%s\n' twonorm ring satimage texture page_blocks phoneme waveform_21 segmentation \
  wine_quality_red yeast vowel vehicle wdbc ionosphere bupa ecoli new_thyroid sonar \
  | xargs -n 1 -P 4 python run_components.py # results/components_shards/ (needs real.csv; STRICT=1 stops on a
                                             #   refit mismatch, STRICT=0 records it)
python select_components.py                  # components_raw.csv, components.csv, components_refit.csv,
                                             #   components_summary.txt (needs real.csv, certificate_real.csv)
python sensitivity_real.py                   # sensitivity_real.csv, sensitivity_real.txt (needs real.csv)
python verify_baselines.py keel              # optional checks of the competitors against their reference code;
python verify_baselines.py drop3             #   keel needs git, javac, java and network access, ps needs R with
python verify_baselines.py lsbo              #   the package class; the reference code is fetched into .cache/
python verify_baselines.py ps
python verify_baselines.py gkn

# Synthetic arm (benchmarks/synthetic)
python run_certificate.py                    # certificate.csv (python run_certificate.py sanity prints the longest
                                             #   bars of every class at seed 0)
python run_ablation.py                       # ablation.csv
python summarize.py                          # summary.txt (needs certificate.csv and ablation.csv)
python run_matched.py                        # matched.csv, matched.txt (needs certificate.csv;
                                             #   python run_matched.py summary rewrites matched.txt only)
python plot_schematic.py                     # figures/schematic.pdf, results/schematic.txt
python plot_diagrams.py                      # figures/diagrams.pdf, results/diagrams.txt (needs certificate.csv)
python run_remark1.py                        # results/remark1.txt

# Scalability arm (benchmarks/scale)
python run_scale.py                          # scale.csv, summary.txt (MAX_N caps the training sizes on machines
                                             #   with less than about 8 GB of free memory)
```

`certify_real.py summary` and `run_matched.py summary` rewrite their text summaries from the committed CSVs without refitting. Regenerated summaries can differ from the committed ones in the last printed digit of a few cells (values that sit exactly on a rounding boundary are formatted differently across pandas and numpy versions); none of those cells is quoted in the paper.

## Citation

If you use PENS, please cite the paper above and the software using the metadata in [CITATION.cff](CITATION.cff) (version 1.0.0).

## License

MIT. See [LICENSE](LICENSE).
