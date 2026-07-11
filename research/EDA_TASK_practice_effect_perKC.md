# EDA Task — `practice_effect_perKC.csv`

**Deliverable to produce:** a Jupyter notebook `eda_practice_effect_perKC.ipynb`
**Owner:** Data/ML
**Goal:** *Deeply understand* this dataset — what every column means, how trustworthy it is, what latent structure it holds, and **how each feature feeds the four Noxed modules** (Diagnostic, Behavioural, Bandit, Progress Tracking). This is a **learning-curve** dataset; the whole point is to quantify *how fast students get better at a knowledge component (KC) with repeated practice* and turn that into model inputs.

> Context: Noxed is an adaptive **science** quiz game for **Egyptian schools**. `practice_effect_perKC.csv` comes from the **XES3G5M proxy dataset** (Chinese elementary math), used to develop the methodology before Egyptian-science data exists. Treat every finding as *"does this method work / what should the pipeline expect,"* **not** as ground truth about Egyptian students.

---

## 0. What this dataset is (state this in the first markdown cell)

- **Grain:** one row per **KC** (`kc_id` 0–864, 865 leaf knowledge components).
- **Columns:** `kc_id` + `exposure_1 … exposure_56` (56 columns).
- **Cell value** `exposure_k[kc]` = **error rate on KC `kc` at the k-th chronological exposure**, aggregated across all students who reached at least `k` exposures of that KC. It is a **right-censored learning curve** read left→right.
- **Semantics from the source doc:** *"we measure how well a student does on KCs with more exposure … we track error rate on the KCs with each exposure."* A **decreasing** curve = practice is working (the practice effect); a **flat/rising** curve = no learning or forgetting/interference.
- **NOT** in this file: per-cell sample size, per-student data, timestamps, question identity. (Those live in `responses.csv`, `practice_effect_perQ.csv`, `chronological_delta.csv`, `kc_metadata.csv`.)

**Ground-truth numbers to reproduce as a sanity check** (assert these in the notebook):
shape `(865, 57)`; values ∈ [0,1]; global mean ≈ 0.2025; non-null count per exposure = 865, 652, 442, 347, 256 … 1; median row coverage = 3 exposures; 213 KCs have only 1 exposure; 93 have ≥10.

---

## 1. Setup & load
- Libraries: `pandas`, `numpy`, `matplotlib` (+ `seaborn` optional), `scipy.optimize.curve_fit`, `scipy.stats`. Set a fixed random seed; define a plotting helper.
- Load `practice_effect_perKC.csv`. Also load `kc_metadata.csv` (join key `kc_id`) and, if present, `kc_tree_structure.md` / `tree_translation.txt` (for module & depth), `practice_effect_perQ.csv`, `chronological_delta.csv`, `question_metadata.csv`.
- Print `df.shape`, `df.dtypes`, `df.head()`, `df.memory_usage`.

## 2. Schema, types & integrity checks
- Confirm exactly 56 `exposure_*` columns, correctly ordered 1→56 (don't trust lexical sort: `exposure_10` must not sort before `exposure_2`). Build `exp_cols` sorted by the integer suffix.
- `kc_id`: unique, contiguous 0–864, integer. **Join test:** every `kc_id` here exists in `kc_metadata.csv` and vice-versa (report any mismatch).
- Value-range assertion: all non-null cells ∈ [0,1]. Flag any exactly-0 or exactly-1 cells (likely tiny samples — see §4).
- Duplicate-row / duplicate-`kc_id` check.

## 3. Reshape to long (the analysis backbone)
- Melt wide → long: `kc_id, exposure (int k), error_rate`. Drop NaNs. This tidy frame drives most plots.
- Derive `n_kc_at_exposure_k` = count of non-null KCs per `k` (the **survivorship curve**) — reused everywhere as a *confidence weight proxy* since we have no true per-cell N.
- Keep both wide (for per-KC curves) and long (for aggregates).

## 4. Missingness = the most important section (censoring & survivorship)
This file's missingness is **structural, not random** — it is the exposure-depth distribution. Get this right or every downstream statistic is biased.
- Plot `n_kc_at_exposure_k` vs `k` (the drop-off 865→652→…→1). Log-y variant.
- Distribution of **per-KC coverage** (how many exposures each KC actually has data for): histogram; report min/median/mean/max = 1/3/4.4/56.
- Quantify the **selection bias**: the KCs that survive to exposure ≥10 are not a random sample — they are the *most-practised* KCs. Compare (a) mean `exposure_1` error of KCs with coverage ≥10 vs coverage =1; (b) their module/topic mix. State the consequence: *late-exposure columns describe a shrinking, self-selected elite of KCs, so an apparent late-curve "improvement" is partly survivorship, not learning.*
- **Guiding questions:** At what `k` does sample size become too small to trust (propose a cutoff, e.g. k where `n_kc < 30`)? Should later columns be down-weighted or truncated?
- **Benefit to us:** defines the **reliability horizon** for any practice-effect feature and warns the Progress-Tracking/BKT layer not to over-fit late-exposure behaviour.

## 5. Univariate distributions
- Distribution of `exposure_1` error across the 865 KCs (this is the **cold-start difficulty proxy** — first-encounter error). Histogram + boxplot; report skew, median, tail of very-hard (→1) and very-easy (→0) KCs.
- Distribution of error at exposure_2, _3, _5, _10 overlaid — watch the mode shift toward 0 (learning) and the variance change.
- Count KCs with `exposure_1 == 0` (trivially easy or tiny sample) and `== 1` (nobody got it right first time).

## 6. Aggregate learning curve
- Mean **and** median error vs exposure `k`, with a band (IQR or bootstrap CI), **annotated with `n_kc` at each k** so the reader sees confidence collapsing.
- Two versions: (a) **naïve** (all KCs available at each k — biased by survivorship); (b) **balanced cohort** (only KCs present at *both* k=1 and k=K, so the curve follows a fixed set) — compare the two to isolate true learning from composition change.
- Report the aggregate practice effect: mean error exp1→exp5 (≈0.224→0.192) and % relative reduction.
- **Guiding question:** is the curve monotonic decreasing, or does it plateau / rebound?

## 7. Per-KC curve shapes (taxonomy)
- Plot a spaghetti sample of ~40 KC curves (faded) + the mean.
- Classify each KC's curve (using its available points) into shapes: **(i) clean decreasing** (learning), **(ii) flat-high** (persistently hard, no learning — candidate for re-teaching/scaffolding), **(iii) flat-low** (already mastered — low training value), **(iv) noisy/non-monotonic** (small sample or interference), **(v) increasing** (forgetting/interference/harder variants introduced later). Report counts per class.
- Compute **exp1→exp2 direction** per KC: 65% decrease, 35% increase — discuss why a third get worse (small N, harder follow-up items, mixed practice).
- **Benefit to us:** the shape class is directly actionable — flat-high KCs trigger the "Weak Prior Knowledge / Knowledge Absence" treatments; flat-low KCs should be *deprioritised* by the bandit; increasing curves flag forgetting → spaced-repetition candidates.

## 8. Engineer practice-effect features (per KC)
For each KC, compute and tabulate (this becomes an exportable feature table):

| Feature | Definition | Primary Noxed use |
|---|---|---|
| `initial_error` | error at exposure_1 | Cold-start **difficulty** prior; BKT `P(L₀)` seed |
| `asymptotic_error` | mean of last 2–3 available exposures (or fitted asymptote) | Estimate of **P(slip)** floor / ceiling of learnability |
| `total_gain` | `initial_error − asymptotic_error` | Magnitude of the practice effect |
| `relative_gain` | `total_gain / initial_error` | Normalised learnability (fair across easy/hard KCs) |
| `learning_rate` | fitted decay constant (see §9) | **BKT `P(T)`** initial estimate; bandit **`Q_Learning`** term |
| `half_life_exposures` | exposures to halve the error toward asymptote | Spaced-repetition scheduling; "how many reps to mastery" |
| `auc_error` | area under the (available) curve | Overall difficulty-over-practice score |
| `coverage_depth` | # non-null exposures | **Confidence weight** for every feature above |
| `curve_class` | shape label from §7 | Treatment-plan trigger |
| `monotonicity` | Spearman(error, k) | Is this KC "well-behaved"? |

- Explicitly propagate `coverage_depth` as a confidence flag — features from KCs with coverage 1–2 are unreliable and must be blended toward the global prior (mirror the empirical-Bayes idea in the difficulty formula).

## 9. Curve fitting (extract the learning-rate parameter)
- Fit two classic learning-curve models per KC that has enough points (e.g. coverage ≥ 4):
  - **Exponential:** `error(k) = a·exp(−b·(k−1)) + c` → `b` is the learning rate, `c` the asymptote.
  - **Power law (Newell & Rosenbloom):** `error(k) = a·k^(−b) + c`.
- Report fit quality (R²) distribution; which model fits better on average; how many KCs are unfittable (too few points).
- Extract `b` (learning rate) and `c` (asymptote) as the key model-ready parameters. Sanity-check ranges and correlation with `total_gain`.
- **Benefit to us:** `b` → initialises **BKT `P(T)`** (probability of learning per opportunity) instead of a flat guess; `c` → informs **`P(S)`/difficulty** floor. This is the single highest-value output of the notebook.

## 10. Structure: does practice effect vary by topic / depth / difficulty?
- Join the per-KC feature table with `kc_metadata` (`kc_route`, `attempted`, `error_rate`) and the KC **tree** (module = level-1/2 ancestor, tree depth).
- Group `learning_rate`, `total_gain`, `initial_error` by **module** (Geometry, Number Theory, Combinatorics, Calculation, Word Problems, …) and by **tree depth** — which topics improve with practice and which stay stubbornly hard?
- Correlate `initial_error` with `kc_metadata.error_rate` (they should broadly agree — a cross-file consistency check) and with `attempted` (do heavily-practised KCs learn faster?).
- **Benefit to us:** feeds **curriculum sequencing** — modules with high learnability are good early wins; low-learnability modules need scaffolding, not repetition.

## 11. Cross-file consistency & triangulation
- `exposure_1` error vs `kc_metadata.error_rate`: correlate; explain divergence (exposure_1 is *first-attempt only*; kc_metadata is *all attempts pooled*, so exposure_1 ≥ overall on learnable KCs).
- Relate to `chronological_delta.csv` (before/after error + `learning_delta`) and `practice_effect_perQ.csv` (same idea at question grain) — do the three tell a consistent practice-effect story? Note grain differences (KC vs question).
- **Benefit to us:** confirms which practice-effect signal is the most stable to productionise.

## 12. "How each feature benefits us" — the required synthesis section
Write a markdown table mapping every raw and derived feature to the Noxed module it serves. Minimum coverage:

- **`exposure_1` / `initial_error`** → **Diagnostic + Difficulty formula:** cold-start difficulty and initial mastery prior `P(L₀)` for a KC with no student history (exactly the Egyptian-science cold-start case).
- **`learning_rate` (b)** → **Progress Tracking (BKT `P(T)`)** and **Bandit reward `Q_Learning`:** how much a student is expected to improve by practising this KC → the reward should favour high-`P(T)`, not-yet-mastered KCs.
- **`asymptotic_error` (c)** → **BKT `P(S)` / ceiling:** the irreducible error even after practice; bounds achievable mastery.
- **`half_life` / curve shape** → **Spaced repetition & treatment plan:** when to re-surface a KC; forgetting curves (rising) trigger review; flat-high curves trigger scaffolding (Weak Prior Knowledge / Knowledge Absence cases).
- **`coverage_depth`** → **Confidence weighting everywhere:** low-coverage KCs blend toward global priors (empirical-Bayes), preventing the bandit from acting on noise.
- **Aggregate curve** → **Product KPI:** proof the adaptive loop actually teaches (a negative practice-effect delta is the RQ4 efficacy signal).

## 13. Data-quality caveats & recommendations (markdown)
State plainly: (1) **survivorship/right-censoring** dominates columns beyond ~exposure 8–10; (2) **no per-cell sample size** → late columns are low-confidence and must be weighted by `coverage_depth`; (3) 35% of KCs get *worse* exp1→exp2 (small-N + interleaving) — don't treat the practice effect as guaranteed monotonic; (4) this is a **math proxy** — validate that the *method* (fit a decay, extract `P(T)`) transfers, and re-estimate all numbers on Egyptian-science logs once collected; (5) recommend the pipeline log **per-exposure N** in production so this file's biggest gap is closed.

## 14. Exports / artifacts the notebook must produce
- `kc_practice_features.csv` — the per-KC feature table from §8–§9 (this is the reusable output for the modelling team).
- Saved figures: survivorship curve, aggregate learning curve (naïve vs balanced), curve-shape gallery, module-level comparison, R² distribution.
- A short "Findings & model-ready parameters" markdown summary at the end (5–8 bullets).

---

## Acceptance criteria
- [ ] All §0 ground-truth numbers reproduced via asserts.
- [ ] Missingness treated as **censoring/survivorship**, with the balanced-cohort curve shown alongside the naïve one.
- [ ] Per-KC feature table exported, every feature carrying a `coverage_depth` confidence flag.
- [ ] At least one learning-curve model fitted; `learning_rate` and `asymptote` extracted with R² reported.
- [ ] Structure-by-module/depth analysis present (joined to `kc_metadata` + tree).
- [ ] Explicit **feature → Noxed module** benefit table (§12).
- [ ] Caveats section names the math-proxy / cold-start / no-per-cell-N limitations.
- [ ] Every non-trivial cell preceded by a markdown cell stating *the question it answers* and *why it matters for Noxed*.

## Guardrails
- Sort exposure columns numerically, never lexically.
- Never compute a cross-KC mean at exposure `k` without reporting `n_kc` at that `k`.
- Do not interpret late-exposure improvement as learning without the balanced-cohort control.
- Keep raw vs derived features separate; treat everything as proxy evidence for *method validation*, not Egyptian-science fact.
