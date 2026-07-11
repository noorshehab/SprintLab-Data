# Noxed — Behavioural Diagnosis Module: Literature-Review Research Plan

**Project:** Noxed / SprintLab — adaptive competitive science-quiz game for Egyptian schools
**Module in focus:** M2 — Behavioural Diagnosis (response metadata → behavioural profile → treatment plan)
**Deliverable type:** Structured literature review + a validated catalogue of behavioural metrics
**Status:** Planning note (defines scope, method, and output before the review is written)

---

## 1. Purpose of this task

The Behavioural Diagnosis module is meant to train a neural network on a dataset of **"scenarios."** A scenario is a stable behavioural pattern — a student who is bad at time management, or impulsive, or nervous, or bad at context-switching between topics — and each scenario generates a characteristic set of responses (correct vs. incorrect) with associated **metadata** (time spent solving, mistake patterns on specific question types). We already own a **map** from scenarios → treatment plans. The NN's job is to read the metadata and some **calculated behavioural metrics** and decide *which scenario the student is in*.

The blocker: **we do not yet know what those calculated metrics should be, and we have no labelled behavioural data.** Before we can build features or synthesise training data, we need to know how the education / cognitive-psychology literature actually identifies behavioural problems in learners, what student "profiles" are recognised, and which observable indicators map to each latent trait.

**This task is therefore a literature review**, not a modelling task. Its output feeds directly into feature engineering (RQ2.1), synthetic-scenario generation (RQ2.3), and the treatment mapping (RQ2.4) of the [Noxed Research Questions](./Noxed_Research_Questions.pdf) note.

---

## 2. Problem definition

**Core working assumption.** Student behaviour is driven by **stable individual differences** — time-management ability, impulse control, anxiety, executive function. If these latent traits are real and stable, then:

> **Main problem:** Can we *measure* a student's latent behavioural traits from in-game **response metadata + response patterns + derived metrics** alone, well enough to classify which behavioural scenario they are in and trigger the matched treatment?

This is a *measurement/operationalisation* problem before it is a modelling problem: we must turn fuzzy psychological constructs ("anxiety," "impulsivity") into concrete, log-computable numbers, and know from the literature which indicators are valid.

### 2.1 Sub-problems

| # | Sub-problem | What the review must answer |
|---|---|---|
| **SP1 — Construct inventory** | *What behavioural profiles even exist?* | Which learner traits/profiles does the education literature recognise as real, distinct, and stable? Do our 12 authored cases (4 families) line up with established constructs, or are we conflating / missing some? |
| **SP2 — Indicator mapping** | *What observable indicator predicts each trait?* | For each trait, what behavioural/log-derived signal has been shown to indicate it (e.g. response-time effort → disengagement; RT variance → anxiety)? |
| **SP3 — Metric operationalisation** | *How is each indicator computed from logs?* | Exact formulas/thresholds used in prior work (fast-guess cutoffs, wheel-spinning attempt counts, slip estimation, context-switch cost). Feeds our metrics table. |
| **SP4 — Confound & validity** | *When does a metric measure behaviour vs. ability?* | How does prior work separate a slow-because-anxious student from a slow-because-struggling one? Normalisation, within-student contrasts, difficulty control. |
| **SP5 — Ground truth & labels** | *How do others get labels without observers?* | BROMP field observation, sensor-free detectors, self-report scales, synthetic generation. What is the minimum-cost path to labels for us? |
| **SP6 — Trait → treatment (ATI)** | *Does matching treatment to trait actually help?* | Evidence from Aptitude–Treatment Interaction that trait-matched instruction beats a one-size baseline, and for which profiles it does/doesn't hold. |
| **SP7 — Transfer & cold start** | *Does any of this survive the domain gap?* | Our proxy data is Chinese math (XES3G5M); target is Arabic school science. What transfers, what must be re-instrumented? |

---

## 3. Scope — the seven research areas

The task names seven areas. Each is mapped below to the sub-problems it answers and to the Noxed case families (A–D) it informs.

| # | Research area | Serves | Informs case family | Key questions to extract |
|---|---|---|---|---|
| 1 | **Learning Analytics** | SP2, SP3, SP5 | all | What log features do LA/EDM systems compute? Detector definitions (gaming, wheel-spinning, carelessness/slip, off-task). |
| 2 | **Metacognition & self-regulation (SRL)** | SP1, SP2 | C (exam skills), B | Help-seeking, monitoring, planning; how SRL failures show up in behaviour logs. |
| 3 | **Aptitude–Treatment Interaction (ATI)** | SP6 | all (the treatment map) | Does trait-matched instruction work? Effect sizes, replication caveats (Cronbach & Snow). |
| 4 | **Test anxiety & emotional regulation** | SP1, SP3, SP4 | D (psychological) | RT/error signatures of anxiety; recovery after failure; validated Arabic self-report scales. |
| 5 | **Executive function in learning** | SP1, SP3 | B (cognitive skills) | Working memory, inhibition, cognitive flexibility (Miyake unity/diversity); behavioural proxies for each. |
| 6 | **Time management in testing/learning** | SP3, SP4 | C (time mgmt.), D | Pacing, response-time effort, rapid-guessing, timeout behaviour, perfectionist over-dwell. |
| 7 | **Student affect & its detection** | SP1, SP2, SP5 | D | Affective states (boredom, frustration, confusion, flow), sensor-free detection, affect dynamics. |

**Out of scope for this review:** the knowledge-tracing / BKT / bandit modelling (covered in the RQ note), UI/game design of treatments, and any data-collection engineering. This review stops at *"here are the validated constructs, indicators, metrics, and labelling methods."*

---

## 4. Method

### 4.1 Search strategy
- **Databases / sources:** Google Scholar, ACM DL, ERIC, PsycINFO, arXiv/cs.CY, and the proceedings of **EDM, AIED, LAK, L@S** (the four venues where behaviour-in-logs work is published).
- **Seed papers (already identified — snowball from these):** Baker et al. (gaming the system), Beck & Gong (wheel-spinning), San Pedro/Baker (carelessness via slip), D'Mello & Graesser (affect dynamics), Ocumpaugh (BROMP), Miyake et al. (executive functions), Cassady & Johnson (cognitive test anxiety), Wise & Kong (response-time effort), Cronbach & Snow (ATI).
- **Snowballing:** forward (who cites the seed) + backward (seed's own references). Two hops per seed.
- **Query templates:** `("<trait>" OR "<synonym>") AND (log OR "response time" OR clickstream OR detector) AND (student OR learner)`.

### 4.2 Inclusion / exclusion
- **Include:** peer-reviewed or well-cited work that (a) defines a learner behavioural construct, or (b) gives a *computable* indicator from interaction logs, or (c) reports a trait→treatment efficacy result.
- **Prefer:** studies with per-attempt logs, K-12 or comparable populations, and reproducible metric definitions.
- **Exclude:** pure classroom-observation studies with no log link (unless they define a construct we need), opinion pieces, and anything without an operational indicator — *unless* it is a foundational construct paper (e.g. Miyake).
- **Flag domain gap explicitly** for every source: note domain (math/language/science), language (mostly non-Arabic), and age band, since our target is Arabic × science × school-age.

### 4.3 Extraction template (one row per source)
`Citation | Construct(s) | Observable indicator | Exact metric/formula/threshold | Data used | Validation method | Effect size / accuracy | Domain–language–age | Relevance to Noxed case | Confound notes`

### 4.4 Synthesis
1. Collapse extracted indicators into a **master metrics table** (extends §Metrics of the RQ note), tagging each metric *validated / adapted / novel*.
2. Reconcile the **12 authored cases** against literature constructs — confirm, split, merge, or mark "not in literature (self-report only)."
3. Produce a **construct → indicator → metric → treatment → evidence** matrix — the single artifact the NN feature set and the synthetic-data generator are built from.
4. Write the **confound/validity section** (SP4): the normalisation and within-student-contrast rules that keep behaviour separable from ability.

---

## 5. Deliverable structure (the literature review document)

1. **Introduction** — the measurement problem, why behaviour needs its own labels.
2. **Section per research area (×7)** — findings, key papers, extracted indicators.
3. **Construct catalogue** — every recognised behavioural profile, cross-walked to Noxed's 4 families / 12 cases.
4. **Master metrics table** — every log-derived metric, its construct, formula, source, and validated/adapted/novel tag.
5. **Confounds & validity** — separating behaviour from ability; competitive-timer confounds; normalisation rules.
6. **Ground-truth strategy** — BROMP vs. sensor-free vs. self-report vs. synthetic; the recommended path for Noxed.
7. **ATI efficacy evidence** — does trait-matched treatment work; caveats.
8. **Gaps & risks** — the Arabic-science-schoolchild gap; traits with no log-only ground truth (anxiety, impulsivity).
9. **Recommendations** — the metric set to implement first; the synthetic-scenario parameters to seed; open questions for RQ2.
10. **Annotated bibliography.**

---

## 6. Phasing

| Phase | Work | Output |
|---|---|---|
| **P0 (done)** | Problem framing, RQ decomposition, seed bibliography, first metrics draft | `Noxed_Research_Questions.pdf`, this plan |
| **P1** | Areas 1, 6, 7 (Learning Analytics, Time Mgmt, Affect) — the log-signal-heavy areas | Extraction rows + draft metrics |
| **P2** | Areas 4, 5 (Test anxiety, Executive function) — the trait/construct areas | Construct catalogue |
| **P3** | Areas 2, 3 (SRL, ATI) — the strategy/efficacy areas | ATI + treatment-mapping section |
| **P4** | Synthesis: master metrics table, construct cross-walk, confounds, recommendations | Final literature-review document |

---

## 7. Success criteria

- Every one of the **12 authored cases** is either grounded in a cited construct or explicitly flagged as self-report-only.
- Every metric in the **master table** has a source and a computability note (can we compute it from current logs — yes/no/needs-instrumentation).
- SP4 (confounds) yields **concrete normalisation rules**, not just a warning.
- SP5 yields a **ranked, costed labelling recommendation** (what we synthesise vs. what we must collect).
- The output plugs directly into RQ2.1 (features), RQ2.3 (synthetic labels), and RQ2.4 (treatment smoothness) with no re-framing needed.

---

*Companion documents:* [`Noxed_Research_Questions.pdf`](./Noxed_Research_Questions.pdf) (RQs, case taxonomy, metrics, difficulty formula), [`Noxed_Flowchart.pdf`](./Noxed_Flowchart.pdf) (module pipeline), [`EDA_TASK_practice_effect_perKC.md`](./EDA_TASK_practice_effect_perKC.md) (proxy-data EDA).
