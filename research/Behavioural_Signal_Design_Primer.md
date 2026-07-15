# Designing Trustworthy Behavioural Signals from Telemetry — A Primer

*A mental-model guide for the Sprintlab LIS behavioural-diagnosis work (Framework §4–6). It is not a signal spec — it is the reasoning that lets you design or reject any signal someone proposes.*

The craft here is one thing: **turning a fuzzy psychological word ("impulsive", "anxious", "disengaged") into a number you can trust from logs — without fooling yourself.** Master the five ideas below and the specific formulas write themselves.

---

## Start here: you measure *process*, not *answers*

A right/wrong mark is an **outcome** — one bit. Behaviour is the **process** that produced it. One student, one wrong answer to an easy question. That `0` could be:

- never learned it (a pure guess),
- knew it but rushed,
- misread "which is **not** true",
- or holds a confident misconception.

The bit is *identical* in all four cases. So diagnosis is: **add observed dimensions until the causes separate.** Time separates the rusher from the careful student. *Which* wrong option they picked separates the misreader from the misconception-holder. *What happened just before* separates the stressed student from the disengaged one.

> Every signal you add is another axis that splits a lump of confounded causes. The job is buying dimensions to break ties.

---

## Idea 1 — The one confound that contaminates everything: ability

**A student who simply doesn't know the material produces the exact surface signature of almost every behavioural problem at once.** They're slow (looks like a processing-speed deficit), they err near distractors (looks like poor selective attention), they hesitate and change answers (looks like low self-efficacy), they fade late in a match when the hard items pile up (looks like fatigue).

So "slow → processing-speed problem" is **wrong more often than right**, because slowness isn't a trait — it's a symptom with a dozen causes, and *"didn't know it"* is the most common one.

> Ability leaks into every raw metric. **Any signal that plain not-knowing can also produce is not yet a signal.**

---

## Idea 2 — The fix is an identification strategy: make each student their own control

The deepest, most transferable lesson — borrowed straight from experimental design. You cannot compare a student to a **population average**: students differ in baseline speed, age, and ability, so the average is the *wrong baseline* (a naturally fast reader trips every "impulsive" threshold; a careful one trips every "stressed" threshold).

> **A signal is a *difference between two conditions that share everything except the one thing you're measuring* — read within the same student.**

Watch it work:

| Construct | The within-student contrast |
|---|---|
| Selective attention | same item type, **distractor tile present vs absent** |
| Reading skill | same student, **plain-worded vs tricky-worded** items on matched content |
| Working memory | **low-load vs high-load** items *in a cluster already mastered at low load* |
| Processing speed | fast vs slow **on items they got right** |

The pattern: hold content/difficulty fixed, vary *only* the cognitive demand, read the **difference**. Apply the same logic to time — don't ask "is this student slow?", ask "is this student slow **relative to their own normal pace**?" That self-normalisation is exactly what **Personal Pace Factor** is for.

This is why good signals are almost always **contrasts and differences**, never raw thresholds like "90% of the time on one item".

---

## Idea 3 — The 2×2 that does most of the work: speed × correctness

One continuous number (response time) plus one bit (correct?) already carves the space:

|            | **Fast**                                   | **Slow**                                             |
|------------|--------------------------------------------|------------------------------------------------------|
| **Right**  | fluency / mastery (or a lucky guess)       | processing-speed cost, *or* careful/anxious checking |
| **Wrong**  | impulsivity / low effort / disengaged guess| genuine difficulty — engaged but failed (reasoning ceiling, WM overload) |

Two non-obvious rules fall out:

1. **Always condition on correctness.** RT on *wrong* answers is hopelessly ambiguous. Clean processing-speed signal lives only in **fast/slow among items they got *right***. Discard the wrong ones for that measurement.
2. **Never read speed without accuracy beside it.** The *speed–accuracy tradeoff* means a student can convert one into the other at will. "Fast" alone means nothing; "fast **and** on the specific intuitive wrong lure" means impulsivity.

If you keep only one tool from this primer, keep this grid.

---

## Idea 4 — Trait vs State decides your *window*

Constructs come in two flavours that demand opposite strategies:

- **Traits** (impulsivity, self-efficacy) are *stable dispositions*. Seen only by **averaging over many observations** (law of large numbers). A single fast wrong answer, or one talk-yourself-out-of-a-right-answer, is *noise*. Long rolling windows, confidence tiers.
- **States** (stress, tilt) are *transient and trigger-caused*. Seen by **tight time-locking**: compare the item right *before* a trigger (lost the lead, got attacked) to the item right *after*. Average a state over a whole session and you **erase** it.

> The object you measure dictates the window. The classic error is reading a *trait* from one *state-like* event ("he rushed once → he's impulsive"). Same data, wrong window, false diagnosis.

---

## Idea 5 — Reliability and validity are different problems

- **Reliability** — is the signal stable, or noise? Fixed by **more data** (rolling windows, confidence tiers).
- **Validity** — does it measure what you *claim*? Fixed by **controlling the confound** (the contrasts in Idea 2).

The trap: when a signal looks shaky, the instinct is "collect more data." But if the signal is *confounded* (measuring ability, not impulsivity), more data just makes it **confidently wrong** — a sharper biased estimate. More data buys reliability; only good design buys validity. You need both, and they are not substitutes.

---

## The portable checklist

Whenever *anyone* — you, a teammate, the framework — proposes a behavioural signal, run it through these five questions:

1. **What's the contrast?** Against the student's own baseline, or a bare population threshold? (The latter is a red flag.)
2. **Did you condition on correctness and ability?** Could "didn't know it" produce this exact signal?
3. **Trait or state — is the window right?** Averaging a state, or over-reading a single event as a trait?
4. **Reliable *and* valid?** Enough observations *and* the confound controlled — not just one.
5. **What else produces this signature?** List every alternative cause; if you can't rule them out with another dimension, the signal isn't ready.

That is the entire discipline. The concrete formulas — PPF denominators, lure-specific fast errors, plain-vs-tricky gaps — are just these five ideas applied over and over.
