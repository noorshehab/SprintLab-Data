"""Validate whether the authored KC tree encodes real prerequisite
structure: does a student's parent-KC accuracy predict child-KC accuracy
above baseline, and does a data-driven accuracy-ordering agree with the
authored parent->child edges (RQ1.3)?"""
import numpy as np
import pandas as pd


def build_parent_child_edges(tree_df: pd.DataFrame) -> pd.DataFrame:
    """Derive parent->child edges from the parsed tree (consecutive depths
    sharing the same top_module lineage), keyed by en_name."""
    edges = []
    by_depth = {d: g for d, g in tree_df.groupby("depth")}
    for depth in sorted(by_depth):
        if depth + 1 not in by_depth:
            continue
        parents = by_depth[depth]
        children = by_depth[depth + 1]
        for _, child in children.iterrows():
            same_module = parents[parents["top_module"] == child["top_module"]]
            if len(same_module):
                edges.append({"parent": same_module.iloc[-1]["en_name"], "child": child["en_name"]})
    return pd.DataFrame(edges).drop_duplicates()


def conditional_accuracy_lift(student_kc_accuracy: pd.DataFrame, parent_kc: int, child_kc: int) -> dict:
    """Among students who attempted both `parent_kc` and `child_kc`, compare
    child accuracy for students who are ABOVE-median on the parent vs.
    BELOW-median -- a positive lift is evidence the parent gates the child.
    `student_kc_accuracy` is indexed by uid with one column per kc_id."""
    if parent_kc not in student_kc_accuracy.columns or child_kc not in student_kc_accuracy.columns:
        return {"lift": np.nan, "n_students": 0}
    both = student_kc_accuracy[[parent_kc, child_kc]].dropna()
    if len(both) < 10:
        return {"lift": np.nan, "n_students": len(both)}
    median_parent = both[parent_kc].median()
    high = both[both[parent_kc] >= median_parent][child_kc].mean()
    low = both[both[parent_kc] < median_parent][child_kc].mean()
    return {"lift": float(high - low), "n_students": int(len(both)), "high_group_acc": float(high), "low_group_acc": float(low)}


def discover_prerequisite_direction(kc_error_rates: pd.Series, co_attempt_pairs: pd.DataFrame) -> pd.DataFrame:
    """For each candidate KC pair that co-occurs, propose a direction via
    the easier/lower-error KC as the likely prerequisite (a simple,
    documented heuristic -- prerequisites tend to be mastered earlier and
    thus show lower pooled error than the topics that build on them)."""
    out = co_attempt_pairs.copy()
    out["error_a"] = out["kc_a"].map(kc_error_rates)
    out["error_b"] = out["kc_b"].map(kc_error_rates)
    out["proposed_parent"] = np.where(out["error_a"] <= out["error_b"], out["kc_a"], out["kc_b"])
    out["proposed_child"] = np.where(out["error_a"] <= out["error_b"], out["kc_b"], out["kc_a"])
    return out


def agreement_with_authored_tree(discovered: pd.DataFrame, authored_edges: set[tuple]) -> float:
    """Fraction of discovered (parent, child) pairs that also appear as an
    authored edge (in either direction, since authored depth doesn't always
    align with empirical difficulty ordering) -- an honest agreement score,
    not an assumption that the tree is correct."""
    if len(discovered) == 0:
        return float("nan")
    hits = discovered.apply(
        lambda r: (r["proposed_parent"], r["proposed_child"]) in authored_edges
        or (r["proposed_child"], r["proposed_parent"]) in authored_edges,
        axis=1,
    )
    return float(hits.mean())
