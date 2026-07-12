"""Loaders for the XES3G5M proxy dataset (sprintlabfiles) and the
canonical-event reconstruction used by every downstream notebook.

Raw `responses.csv` grain is (student, question, KC): a question carrying
k knowledge components is expanded into k rows sharing the same timestamp
and the same correctness. `canonicalize_events` collapses that back to one
row per real answering event and derives an attempt index, since the raw
file has no native attempt/sequence column.
"""
import re
from pathlib import Path

import pandas as pd

RESPONSES_COLS = ["fold", "uid", "questions", "concepts", "responses", "timestamps"]


def load_responses(data_dir: Path, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(data_dir / "responses.csv", nrows=nrows)


def load_question_metadata(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "question_metadata.csv")


def load_analysis_metadata(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "analysis_metadata.csv")


def load_kc_metadata(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "kc_metadata.csv")


def load_practice_effect_perKC(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "practice_effect_perKC.csv")


def load_practice_effect_perQ(data_dir: Path, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(data_dir / "practice_effect_perQ.csv", nrows=nrows)


def load_question_differentiation(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "question_diffrentiation.csv")


def load_question_differentiation_cumulative(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "question_diffrentiation_cumulative.csv")


def load_chronological_delta(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(data_dir / "chronological_delta.csv")


def canonicalize_events(responses: pd.DataFrame) -> pd.DataFrame:
    """Collapse (uid, questions, timestamps) duplicate rows produced by
    multi-KC expansion into one row per real answering event, with a list
    of the KCs involved and a per-student, per-question attempt index
    derived from timestamp order (ties broken by original row order).
    """
    grouped = (
        responses.sort_values(["uid", "timestamps"], kind="stable")
        .groupby(["uid", "questions", "timestamps"], sort=False, as_index=False)
        .agg(concepts=("concepts", lambda s: sorted(set(s))), response=("responses", "first"))
    )
    grouped = grouped.sort_values(["uid", "timestamps"], kind="stable").reset_index(drop=True)
    grouped["event_seq"] = grouped.groupby("uid").cumcount()
    grouped["attempt_index"] = grouped.groupby(["uid", "questions"]).cumcount()
    return grouped


def load_or_build_canonical_events(data_dir: Path, cache_dir: Path, force: bool = False) -> pd.DataFrame:
    """Cached wrapper around load_responses + canonicalize_events. Building
    the canonical table from the full 5.1M-row log takes ~2 minutes; every
    notebook after NB00 reuses the cached parquet instead of repeating it."""
    cache_path = cache_dir / "canonical_events.parquet"
    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)
    responses = load_responses(data_dir)
    canonical = canonicalize_events(responses)
    canonical.to_parquet(cache_path, index=False)
    return canonical


_TREE_LINE = re.compile(r"^(?P<indent>[\s│]*)(?:[├└]──\s*)?(?P<en>[^(]+?)\s*\((?P<zh>[^)]+)\)\s*$")


def parse_kc_tree(tree_translation_path: Path) -> pd.DataFrame:
    """Parse tree_translation.txt (an indented English(Chinese) tree) into a
    flat table: zh_name, en_name, depth, top_module (the depth-1 ancestor).
    Depth is inferred from indentation width (4 spaces == 1 level).
    """
    rows = []
    stack: list[str] = []  # en names at each depth, index = depth-1
    with open(tree_translation_path, encoding="utf-8") as f:
        for raw in f:
            if not raw.strip():
                continue
            m = _TREE_LINE.match(raw.rstrip("\n"))
            if not m:
                continue
            indent = m.group("indent").replace("│", " ")
            depth = indent.count(" ") // 4 + 1
            en, zh = m.group("en").strip(), m.group("zh").strip()
            stack = stack[: depth - 1] + [en]
            top_module = stack[0] if stack else en
            rows.append({"zh_name": zh, "en_name": en, "depth": depth, "top_module": top_module})
    return pd.DataFrame(rows).drop_duplicates(subset=["zh_name"], keep="first")


def attach_kc_taxonomy(kc_metadata: pd.DataFrame, tree_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join kc_metadata (kc_route in Chinese) onto the parsed tree to
    recover each KC's English name, tree depth, and top-level module."""
    out = kc_metadata.merge(
        tree_df, left_on="kc_route", right_on="zh_name", how="left", validate="m:1"
    )
    return out
