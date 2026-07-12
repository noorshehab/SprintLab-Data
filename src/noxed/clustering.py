"""Diagnostic-module clustering: Agglomerative Hierarchical Clustering over
KC feature vectors, a feature-set ablation harness (silhouette + external
coherence against the authored taxonomy), and the cluster-size/multiplicity
-> prior mapping (RQ1.1/RQ1.2, backlog D-3)."""
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler


def cluster_kcs(features: pd.DataFrame, n_clusters: int = 12) -> np.ndarray:
    """Agglomerative (Ward-linkage) clustering over standardised KC features.
    Every KC is assigned to exactly one cluster (a true partition)."""
    X = StandardScaler().fit_transform(features.to_numpy())
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    return model.fit_predict(X)


def evaluate_feature_set(features: pd.DataFrame, taxonomy_labels: pd.Series, n_clusters: int = 12) -> dict:
    """One ablation cell: cluster on `features`, report internal quality
    (silhouette) and external agreement with the authored module taxonomy
    (adjusted Rand index) -- meaningful clusters should score above-chance
    on both, not just look tidy internally."""
    labels = cluster_kcs(features, n_clusters=n_clusters)
    X = StandardScaler().fit_transform(features.to_numpy())
    sil = silhouette_score(X, labels) if len(set(labels)) > 1 else float("nan")
    ari = adjusted_rand_score(taxonomy_labels.astype(str), labels)
    return {"silhouette": float(sil), "ari_vs_taxonomy": float(ari), "n_clusters_found": len(set(labels))}


def cluster_to_prior(cluster_labels: np.ndarray, is_wrong_pass: bool, multiplicity: np.ndarray | None = None) -> np.ndarray:
    """Rank-based, bounded-by-construction cluster -> prior[0,1] mapping.
    Larger wrong-answer clusters => lower prior (weaker topic); larger
    correct-answer clusters => higher prior. `multiplicity` (how many
    distinct clusters a KC's answers spread across) further lowers the
    prior when supplied, matching the RQ1.2 cross-cluster-multiplicity rule.
    """
    sizes = pd.Series(cluster_labels).map(pd.Series(cluster_labels).value_counts())
    rank = sizes.rank(pct=True).to_numpy()  # in (0,1], larger cluster -> higher rank
    prior = (1 - rank) if is_wrong_pass else rank
    if multiplicity is not None:
        mult = np.asarray(multiplicity, dtype=float)
        mult_rank = pd.Series(mult).rank(pct=True).to_numpy()
        prior = prior * (1 - 0.5 * mult_rank) if is_wrong_pass else prior * (1 - 0.3 * mult_rank)
    return np.clip(prior, 0.0, 1.0)
