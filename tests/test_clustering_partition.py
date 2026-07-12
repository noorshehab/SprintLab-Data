import numpy as np
import pandas as pd
from hypothesis import given, settings, strategies as st

from noxed.clustering import cluster_kcs, cluster_to_prior


@given(
    n_kcs=st.integers(min_value=10, max_value=60),
    n_clusters=st.integers(min_value=2, max_value=8),
    seed=st.integers(min_value=0, max_value=1000),
)
@settings(deadline=None, max_examples=25)
def test_clustering_is_a_true_partition(n_kcs, n_clusters, seed):
    """Every KC belongs to exactly one cluster: labels array has one entry
    per input row, every label in range, no KC dropped or duplicated."""
    n_clusters = min(n_clusters, n_kcs)
    rng = np.random.default_rng(seed)
    features = pd.DataFrame(rng.normal(size=(n_kcs, 3)), columns=["a", "b", "c"])
    labels = cluster_kcs(features, n_clusters=n_clusters)
    assert len(labels) == n_kcs
    assert set(labels).issubset(set(range(n_clusters)))
    assert not np.isnan(labels).any()


@given(
    n_kcs=st.integers(min_value=10, max_value=50),
    is_wrong=st.booleans(),
    seed=st.integers(min_value=0, max_value=1000),
)
@settings(deadline=None, max_examples=25)
def test_cluster_prior_bounded(n_kcs, is_wrong, seed):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 5, size=n_kcs)
    prior = cluster_to_prior(labels, is_wrong_pass=is_wrong)
    assert len(prior) == n_kcs
    assert np.all(prior >= 0.0) and np.all(prior <= 1.0)


def test_cluster_prior_monotonic_in_size_for_wrong_pass():
    """Larger wrong-answer clusters should map to a lower-or-equal prior
    than smaller ones (RQ1.2)."""
    labels = np.array([0] * 20 + [1] * 2)  # cluster 0 much bigger than cluster 1
    prior = cluster_to_prior(labels, is_wrong_pass=True)
    assert prior[labels == 0].mean() <= prior[labels == 1].mean()


def test_cluster_prior_monotonic_in_size_for_correct_pass():
    labels = np.array([0] * 20 + [1] * 2)
    prior = cluster_to_prior(labels, is_wrong_pass=False)
    assert prior[labels == 0].mean() >= prior[labels == 1].mean()
