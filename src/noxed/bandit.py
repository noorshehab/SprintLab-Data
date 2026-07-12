"""Question-Bandit reward design (RQ3.1): the original reward
R = (1/sum(priors)) * Q_diff * Q_learn favours maximum difficulty; the
desirable-difficulty variant R' replaces Q_diff with a Gaussian match
against a target success probability p* (~0.8), keeping the learner in the
Zone of Proximal Development."""
import numpy as np


def success_probability(p_l: np.ndarray, p_s: np.ndarray, p_g: np.ndarray) -> np.ndarray:
    return p_l * (1 - p_s) + (1 - p_l) * p_g


def reward_raw(sum_priors: np.ndarray, q_difficulty: np.ndarray, q_learning: np.ndarray) -> np.ndarray:
    return (1.0 / np.clip(sum_priors, 1e-6, None)) * q_difficulty * q_learning


def desirable_difficulty_match(p_success: np.ndarray, p_star: float = 0.8, tau: float = 0.15) -> np.ndarray:
    return np.exp(-((p_success - p_star) ** 2) / (2 * tau**2))


def reward_desirable(sum_priors: np.ndarray, p_success: np.ndarray, q_learning: np.ndarray, p_star: float = 0.8, tau: float = 0.15) -> np.ndarray:
    match = desirable_difficulty_match(p_success, p_star=p_star, tau=tau)
    return (1.0 / np.clip(sum_priors, 1e-6, None)) * match * q_learning


def policy_argmax_success_prob(p_success: np.ndarray, reward: np.ndarray, candidate_mask: np.ndarray | None = None) -> float:
    """Given a candidate pool's reward vector, pick the arg-max item and
    return its success probability -- run once per (policy, context) to
    build the success-probability distribution each policy induces."""
    r = reward.copy()
    if candidate_mask is not None:
        r = np.where(candidate_mask, r, -np.inf)
    idx = np.argmax(r)
    return float(p_success[idx])


def direct_method_ope(observed_reward: np.ndarray, predicted_reward_model, contexts) -> dict:
    """A minimal direct-method off-policy-evaluation sketch: fit
    `predicted_reward_model` (any sklearn-like regressor) on observed
    (context, reward) pairs, then report in-sample R^2 as a rough calibration
    check. Documented as a SKETCH: the proxy has no logged propensities, so a
    real doubly-robust OPE is out of scope -- this only checks the reward
    model itself is learnable, not that the policy is safe to deploy."""
    predicted_reward_model.fit(contexts, observed_reward)
    r2 = predicted_reward_model.score(contexts, observed_reward)
    return {"in_sample_r2": float(r2)}
