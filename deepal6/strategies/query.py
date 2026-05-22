"""
deepal.strategies.query
-----------------------
All six query strategies with a unified interface:

    fn(model, data, pool_idx, n_query, **kwargs) -> np.ndarray of local indices

'model'    : a trained PyTorch model (CreditNet or ResNet-18 head)
'data'     : a DataModule (TabularDataModule or ImageDataModule)
'pool_idx' : array/list of global indices into the unlabeled pool
'n_query'  : number of samples to select
**kwargs   : strategy-specific extras (e.g. mc_passes, labeled_idx)

Returns: 1-D np.ndarray of LOCAL indices (into pool_idx, not X_train).
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from deepal6.strategies import register_strategy
from deepal6.exceptions import StrategyError


# ──────────────────────────────────────────────────────────────────────────────
# Strategy 1 — Random Sampling (Baseline)
# ──────────────────────────────────────────────────────────────────────────────

def random_sampling(model, data, pool_idx, n_query, **kwargs) -> np.ndarray:
    """
    Uniform random draw from the unlabeled pool.

    This is the baseline. Any strategy that cannot consistently beat random
    provides no practical labelling efficiency gain.

    No model information is used. Serves as the lower bound in all comparisons.
    """
    if len(pool_idx) == 0:
        raise StrategyError("random_sampling: pool_idx is empty — no samples to query.")
    n = min(n_query, len(pool_idx))
    return np.random.choice(len(pool_idx), n, replace=False)


# ──────────────────────────────────────────────────────────────────────────────
# Strategy 2 — Entropy Sampling
# ──────────────────────────────────────────────────────────────────────────────

def entropy_sampling(model, data, pool_idx, n_query, **kwargs) -> np.ndarray:
    """
    Select samples with the highest predictive entropy:
        H[y|x] = -(p log p + (1-p) log(1-p))

    Targets total uncertainty (aleatoric + epistemic).
    Sensitive to miscalibration: overconfident models compress entropy toward
    zero, degrading selection quality — especially on small labeled sets or
    with ResNet on image data.
    """
    if len(pool_idx) == 0:
        raise StrategyError("entropy_sampling: pool_idx is empty.")

    probs = data.predict_proba(model, pool_idx)  # shape (N,)
    probs = np.clip(probs, 1e-10, 1 - 1e-10)
    entropy = -(probs * np.log(probs) + (1 - probs) * np.log(1 - probs))
    n = min(n_query, len(pool_idx))
    return np.argsort(entropy)[-n:]


# ──────────────────────────────────────────────────────────────────────────────
# Strategy 3 — Margin Sampling
# ──────────────────────────────────────────────────────────────────────────────

def margin_sampling(model, data, pool_idx, n_query, **kwargs) -> np.ndarray:
    """
    Select samples with the smallest margin |p - 0.5|.

    Focuses on the decision boundary. For binary classification this is
    equivalent to least-confidence sampling.
    Shares entropy's vulnerability to miscalibration; ignores full
    distribution shape.
    """
    if len(pool_idx) == 0:
        raise StrategyError("margin_sampling: pool_idx is empty.")

    probs = data.predict_proba(model, pool_idx)
    margin = np.abs(probs - 0.5)
    n = min(n_query, len(pool_idx))
    return np.argsort(margin)[:n]


# ──────────────────────────────────────────────────────────────────────────────
# Strategy 4 — BALD (Bayesian Active Learning by Disagreement)
# ──────────────────────────────────────────────────────────────────────────────

def bald_sampling(
    model,
    data,
    pool_idx,
    n_query,
    mc_passes: int = 20,
    **kwargs,
) -> np.ndarray:
    """
    Select samples maximising mutual information I[y;θ|x,D]:
        BALD(x) = H_total - E_θ[H[y|x,θ]]
                = H[mean_p] - mean(H[p_t])  over T MC-Dropout passes

    Targets *epistemic* uncertainty — disagreement between model parameter
    samples. More robust to miscalibration than Entropy/Margin because it
    measures stochastic forward-pass disagreement, not raw softmax values.

    Parameters
    ----------
    mc_passes : int
        Number of stochastic forward passes with dropout active.
        Higher = lower variance BALD estimate, but higher cost.
        Default: 20.

    Raises
    ------
    StrategyError : if the model contains no Dropout layers (MC Dropout
        requires dropout to remain active at inference time).
    """
    if len(pool_idx) == 0:
        raise StrategyError("bald_sampling: pool_idx is empty.")

    # Verify the model has Dropout (required for MC Dropout)
    has_dropout = any(isinstance(m, nn.Dropout) for m in model.modules())
    if not has_dropout:
        raise StrategyError(
            "bald_sampling requires a model with nn.Dropout layers "
            "(MC Dropout). Your model has none.\n"
            "Tip: pass dropout_rate > 0 when building the model, or switch "
            "to Entropy/Margin sampling."
        )

    # shape: (mc_passes, N)
    all_probs = data.predict_proba(model, pool_idx, mc_passes=mc_passes)
    all_probs = np.clip(all_probs, 1e-10, 1 - 1e-10)

    mean_p = all_probs.mean(axis=0)                          # (N,)
    total_entropy = -(
        mean_p * np.log(mean_p) + (1 - mean_p) * np.log(1 - mean_p)
    )
    per_pass_H = -(
        all_probs * np.log(all_probs) + (1 - all_probs) * np.log(1 - all_probs)
    )
    expected_H = per_pass_H.mean(axis=0)                     # (N,)
    bald_score = total_entropy - expected_H                  # (N,)

    n = min(n_query, len(pool_idx))
    return np.argsort(bald_score)[-n:]


# ──────────────────────────────────────────────────────────────────────────────
# Strategy 5 — Core-Set (Greedy k-Center)
# ──────────────────────────────────────────────────────────────────────────────

def coreset_sampling(
    model,
    data,
    pool_idx,
    n_query,
    labeled_idx=None,
    **kwargs,
) -> np.ndarray:
    """
    Greedy 2-approximation to the k-center problem in embedding space.

    Iteratively selects the pool point farthest from its nearest already-
    selected (or labeled) neighbour, maximising coverage of the representation
    space.

    Unaffected by model miscalibration. Effectiveness depends on embedding
    quality:
    - Tabular (CreditNet penultimate layer, ~16 dims): limited geometric richness
    - Image (ResNet-18 avgpool, 512 dims): richer, Core-Set expected to be
      more competitive (RQ3 hypothesis from thesis Chapter 5).

    Parameters
    ----------
    labeled_idx : array-like or None
        Global indices of currently labeled samples, used to initialise
        distances. If None, distances start from the first pool sample.
    """
    if len(pool_idx) == 0:
        raise StrategyError("coreset_sampling: pool_idx is empty.")

    embed_pool = data.get_embeddings(model, pool_idx)   # (|pool|, d)

    if labeled_idx is not None and len(labeled_idx) > 0:
        embed_labeled = data.get_embeddings(model, labeled_idx)
    else:
        embed_labeled = embed_pool[:1]

    # Initialise min-distances to labeled set
    dist = np.full(len(embed_pool), np.inf)
    for xl in embed_labeled:
        d = np.linalg.norm(embed_pool - xl, axis=1)
        dist = np.minimum(dist, d)

    selected = []
    n = min(n_query, len(pool_idx))
    for _ in range(n):
        idx = int(np.argmax(dist))
        selected.append(idx)
        d = np.linalg.norm(embed_pool - embed_pool[idx], axis=1)
        dist = np.minimum(dist, d)
        dist[idx] = -np.inf  # mark as selected

    return np.array(selected)


# ──────────────────────────────────────────────────────────────────────────────
# Strategy 6 — BADGE
# (Batch Active learning by Diverse Gradient Embeddings)
# ──────────────────────────────────────────────────────────────────────────────

def badge_sampling(model, data, pool_idx, n_query, **kwargs) -> np.ndarray:
    """
    Gradient embedding k-means++ (Ash et al., 2020).

    For each pool sample, computes the gradient of the loss w.r.t. the last
    linear layer weights, using the predicted pseudo-label ŷ = argmax p(y|x):
        g_x = ∇_{θ_last} ℓ(f_θ(x), ŷ_x)

    Gradient magnitude encodes prediction uncertainty;
    gradient direction encodes class discrimination.
    k-means++ seeding over the gradient matrix G ensures the selected batch
    is both uncertain and diverse.

    Hybrid strategy: combines the strengths of uncertainty (Entropy/Margin)
    and diversity (Core-Set) in a single principled objective.
    """
    if len(pool_idx) == 0:
        raise StrategyError("badge_sampling: pool_idx is empty.")

    G = data.get_gradient_embeddings(model, pool_idx)   # (|pool|, d_last)

    if G is None or len(G) == 0:
        raise StrategyError(
            "badge_sampling: gradient embedding extraction returned empty array. "
            "Ensure the model has at least one nn.Linear layer."
        )

    n = min(n_query, len(pool_idx))
    return _kmeans_pp(G, n)


def _kmeans_pp(G: np.ndarray, k: int) -> np.ndarray:
    """
    k-means++ seeding in gradient embedding space.
    Returns k indices into G.
    """
    n = len(G)
    idx = [int(np.random.randint(n))]
    for _ in range(k - 1):
        # Squared distance from each point to its nearest centre
        d2 = np.array([
            min(np.linalg.norm(G[i] - G[c]) ** 2 for c in idx)
            for i in range(n)
        ])
        total = d2.sum()
        if total == 0:
            # All embeddings identical — fall back to random
            idx.append(int(np.random.randint(n)))
        else:
            idx.append(int(np.random.choice(n, p=d2 / total)))
    return np.array(idx)


# ──────────────────────────────────────────────────────────────────────────────
# Register all strategies
# ──────────────────────────────────────────────────────────────────────────────

register_strategy("Random",  random_sampling)
register_strategy("Entropy", entropy_sampling)
register_strategy("Margin",  margin_sampling)
register_strategy("BALD",    bald_sampling)
register_strategy("CoreSet", coreset_sampling)
register_strategy("BADGE",   badge_sampling)
