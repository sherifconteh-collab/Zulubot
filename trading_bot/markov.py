from __future__ import annotations

"""Markov Chain market regime detection engine.

Pure computation — no I/O, no broker, no PortfolioState.
Uses numpy for matrix power and eigenvalue decomposition.

States
------
BULL     = 0  (rolling_return > bull_threshold)
BEAR     = 1  (rolling_return < bear_threshold)
SIDEWAYS = 2  (in between)

The transition matrix P is estimated by maximum-likelihood with optional
Laplace smoothing (additive constant on each cell) to regularise sparse rows.
"""

from dataclasses import dataclass

import numpy as np

BULL = 0
BEAR = 1
SIDEWAYS = 2
N_STATES = 3
STATE_NAMES: dict[int, str] = {BULL: "bull", BEAR: "bear", SIDEWAYS: "sideways"}


@dataclass(frozen=True)
class MarkovResult:
    """Full diagnostic payload returned by compute_markov_signal."""

    current_state: int
    current_state_name: str
    transition_matrix: list[list[float]]   # 3×3; rows sum to 1.0
    next_step_probs: list[float]           # P(state t+1 | current_state)
    stationary_distribution: list[float]  # long-run regime proportions
    signal: float                          # bull_prob − bear_prob ∈ [-1, 1]
    bull_prob: float
    bear_prob: float
    sideways_prob: float
    n_transitions: int                     # transitions observed in window
    history_days: int                      # price observations used


# ------------------------------------------------------------------ #
# Core helpers                                                         #
# ------------------------------------------------------------------ #

def classify_state(
    rolling_return: float,
    bull_threshold: float,
    bear_threshold: float,
) -> int:
    if rolling_return > bull_threshold:
        return BULL
    if rolling_return < bear_threshold:
        return BEAR
    return SIDEWAYS


def rolling_returns(prices: list[float], window: int) -> list[float]:
    """Compute (price[i] - price[i-window]) / price[i-window] for all valid i.

    Returns a list of length max(0, len(prices) - window).
    """
    if len(prices) < window + 1:
        return []
    result: list[float] = []
    for i in range(window, len(prices)):
        prev = prices[i - window]
        curr = prices[i]
        if prev <= 0 or curr <= 0:
            continue
        result.append((curr - prev) / prev)
    return result


def build_state_sequence(
    prices: list[float],
    *,
    window: int,
    bull_threshold: float,
    bear_threshold: float,
) -> list[int]:
    rets = rolling_returns(prices, window)
    return [classify_state(r, bull_threshold, bear_threshold) for r in rets]


def estimate_transition_matrix(
    state_sequence: list[int],
    n_states: int = N_STATES,
    laplace_alpha: float = 0.0,
) -> np.ndarray:
    """MLE transition matrix with optional Laplace smoothing.

    Returns an (n_states × n_states) float64 array where each row sums to 1.
    Rows with zero observed transitions (after smoothing) are set to uniform
    distribution regardless of laplace_alpha.
    """
    counts = np.full((n_states, n_states), laplace_alpha, dtype=np.float64)
    for i in range(len(state_sequence) - 1):
        counts[state_sequence[i], state_sequence[i + 1]] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    zero_rows = (row_sums == 0).flatten()
    matrix = counts / np.where(row_sums == 0, 1.0, row_sums)
    if zero_rows.any():
        matrix[zero_rows] = 1.0 / n_states
    return matrix


def stationary_distribution(transition_matrix: np.ndarray) -> np.ndarray:
    """Compute stationary distribution π such that πP = π via left eigenvector."""
    n = transition_matrix.shape[0]
    try:
        eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
        idx = int(np.argmin(np.abs(eigenvalues - 1.0)))
        vec = np.real(eigenvectors[:, idx])
        vec = np.abs(vec)
        total = vec.sum()
        if total <= 0:
            return np.full(n, 1.0 / n)
        return vec / total
    except np.linalg.LinAlgError:
        return np.full(n, 1.0 / n)


# ------------------------------------------------------------------ #
# Main entry point                                                     #
# ------------------------------------------------------------------ #

def compute_markov_signal(
    prices: list[float],
    *,
    return_window: int = 20,
    bull_threshold: float = 0.02,
    bear_threshold: float = -0.02,
    forecast_steps: int = 1,
    laplace_alpha: float = 0.1,
) -> MarkovResult | None:
    """Run the full Markov chain regime pipeline on a daily price series.

    Returns None when there is insufficient data.  Callers should treat
    None as "no-trade / hold".

    Parameters
    ----------
    prices:          Daily close prices, oldest first.
    return_window:   Lookback window for rolling return classification.
    bull_threshold:  Minimum rolling return to enter BULL state (e.g. 0.02).
    bear_threshold:  Maximum rolling return to enter BEAR state (e.g. -0.02).
    forecast_steps:  Chapman-Kolmogorov horizon (1 = next period).
    laplace_alpha:   Additive smoothing constant (0 = pure MLE).
    """
    if len(prices) < return_window + 2:
        return None

    state_seq = build_state_sequence(
        prices,
        window=return_window,
        bull_threshold=bull_threshold,
        bear_threshold=bear_threshold,
    )
    if len(state_seq) < 2:
        return None

    current_state = state_seq[-1]
    trans_matrix = estimate_transition_matrix(state_seq, laplace_alpha=laplace_alpha)

    if forecast_steps > 1:
        stepped = np.linalg.matrix_power(trans_matrix, forecast_steps)
    else:
        stepped = trans_matrix

    next_probs: list[float] = stepped[current_state].tolist()
    bull_prob = next_probs[BULL]
    bear_prob = next_probs[BEAR]
    sideways_prob = next_probs[SIDEWAYS]
    signal = bull_prob - bear_prob

    stat_dist: list[float] = stationary_distribution(trans_matrix).tolist()

    return MarkovResult(
        current_state=current_state,
        current_state_name=STATE_NAMES[current_state],
        transition_matrix=trans_matrix.tolist(),
        next_step_probs=next_probs,
        stationary_distribution=stat_dist,
        signal=signal,
        bull_prob=bull_prob,
        bear_prob=bear_prob,
        sideways_prob=sideways_prob,
        n_transitions=max(0, len(state_seq) - 1),
        history_days=len(prices),
    )
