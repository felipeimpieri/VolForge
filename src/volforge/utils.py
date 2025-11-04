from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class OptionSpec:
    S0: float
    K: float
    r: float
    q: float
    T: float
    call: bool = True

def payoff_terminal(ST: np.ndarray, K: float, call: bool) -> np.ndarray:
    return np.maximum(ST - K, 0.0) if call else np.maximum(K - ST, 0.0)

def ensure_rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed if seed is not None else None)
