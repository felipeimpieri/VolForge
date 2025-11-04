from __future__ import annotations
import csv
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from scipy.optimize import minimize
from .utils import OptionSpec
from .bs import bs_price
from .heston_mc import HestonParams, heston_mc_price

@dataclass
class MarketPoint:
    K: float
    T: float
    sigma_bs: float

def _target_prices(S0: float, r: float, q: float, call: bool, points: List[MarketPoint]) -> np.ndarray:
    return np.array([bs_price(OptionSpec(S0, mp.K, r, q, mp.T, call), mp.sigma_bs) for mp in points], dtype=float)

def load_market_csv(path: str) -> Tuple[float, float, float, bool, list[MarketPoint]]:
    with open(path, "r", newline="") as f:
        rdr = list(csv.reader(f))
    S0, r, q, call = float(rdr[1][0]), float(rdr[1][1]), float(rdr[1][2]), rdr[1][3].strip().lower() == "true"
    points = []
    for row in rdr[3:]:
        if not row: 
            continue
        K, T, sigma = float(row[0]), float(row[1]), float(row[2])
        points.append(MarketPoint(K, T, sigma))
    return S0, r, q, call, points

def calibrate_heston_from_csv(
    csv_path: str,
    init: HestonParams,
    bounds: tuple[tuple[float,float], ...] = ((1e-6, 3.0),(1e-6, 10.0),(1e-6, 3.0),(1e-6, 5.0),(-0.99,0.99)),
    n_paths: int = 80_000,
    n_steps: int = 128,
    antithetic: bool = True,
    seed: int | None = 42,
    sigma_cv: float | None = 0.2,
) -> dict:
    S0, r, q, call, points = load_market_csv(csv_path)
    target = _target_prices(S0, r, q, call, points)

    def obj(x):
        hp = HestonParams(v0=x[0], kappa=x[1], theta=x[2], xi=x[3], rho=x[4])
        preds = [heston_mc_price(OptionSpec(S0, mp.K, r, q, mp.T, call), hp, n_paths=n_paths, n_steps=n_steps, antithetic=antithetic, seed=seed, sigma_cv=sigma_cv)["price"] for mp in points]
        err = np.array(preds) - target
        return float(np.mean(err*err))

    x0 = np.array([init.v0, init.kappa, init.theta, init.xi, init.rho], dtype=float)
    res = minimize(obj, x0, bounds=bounds, method="L-BFGS-B", options={"maxiter": 30})
    x = res.x
    return {
        "success": bool(res.success),
        "message": res.message,
        "nfev": int(res.nfev),
        "heston_params": {"v0": float(x[0]), "kappa": float(x[1]), "theta": float(x[2]), "xi": float(x[3]), "rho": float(x[4])},
        "objective": float(res.fun),
    }
