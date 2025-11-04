from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .utils import OptionSpec, payoff_terminal, ensure_rng

@dataclass(frozen=True)
class HestonParams:
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

def _gbm_one_step(S: np.ndarray, r: float, q: float, v: np.ndarray, dt: float, dW1: np.ndarray) -> np.ndarray:
    return S * np.exp((r - q - 0.5*v)*dt + np.sqrt(v)*np.sqrt(dt)*dW1)

def heston_mc_price(
    spec: OptionSpec,
    hp: HestonParams,
    n_paths: int = 200_000,
    n_steps: int = 252,
    antithetic: bool = True,
    seed: int | None = None,
    sigma_cv: float | None = None,
    greek_delta: bool = True,
) -> dict[str, float]:
    if n_paths <= 0 or n_steps <= 0:
        raise ValueError("n_paths and n_steps must be positive")
    if not (-0.999 < hp.rho < 0.999):
        raise ValueError("rho must be in (-0.999, 0.999)")

    rng = ensure_rng(seed)
    dt = spec.T / n_steps

    rho = hp.rho
    L = np.array([[1.0, 0.0], [rho, np.sqrt(1 - rho*rho)]], dtype=float)

    m = n_paths // 2 if antithetic else n_paths
    use_anti = antithetic

    S = np.full(m, spec.S0, dtype=float)
    v = np.full(m, hp.v0, dtype=float)

    use_cv = sigma_cv is not None
    if use_cv and sigma_cv <= 0:
        raise ValueError("sigma_cv must be positive when provided")
    if use_cv:
        S_cv = np.full(m, spec.S0, dtype=float)
        from .bs import bs_price
        gbm_mean = bs_price(spec, sigma_cv)

    for _ in range(n_steps):
        Z = rng.standard_normal((2, m), dtype=float)
        Z_all = np.concatenate([Z, -Z], axis=1) if use_anti else Z
        dW = L @ Z_all
        dW1, dW2 = dW[0], dW[1]

        if use_anti:
            S_pair = np.concatenate([S, S])
            v_pair = np.concatenate([v, v])
            if use_cv:
                S_cv_pair = np.concatenate([S_cv, S_cv])
        else:
            S_pair, v_pair = S, v
            if use_cv:
                S_cv_pair = S_cv

        v_pos = np.maximum(v_pair, 0.0)
        v_next = v_pair + hp.kappa*(hp.theta - v_pos)*dt + hp.xi*np.sqrt(v_pos)*np.sqrt(dt)*dW2
        v_next = np.maximum(v_next, 0.0)

        S_next = _gbm_one_step(S_pair, spec.r, spec.q, v_pos, dt, dW1)
        if use_cv:
            S_cv_next = _gbm_one_step(S_cv_pair, spec.r, spec.q, np.full_like(S_cv_pair, sigma_cv*sigma_cv), dt, dW1)

        if use_anti:
            S = 0.5*(S_next[:m] + S_next[m:])
            v = 0.5*(v_next[:m] + v_next[m:])
            if use_cv:
                S_cv = 0.5*(S_cv_next[:m] + S_cv_next[m:])
        else:
            S, v = S_next, v_next
            if use_cv:
                S_cv = S_cv_next

    pay = payoff_terminal(S, spec.K, spec.call)
    if use_cv:
        pay_cv = payoff_terminal(S_cv, spec.K, spec.call)
        pay = pay - (pay_cv - gbm_mean)

    disc = np.exp(-spec.r*spec.T)
    price = disc*float(np.mean(pay))
    stderr = disc*float(np.std(pay, ddof=1))/np.sqrt(m)

    out = {"price": price, "stderr": stderr}
    if greek_delta:
        ind = (S > spec.K).astype(float) if spec.call else -(S < spec.K).astype(float)
        dST_dS0 = S / spec.S0
        delta_pw = disc * np.mean(ind * dST_dS0)
        delta_se = disc * float(np.std(ind * dST_dS0, ddof=1)) / np.sqrt(m)
        out.update({"delta": float(delta_pw), "delta_stderr": float(delta_se)})
    return out

def heston_fd_greek(
    spec: OptionSpec,
    hp: HestonParams,
    greek: str = "delta",
    eps: float = 1e-3,
    n_paths: int = 200_000,
    n_steps: int = 252,
    antithetic: bool = True,
    seed: int | None = 12345,
    sigma_cv: float | None = None,
):
    def price_with_params(S0=None, v0=None, rho=None, kappa=None, theta=None, xi=None):
        sp = spec if S0 is None else OptionSpec(S0, spec.K, spec.r, spec.q, spec.T, spec.call)
        hp2 = HestonParams(
            v0 = hp.v0 if v0 is None else v0,
            kappa = hp.kappa if kappa is None else kappa,
            theta = hp.theta if theta is None else theta,
            xi = hp.xi if xi is None else xi,
            rho = hp.rho if rho is None else rho,
        )
        return heston_mc_price(sp, hp2, n_paths=n_paths, n_steps=n_steps, antithetic=antithetic, seed=seed, sigma_cv=sigma_cv, greek_delta=False)["price"]

    if greek == "delta":
        p1 = price_with_params(S0=spec.S0*(1+eps)); p2 = price_with_params(S0=spec.S0*(1-eps)); return (p1 - p2)/(2*spec.S0*eps)
    if greek == "vega_v0":
        p1 = price_with_params(v0=hp.v0*(1+eps)); p2 = price_with_params(v0=hp.v0*(1-eps)); return (p1 - p2)/(2*hp.v0*eps)
    if greek == "rho_corr":
        p1 = price_with_params(rho=hp.rho + eps); p2 = price_with_params(rho=hp.rho - eps); return (p1 - p2)/(2*eps)
    if greek == "kappa":
        p1 = price_with_params(kappa=hp.kappa*(1+eps)); p2 = price_with_params(kappa=hp.kappa*(1-eps)); return (p1 - p2)/(2*hp.kappa*eps)
    if greek == "theta":
        p1 = price_with_params(theta=hp.theta*(1+eps)); p2 = price_with_params(theta=hp.theta*(1-eps)); return (p1 - p2)/(2*hp.theta*eps)
    if greek == "xi":
        p1 = price_with_params(xi=hp.xi*(1+eps)); p2 = price_with_params(xi=hp.xi*(1-eps)); return (p1 - p2)/(2*hp.xi*eps)
    raise ValueError(f"Unknown greek '{greek}'")
