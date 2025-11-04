from __future__ import annotations
import numpy as np
from scipy.stats import norm
from .utils import OptionSpec

def _d1_d2(S0: float, K: float, r: float, q: float, sigma: float, T: float):
    if sigma <= 0 or T <= 0:
        raise ValueError("sigma and T must be positive")
    volT = sigma*np.sqrt(T)
    m = np.log(S0/K) + (r - q + 0.5*sigma*sigma)*T
    d1 = m/volT
    d2 = d1 - volT
    return d1, d2

def bs_price(spec: OptionSpec, sigma: float) -> float:
    d1, d2 = _d1_d2(spec.S0, spec.K, spec.r, spec.q, sigma, spec.T)
    df_r = np.exp(-spec.r*spec.T)
    df_q = np.exp(-spec.q*spec.T)
    if spec.call:
        return df_q*spec.S0*norm.cdf(d1) - df_r*spec.K*norm.cdf(d2)
    return df_r*spec.K*norm.cdf(-d2) - df_q*spec.S0*norm.cdf(-d1)

def bs_greeks(spec: OptionSpec, sigma: float) -> dict[str, float]:
    d1, d2 = _d1_d2(spec.S0, spec.K, spec.r, spec.q, sigma, spec.T)
    df_r = np.exp(-spec.r*spec.T); df_q = np.exp(-spec.q*spec.T)
    pdf_d1 = norm.pdf(d1); sign = 1.0 if spec.call else -1.0
    delta = sign * df_q*norm.cdf(sign*d1)
    gamma = df_q*pdf_d1/(spec.S0*sigma*np.sqrt(spec.T))
    vega  = spec.S0*df_q*pdf_d1*np.sqrt(spec.T)
    theta = (-spec.S0*df_q*pdf_d1*sigma/(2*np.sqrt(spec.T))
             - sign*(spec.r*df_r*spec.K*norm.cdf(sign*d2) - spec.q*df_q*spec.S0*norm.cdf(sign*d1)))
    rho   = sign*spec.K*spec.T*df_r*norm.cdf(sign*d2)
    return {"price": bs_price(spec, sigma), "delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
