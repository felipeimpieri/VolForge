from __future__ import annotations
import argparse, json
from .utils import OptionSpec
from .bs import bs_price, bs_greeks
from .heston_mc import HestonParams, heston_mc_price, heston_fd_greek
from .calibrate import calibrate_heston_from_csv

def _add_common(p: argparse.ArgumentParser):
    p.add_argument("--S0", type=float, required=True)
    p.add_argument("--K", type=float, required=True)
    p.add_argument("--r", type=float, required=True)
    p.add_argument("--q", type=float, default=0.0)
    p.add_argument("--T", type=float, required=True)
    p.add_argument("--put", action="store_true")
    return p

def cmd_bs(a):
    spec = OptionSpec(a.S0, a.K, a.r, a.q, a.T, call=not a.put)
    print(json.dumps(bs_greeks(spec, a.sigma) if a.greeks else {"price": bs_price(spec, a.sigma)}, indent=2))

def cmd_heston(a):
    spec = OptionSpec(a.S0, a.K, a.r, a.q, a.T, call=not a.put)
    hp = HestonParams(a.v0, a.kappa, a.theta, a.xi, a.rho)
    out = heston_mc_price(spec, hp, n_paths=a.paths, n_steps=a.steps, antithetic=not a.no_antithetic, seed=a.seed, sigma_cv=(a.sigma_cv if a.sigma_cv>0 else None))
    print(json.dumps(out, indent=2))

def cmd_greek(a):
    spec = OptionSpec(a.S0, a.K, a.r, a.q, a.T, call=not a.put)
    hp = HestonParams(a.v0, a.kappa, a.theta, a.xi, a.rho)
    g = heston_fd_greek(spec, hp, greek=a.greek, eps=a.eps, n_paths=a.paths, n_steps=a.steps, antithetic=not a.no_antithetic, seed=a.seed, sigma_cv=(a.sigma_cv if a.sigma_cv>0 else None))
    print(json.dumps({a.greek: g}, indent=2))

def cmd_calibrate(a):
    init = HestonParams(a.v0, a.kappa, a.theta, a.xi, a.rho)
    res = calibrate_heston_from_csv(a.csv, init, n_paths=a.paths, n_steps=a.steps, antithetic=not a.no_antithetic, seed=a.seed, sigma_cv=(a.sigma_cv if a.sigma_cv>0 else None))
    print(json.dumps(res, indent=2))

def main():
    ap = argparse.ArgumentParser(prog="volforge", description="VolForge – option pricing lab (BS & Heston)")
    sp = ap.add_subparsers(required=True)

    p_bs = sp.add_parser("bs", help="Black–Scholes analytic")
    _add_common(p_bs); p_bs.add_argument("--sigma", type=float, required=True); p_bs.add_argument("--greeks", action="store_true"); p_bs.set_defaults(func=cmd_bs)

    p_he = sp.add_parser("heston", help="Heston Monte Carlo")
    _add_common(p_he)
    for nm in ["v0","kappa","theta","xi","rho"]: p_he.add_argument(f"--{nm}", type=float, required=True)
    p_he.add_argument("--paths", type=int, default=100_000); p_he.add_argument("--steps", type=int, default=252)
    p_he.add_argument("--no-antithetic", action="store_true"); p_he.add_argument("--seed", type=int, default=123)
    p_he.add_argument("--sigma_cv", type=float, default=0.2)
    p_he.set_defaults(func=cmd_heston)

    p_gr = sp.add_parser("greek", help="Heston finite-difference Greek (CRN)")
    _add_common(p_gr)
    for nm in ["v0","kappa","theta","xi","rho"]: p_gr.add_argument(f"--{nm}", type=float, required=True)
    p_gr.add_argument("--greek", choices=["delta","vega_v0","rho_corr","kappa","theta","xi"], default="delta")
    p_gr.add_argument("--eps", type=float, default=1e-3); p_gr.add_argument("--paths", type=int, default=100_000)
    p_gr.add_argument("--steps", type=int, default=252); p_gr.add_argument("--no-antithetic", action="store_true")
    p_gr.add_argument("--seed", type=int, default=123); p_gr.add_argument("--sigma_cv", type=float, default=0.2)
    p_gr.set_defaults(func=cmd_greek)

    p_cal = sp.add_parser("calibrate", help="Calibrate Heston to BS surface (CSV)")
    p_cal.add_argument("csv"); p_cal.add_argument("--v0", type=float, default=0.04)
    p_cal.add_argument("--kappa", type=float, default=1.5); p_cal.add_argument("--theta", type=float, default=0.04)
    p_cal.add_argument("--xi", type=float, default=0.5); p_cal.add_argument("--rho", type=float, default=-0.5)
    p_cal.add_argument("--paths", type=int, default=80_000); p_cal.add_argument("--steps", type=int, default=128)
    p_cal.add_argument("--no-antithetic", action="store_true"); p_cal.add_argument("--seed", type=int, default=42)
    p_cal.add_argument("--sigma_cv", type=float, default=0.2)
    p_cal.set_defaults(func=cmd_calibrate)

    a = ap.parse_args(); a.func(a)

if __name__ == "__main__":
    main()
