from math import exp
from volforge.utils import OptionSpec
from volforge.bs import bs_price
from volforge.heston_mc import HestonParams, heston_mc_price

def test_bs_put_call_parity():
    S0=100; K=100; r=0.02; q=0.0; T=1.0; sigma=0.2
    spec_c = OptionSpec(S0, K, r, q, T, call=True)
    spec_p = OptionSpec(S0, K, r, q, T, call=False)
    c = bs_price(spec_c, sigma)
    p = bs_price(spec_p, sigma)
    assert abs((c - p) - (S0*exp(-q*T) - K*exp(-r*T))) < 1e-8

def test_heston_mc_runs():
    spec = OptionSpec(100, 100, 0.02, 0.0, 1.0, call=True)
    hp = HestonParams(0.04, 2.0, 0.04, 0.5, -0.5)
    out = heston_mc_price(spec, hp, n_paths=20000, n_steps=64, seed=123)
    assert out["price"] > 0 and out["stderr"] > 0
