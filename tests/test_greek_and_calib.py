from volforge.utils import OptionSpec
from volforge.heston_mc import HestonParams, heston_fd_greek

def test_fd_delta_smoke():
    spec = OptionSpec(100, 100, 0.02, 0.0, 1.0, call=True)
    hp = HestonParams(0.04, 2.0, 0.04, 0.5, -0.5)
    d = heston_fd_greek(spec, hp, greek="delta", eps=1e-3, n_paths=10000, n_steps=32, seed=1)
    assert d == d  # not NaN
