import streamlit as st
from volforge.utils import OptionSpec
from volforge.bs import bs_price, bs_greeks
from volforge.heston_mc import HestonParams, heston_mc_price

st.set_page_config(page_title="VolForge Option Pricer", layout="wide")
st.image("assets/volforge_logo.svg", width=320)
st.title("VolForge ⚒️ – Advanced Option Pricer (BS & Heston)")

with st.sidebar:
    st.header("Contract")
    S0 = st.number_input("Spot S0", 1.0, 1e6, 100.0)
    K  = st.number_input("Strike K", 1.0, 1e6, 100.0)
    r  = st.number_input("Risk-free r (ccy)", -0.5, 1.0, 0.02, step=0.001, format="%.3f")
    q  = st.number_input("Dividend q (ccy)", -0.5, 1.0, 0.00, step=0.001, format="%.3f")
    T  = st.number_input("Maturity T (years)", 0.001, 50.0, 1.0, step=0.01)
    call = st.selectbox("Type", ["Call", "Put"]) == "Call"
    model = st.selectbox("Model", ["Black–Scholes", "Heston (MC)"])
    if model == "Black–Scholes":
        sigma = st.number_input("Sigma (vol, annualized)", 0.0001, 5.0, 0.2, step=0.01)
    else:
        v0 = st.number_input("v0 (initial variance)", 0.0001, 5.0, 0.04, step=0.001)
        kappa = st.number_input("kappa (mean reversion)", 0.0, 50.0, 2.0, step=0.1)
        theta = st.number_input("theta (long-run variance)", 0.0001, 5.0, 0.04, step=0.001)
        xi = st.number_input("xi (vol of vol)", 0.0001, 5.0, 0.5, step=0.01)
        rho = st.number_input("rho (correlation)", -0.99, 0.99, -0.5, step=0.01)
        paths = st.number_input("MC paths", 1_000, 2_000_000, 100_000, step=10_000)
        steps = st.number_input("Steps per year", 16, 4096, 252, step=8)
        seed  = st.number_input("Seed", 0, 10_000_000, 123)
        cv_on = st.checkbox("Use GBM Control Variate", value=True)
        sigma_cv = st.number_input("CV sigma (if enabled)", 0.0001, 5.0, 0.2, step=0.01)

run = st.button("Run")
spec = OptionSpec(S0=S0, K=K, r=r, q=q, T=T, call=call)

if run:
    if model == "Black–Scholes":
        st.subheader("Black–Scholes price & Greeks")
        st.json(bs_greeks(spec, sigma))
    else:
        hp = HestonParams(v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho)
        out = heston_mc_price(
            spec, hp, n_paths=int(paths), n_steps=int(steps),
            antithetic=True, seed=int(seed), sigma_cv=(sigma_cv if cv_on else None), greek_delta=True
        )
        st.subheader("Heston Monte Carlo")
        st.json(out)
        st.caption("Price and Delta include discounting. 'stderr' are Monte Carlo standard errors.")
