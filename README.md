# VolForge ⚒️
*Advanced stochastic modeling & option pricing lab.*

![VolForge](assets/volforge_logo.svg)

## Features
- **Models**: Black–Scholes (analytic + Greeks) and Heston Monte Carlo (antithetic, full truncation).
- **Variance Reduction**: GBM control variate, antithetic variates.
- **Greeks**: BS analytics; Heston pathwise Delta + FD with CRN for `delta`, `vega_v0`, `rho_corr`, `kappa`, `theta`, `xi`.
- **Calibration**: Fit Heston to a BS implied-vol surface from CSV.
- **CLI**: `volforge bs|heston|greek|calibrate`
- **UI**: Streamlit app (`streamlit run streamlit_app.py`).

> Educational only. Not investment advice.

## Quickstart
```bash
pip install -e .
pytest -q
streamlit run streamlit_app.py
```
