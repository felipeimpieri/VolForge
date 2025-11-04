# ⚒️ VolForge

<img src="assets/volforge_logo.svg" width="340"/>

**Advanced stochastic modeling & option pricing lab (Black–Scholes & Heston)**  
Educational only — not investment advice.

---

### 💡 What it does

VolForge lets you simulate and price options using:
- **Black–Scholes** (analytical model with full Greeks)
- **Heston Monte Carlo** (variance reduction: antithetic & control variate)
- **CLI support**: `volforge bs | heston | greek | calibrate`
- **UI** built with **Streamlit** for instant visualization

---

### ⚙️ Quick start

```bash
git clone https://github.com/felipeimpieri/VolForge.git
cd VolForge
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
pip install streamlit pytest
streamlit run streamlit_app.py
