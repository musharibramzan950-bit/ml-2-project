#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║         HOMEVAL — AI House Price Predictor               ║
║         Single-file · Run · Done                         ║
╚══════════════════════════════════════════════════════════╝

  pip install flask scikit-learn numpy pandas
  python house_predictor.py
  → http://localhost:5050
"""

# ─── AUTO-INSTALL DEPENDENCIES ───────────────────────────────────────────────
import subprocess, sys

REQUIRED = ["flask", "scikit-learn", "numpy", "pandas"]

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

for pkg in REQUIRED:
    try:
        __import__(pkg.replace("-", "_").split("[")[0])
    except ImportError:
        print(f"  📦 Installing {pkg}...")
        install(pkg)

# ─── IMPORTS ─────────────────────────────────────────────────────────────────
import json, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

from sklearn.ensemble import (
    GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
)
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.inspection import permutation_importance

# ─── DATA GENERATION ─────────────────────────────────────────────────────────

def generate_dataset(n=3000, seed=42):
    rng = np.random.default_rng(seed)

    neighborhoods = {
        "Downtown":       dict(base=520_000, sqft_mul=1.4, crime=0.2),
        "Midtown":        dict(base=380_000, sqft_mul=1.1, crime=0.3),
        "Suburbs":        dict(base=310_000, sqft_mul=1.0, crime=0.15),
        "East Side":      dict(base=260_000, sqft_mul=0.85,crime=0.45),
        "West Hills":     dict(base=450_000, sqft_mul=1.25,crime=0.1),
        "Riverside":      dict(base=290_000, sqft_mul=0.95,crime=0.25),
        "University":     dict(base=340_000, sqft_mul=1.05,crime=0.35),
        "Old Quarter":    dict(base=410_000, sqft_mul=1.15,crime=0.2),
    }
    n_labels   = list(neighborhoods.keys())
    n_choice   = rng.choice(len(n_labels), size=n)

    sqft        = rng.integers(600, 5500, size=n).astype(float)
    bedrooms    = rng.integers(1, 7, size=n).astype(float)
    bathrooms   = np.clip(rng.normal(bedrooms * 0.65, 0.5, n).round(1), 1, bedrooms + 1)
    floors      = rng.choice([1, 2, 3], size=n, p=[0.45, 0.42, 0.13]).astype(float)
    garage      = rng.integers(0, 4, size=n).astype(float)
    pool        = rng.choice([0, 1], size=n, p=[0.7, 0.3]).astype(float)
    year_built  = rng.integers(1920, 2024, size=n).astype(float)
    lot_size    = rng.integers(2000, 25000, size=n).astype(float)
    school_dist = np.clip(rng.normal(3.5, 1.5, n), 0.5, 6.0)
    renovation  = rng.choice([0, 1], size=n, p=[0.6, 0.4]).astype(float)
    fireplace   = rng.choice([0, 1], size=n, p=[0.55, 0.45]).astype(float)
    basement    = rng.choice([0, 1], size=n, p=[0.5, 0.5]).astype(float)
    condition   = rng.integers(1, 6, size=n).astype(float)

    price = np.zeros(n)
    for i, ni in enumerate(n_choice):
        nb   = neighborhoods[n_labels[ni]]
        age  = 2024 - year_built[i]
        p = (
            nb["base"]
            + sqft[i]       * nb["sqft_mul"] * 180
            + bedrooms[i]   * 15_000
            + bathrooms[i]  * 22_000
            + floors[i]     * 8_000
            + garage[i]     * 14_000
            + pool[i]       * 35_000
            - age           * 900
            + lot_size[i]   * 3.5
            - school_dist[i]* 12_000
            + renovation[i] * 28_000
            + fireplace[i]  * 8_500
            + basement[i]   * 18_000
            + condition[i]  * 14_000
            - nb["crime"]   * 80_000
        )
        price[i] = max(p * rng.normal(1.0, 0.06), 80_000)

    df = pd.DataFrame({
        "sqft":        sqft,
        "bedrooms":    bedrooms,
        "bathrooms":   bathrooms,
        "floors":      floors,
        "garage":      garage,
        "pool":        pool,
        "year_built":  year_built,
        "lot_size":    lot_size,
        "school_dist": school_dist,
        "renovation":  renovation,
        "fireplace":   fireplace,
        "basement":    basement,
        "condition":   condition,
        "neighborhood":n_choice,
        "price":       price.round(0),
    })
    return df

# ─── MODEL TRAINING ──────────────────────────────────────────────────────────

FEATURE_COLS = [
    "sqft","bedrooms","bathrooms","floors","garage","pool",
    "year_built","lot_size","school_dist","renovation",
    "fireplace","basement","condition","neighborhood",
]

class ModelHub:
    def __init__(self):
        self.models   = {}
        self.metrics  = {}
        self.best     = None
        self.feat_imp = {}
        self.scaler   = None

    def train(self, df):
        X = df[FEATURE_COLS].values
        y = df["price"].values

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        candidates = {
            "Gradient Boost": GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.08, max_depth=5,
                subsample=0.85, min_samples_leaf=5, random_state=42),
            "Random Forest":  RandomForestRegressor(
                n_estimators=200, max_depth=20, min_samples_leaf=3,
                n_jobs=-1, random_state=42),
            "Extra Trees":    ExtraTreesRegressor(
                n_estimators=200, max_depth=None, min_samples_leaf=2,
                n_jobs=-1, random_state=42),
            "Ridge (Poly2)":  Pipeline([
                ("sc",  StandardScaler()),
                ("pf",  PolynomialFeatures(2, interaction_only=True, include_bias=False)),
                ("mdl", Ridge(alpha=10)),
            ]),
            "ElasticNet":     Pipeline([
                ("sc",  StandardScaler()),
                ("mdl", ElasticNet(alpha=500, l1_ratio=0.5, max_iter=5000)),
            ]),
        }

        best_r2, best_name = -np.inf, None

        for name, mdl in candidates.items():
            mdl.fit(X_tr, y_tr)
            y_pred = mdl.predict(X_te)
            r2  = r2_score(y_te, y_pred)
            mae = mean_absolute_error(y_te, y_pred)
            rmse= np.sqrt(mean_squared_error(y_te, y_pred))
            cv  = cross_val_score(mdl, X_tr, y_tr, cv=5, scoring="r2", n_jobs=-1).mean()

            self.models[name]  = mdl
            self.metrics[name] = dict(r2=round(r2,4), mae=round(mae,0),
                                      rmse=round(rmse,0), cv_r2=round(cv,4))

            # feature importance
            if hasattr(mdl, "feature_importances_"):
                fi = mdl.feature_importances_
            else:
                fi = permutation_importance(mdl, X_te, y_te, n_repeats=5,
                                            random_state=42).importances_mean
                fi = np.clip(fi, 0, None)
                fi = fi / fi.sum() if fi.sum() > 0 else fi
            self.feat_imp[name] = dict(zip(FEATURE_COLS, fi.round(4).tolist()))

            if r2 > best_r2:
                best_r2, best_name = r2, name

        self.best = best_name
        print(f"\n  ✅ Best model: {best_name}  (R² = {best_r2:.4f})")
        for n, m in self.metrics.items():
            marker = " ◄ BEST" if n == self.best else ""
            print(f"     {n:20s}  R²={m['r2']:.4f}  MAE=${m['mae']:,.0f}  RMSE=${m['rmse']:,.0f}{marker}")

    def predict(self, features: dict, model_name: str = None):
        name = model_name or self.best
        mdl  = self.models[name]
        x    = np.array([[features[c] for c in FEATURE_COLS]])
        pred = mdl.predict(x)[0]
        return max(pred, 50_000), name

# ─── HTML TEMPLATE ───────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HOMEVAL — AI Price Oracle</title>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root{
  --bg:#0a0a0f;
  --surface:#12121a;
  --card:#1a1a26;
  --border:#2a2a3f;
  --accent:#e8c547;
  --accent2:#4fc3f7;
  --accent3:#ef5350;
  --text:#f0eff8;
  --muted:#7a7994;
  --gold:linear-gradient(135deg,#f6d365,#fda085);
  --radius:16px;
}
*{margin:0;padding:0;box-sizing:border-box}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--text);font-family:'DM Sans',sans-serif;min-height:100vh;overflow-x:hidden}

/* Background */
body::before{
  content:'';position:fixed;inset:0;
  background:radial-gradient(ellipse 80% 60% at 20% -10%,rgba(78,62,255,.12),transparent),
             radial-gradient(ellipse 60% 50% at 80% 100%,rgba(232,197,71,.07),transparent);
  pointer-events:none;z-index:0
}

header{
  position:sticky;top:0;z-index:100;
  backdrop-filter:blur(20px);background:rgba(10,10,15,.85);
  border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;
  padding:18px 48px
}
.logo{font-family:'Playfair Display',serif;font-size:1.6rem;font-weight:900;letter-spacing:-.02em}
.logo span{color:var(--accent)}
.badge{font-family:'DM Mono',monospace;font-size:.7rem;background:rgba(232,197,71,.12);
  color:var(--accent);padding:4px 12px;border-radius:99px;border:1px solid rgba(232,197,71,.3)}

main{position:relative;z-index:1;max-width:1400px;margin:0 auto;padding:56px 32px 80px}

.hero{text-align:center;margin-bottom:72px}
.hero h1{font-family:'Playfair Display',serif;font-size:clamp(2.8rem,6vw,5rem);
  font-weight:900;line-height:1.05;letter-spacing:-.03em;
  background:linear-gradient(135deg,#f0eff8 0%,#c8c5f0 40%,#e8c547 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.hero p{margin-top:20px;font-size:1.1rem;color:var(--muted);max-width:560px;
  margin-inline:auto;line-height:1.7;font-weight:300}
.model-pill{display:inline-flex;align-items:center;gap:8px;margin-top:28px;
  background:var(--card);border:1px solid var(--border);
  border-radius:99px;padding:8px 20px;font-size:.85rem}
.model-pill .dot{width:8px;height:8px;border-radius:50%;background:#4caf50;
  box-shadow:0 0 8px #4caf50;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

.grid{display:grid;grid-template-columns:1fr 1fr;gap:28px;align-items:start}
@media(max-width:960px){.grid{grid-template-columns:1fr}}

.card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden}
.card-header{padding:24px 28px 0;display:flex;align-items:center;gap:12px;margin-bottom:20px}
.card-header h2{font-family:'Playfair Display',serif;font-size:1.25rem;font-weight:700}
.card-icon{width:36px;height:36px;border-radius:10px;display:grid;place-items:center;font-size:1rem;flex-shrink:0}
.card-body{padding:0 28px 28px}

/* Form */
.form-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.field{display:flex;flex-direction:column;gap:6px}
.field.full{grid-column:1/-1}
label{font-size:.75rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
input,select{
  background:#0f0f18;border:1px solid var(--border);border-radius:10px;
  color:var(--text);font-family:'DM Sans',sans-serif;font-size:.95rem;
  padding:10px 14px;transition:.2s;outline:none;width:100%
}
input:focus,select:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(232,197,71,.12)}
select option{background:#1a1a26}

.toggle-row{display:flex;gap:10px;flex-wrap:wrap}
.toggle{position:relative}
.toggle input{position:absolute;opacity:0;width:0;height:0}
.toggle label{
  cursor:pointer;padding:8px 16px;border-radius:8px;font-size:.85rem;font-weight:500;
  border:1px solid var(--border);background:#0f0f18;letter-spacing:normal;
  text-transform:none;color:var(--muted);transition:.2s;display:block
}
.toggle input:checked + label{background:rgba(232,197,71,.15);border-color:var(--accent);color:var(--accent)}

.model-select{margin-top:18px}
.model-select label{display:block;margin-bottom:8px}

.predict-btn{
  width:100%;margin-top:20px;padding:15px;
  background:var(--accent);color:#0a0a0f;
  border:none;border-radius:10px;font-family:'DM Sans',sans-serif;
  font-size:1rem;font-weight:700;letter-spacing:.02em;cursor:pointer;
  transition:.2s;position:relative;overflow:hidden
}
.predict-btn:hover{transform:translateY(-1px);box-shadow:0 8px 24px rgba(232,197,71,.35)}
.predict-btn:active{transform:translateY(0)}
.predict-btn.loading{opacity:.7;pointer-events:none}

/* Result */
.result-box{
  background:linear-gradient(135deg,rgba(232,197,71,.08),rgba(78,62,255,.05));
  border:1px solid rgba(232,197,71,.25);border-radius:12px;
  padding:28px;text-align:center;margin-top:20px;
  display:none;animation:slideUp .4s cubic-bezier(.22,1,.36,1)
}
@keyframes slideUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
.result-box.show{display:block}
.result-label{font-size:.75rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}
.result-price{
  font-family:'Playfair Display',serif;font-size:3.2rem;font-weight:900;
  background:var(--gold);-webkit-background-clip:text;-webkit-text-fill-color:transparent;
  background-clip:text;line-height:1;margin:8px 0
}
.result-model{font-size:.8rem;color:var(--muted);font-family:'DM Mono',monospace}
.result-range{display:flex;gap:16px;justify-content:center;margin-top:16px}
.range-item{background:rgba(255,255,255,.04);border-radius:8px;padding:10px 18px}
.range-item .rv{font-weight:700;font-size:.95rem}
.range-item .rl{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}

/* Metrics */
.metrics-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}
.metric-card{background:#0f0f18;border:1px solid var(--border);border-radius:10px;padding:16px}
.metric-name{font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin-bottom:4px}
.metric-val{font-family:'DM Mono',monospace;font-size:1.15rem;font-weight:500}
.metric-val.good{color:#4caf50}
.metric-val.warn{color:var(--accent)}

.model-tabs{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:20px}
.tab-btn{
  padding:6px 14px;border-radius:8px;border:1px solid var(--border);
  background:transparent;color:var(--muted);font-size:.8rem;cursor:pointer;
  font-family:'DM Sans',sans-serif;transition:.15s
}
.tab-btn.active{background:rgba(79,195,247,.12);border-color:var(--accent2);color:var(--accent2)}

.chart-wrap{position:relative;height:240px}

/* Neighborhood table */
.n-table{width:100%;border-collapse:collapse;font-size:.82rem}
.n-table th{text-align:left;padding:8px 10px;font-size:.7rem;font-weight:600;
  text-transform:uppercase;letter-spacing:.08em;color:var(--muted);
  border-bottom:1px solid var(--border)}
.n-table td{padding:9px 10px;border-bottom:1px solid rgba(255,255,255,.04)}
.n-table tr:last-child td{border-bottom:none}
.n-table td:last-child{font-family:'DM Mono',monospace;color:var(--accent);font-weight:500}
.n-bar{height:6px;border-radius:3px;background:rgba(232,197,71,.15);margin-top:4px}
.n-bar-fill{height:100%;border-radius:3px;background:var(--accent)}

footer{text-align:center;padding:40px;color:var(--muted);font-size:.8rem;
  border-top:1px solid var(--border);margin-top:60px}
.tag{background:rgba(255,255,255,.05);border-radius:6px;padding:2px 8px;font-family:'DM Mono',monospace}
</style>
</head>
<body>

<header>
  <div class="logo">HOME<span>VAL</span></div>
  <div style="display:flex;align-items:center;gap:14px">
    <span class="badge">AI PRICE ORACLE v2.0</span>
    <span style="font-size:.82rem;color:var(--muted);font-weight:500">by <a href="https://linktr.ee/Musharib_" target="_blank" style="color:var(--accent);text-decoration:none;font-weight:600">Musharib</a></span>
  </div>
</header>

<main>
  <div class="hero">
    <h1>Predict Property Value<br>with Machine Learning</h1>
    <p>Trained on 3,000 synthetic properties across 8 neighborhoods using an ensemble of 5 ML models. Instant. Accurate. Local.</p>
    <div class="model-pill"><span class="dot"></span><span id="bestModelLabel">Loading models…</span></div>
  </div>

  <div class="grid">

    <!-- LEFT: Predictor Form -->
    <div class="card">
      <div class="card-header">
        <div class="card-icon" style="background:rgba(232,197,71,.12)">🏠</div>
        <h2>Property Details</h2>
      </div>
      <div class="card-body">
        <div class="form-grid">
          <div class="field">
            <label>Living Area (sqft)</label>
            <input type="number" id="sqft" value="1850" min="300" max="10000">
          </div>
          <div class="field">
            <label>Lot Size (sqft)</label>
            <input type="number" id="lot_size" value="6500" min="1000" max="40000">
          </div>
          <div class="field">
            <label>Bedrooms</label>
            <input type="number" id="bedrooms" value="3" min="1" max="10">
          </div>
          <div class="field">
            <label>Bathrooms</label>
            <input type="number" id="bathrooms" value="2" min="1" max="10" step="0.5">
          </div>
          <div class="field">
            <label>Floors</label>
            <select id="floors"><option value="1">1 Floor</option><option value="2" selected>2 Floors</option><option value="3">3 Floors</option></select>
          </div>
          <div class="field">
            <label>Garage Spaces</label>
            <select id="garage"><option value="0">None</option><option value="1">1 Car</option><option value="2" selected>2 Cars</option><option value="3">3 Cars</option></select>
          </div>
          <div class="field">
            <label>Year Built</label>
            <input type="number" id="year_built" value="1998" min="1900" max="2024">
          </div>
          <div class="field">
            <label>Condition (1–5)</label>
            <select id="condition">
              <option value="1">1 — Poor</option><option value="2">2 — Fair</option>
              <option value="3" selected>3 — Average</option><option value="4">4 — Good</option>
              <option value="5">5 — Excellent</option>
            </select>
          </div>
          <div class="field">
            <label>School Distance (km)</label>
            <input type="number" id="school_dist" value="2.5" min="0.1" max="10" step="0.1">
          </div>
          <div class="field">
            <label>Neighborhood</label>
            <select id="neighborhood">
              <option value="0">Downtown</option><option value="1">Midtown</option>
              <option value="2" selected>Suburbs</option><option value="3">East Side</option>
              <option value="4">West Hills</option><option value="5">Riverside</option>
              <option value="6">University</option><option value="7">Old Quarter</option>
            </select>
          </div>
          <div class="field full">
            <label>Features</label>
            <div class="toggle-row">
              <div class="toggle"><input type="checkbox" id="pool" name="pool"><label for="pool">🏊 Pool</label></div>
              <div class="toggle"><input type="checkbox" id="renovation" name="renovation" checked><label for="renovation">🔨 Renovated</label></div>
              <div class="toggle"><input type="checkbox" id="fireplace" name="fireplace" checked><label for="fireplace">🔥 Fireplace</label></div>
              <div class="toggle"><input type="checkbox" id="basement" name="basement"><label for="basement">⬇️ Basement</label></div>
            </div>
          </div>
          <div class="field full model-select">
            <label>Model Override</label>
            <select id="modelSelect"><option value="">Use Best Model (auto)</option></select>
          </div>
        </div>
        <button class="predict-btn" onclick="predict()">⚡ Predict Price</button>

        <div class="result-box" id="resultBox">
          <div class="result-label">Estimated Market Value</div>
          <div class="result-price" id="resultPrice">—</div>
          <div class="result-model" id="resultModelName"></div>
          <div class="result-range">
            <div class="range-item"><div class="rv" id="rangeLow">—</div><div class="rl">Conservative</div></div>
            <div class="range-item"><div class="rv" id="rangeMid">—</div><div class="rl">Most Likely</div></div>
            <div class="range-item"><div class="rv" id="rangeHigh">—</div><div class="rl">Optimistic</div></div>
          </div>
        </div>
      </div>
    </div>

    <!-- RIGHT COLUMN -->
    <div style="display:flex;flex-direction:column;gap:28px">

      <!-- Model Performance -->
      <div class="card">
        <div class="card-header">
          <div class="card-icon" style="background:rgba(79,195,247,.1)">📊</div>
          <h2>Model Performance</h2>
        </div>
        <div class="card-body">
          <div class="model-tabs" id="modelTabs"></div>
          <div class="metrics-grid" id="metricsGrid"></div>
        </div>
      </div>

      <!-- Feature Importance -->
      <div class="card">
        <div class="card-header">
          <div class="card-icon" style="background:rgba(232,197,71,.1)">🔍</div>
          <h2>Feature Importance</h2>
        </div>
        <div class="card-body">
          <div class="chart-wrap"><canvas id="featureChart"></canvas></div>
        </div>
      </div>

      <!-- Neighborhood Comparison -->
      <div class="card">
        <div class="card-header">
          <div class="card-icon" style="background:rgba(239,83,80,.1)">📍</div>
          <h2>Neighborhood Index</h2>
        </div>
        <div class="card-body">
          <table class="n-table">
            <thead><tr><th>Neighborhood</th><th>Avg. Price</th></tr></thead>
            <tbody id="neighborhoodTable"></tbody>
          </table>
        </div>
      </div>

    </div>
  </div>

  <!-- Price Distribution -->
  <div class="card" style="margin-top:28px">
    <div class="card-header">
      <div class="card-icon" style="background:rgba(156,39,176,.1)">📈</div>
      <h2>Price Distribution Across Dataset</h2>
    </div>
    <div class="card-body">
      <div style="position:relative;height:260px"><canvas id="distChart"></canvas></div>
    </div>
  </div>
</main>

<footer>
  <div style="margin-bottom:18px">
    <div style="font-family:'Playfair Display',serif;font-size:1.4rem;font-weight:700;margin-bottom:6px">
      Made with ❤️ by <span style="color:var(--accent)">Musharib</span>
    </div>
    <div style="display:flex;justify-content:center;gap:16px;margin-top:14px;flex-wrap:wrap">
      <a href="https://github.com/musharibramzan950-bit" target="_blank"
         style="display:inline-flex;align-items:center;gap:8px;padding:10px 22px;
                border-radius:99px;background:#1a1a26;border:1px solid #2a2a3f;
                color:#f0eff8;text-decoration:none;font-size:.88rem;font-weight:600;
                transition:.2s"
         onmouseover="this.style.borderColor='#e8c547';this.style.color='#e8c547'"
         onmouseout="this.style.borderColor='#2a2a3f';this.style.color='#f0eff8'">
        <svg height="18" width="18" viewBox="0 0 24 24" fill="currentColor">
          <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/>
        </svg>
        GitHub
      </a>
      <a href="https://linktr.ee/Musharib_" target="_blank"
         style="display:inline-flex;align-items:center;gap:8px;padding:10px 22px;
                border-radius:99px;background:rgba(232,197,71,.1);border:1px solid rgba(232,197,71,.35);
                color:#e8c547;text-decoration:none;font-size:.88rem;font-weight:600;
                transition:.2s"
         onmouseover="this.style.background='rgba(232,197,71,.2)'"
         onmouseout="this.style.background='rgba(232,197,71,.1)'">
        <svg height="18" width="18" viewBox="0 0 24 24" fill="currentColor">
          <path d="M7.953 15.066c-.08.163-.08.324-.08.486C7.873 17.482 9.553 19 11.64 19s3.85-1.518 3.85-3.448c0-.162 0-.323-.08-.486H7.954zM4 9.228l1.698 1.926L4 13.08h4.494L12 9.228 8.494 5.376H4L5.698 7.3 4 9.228zm16 0L18.302 7.3 20 5.376h-4.494L12 9.228l3.506 3.852H20l-1.698-1.926L20 9.228z"/>
        </svg>
        Linktree
      </a>
    </div>
  </div>
  <div style="border-top:1px solid #2a2a3f;padding-top:18px;margin-top:10px">
    Built with <span class="tag">scikit-learn</span> <span class="tag">Flask</span> <span class="tag">Chart.js</span>
    &nbsp;·&nbsp; 3,000 training samples &nbsp;·&nbsp; 5 ML models &nbsp;·&nbsp; 14 features
  </div>
</footer>

<script>
let allMetrics={}, allFeatImp={}, bestModel='', featureChart=null, distChart=null;
let activeTab='';

const fmt=v=>'$'+Math.round(v).toLocaleString();

async function init(){
  const r=await fetch('/api/info');
  const d=await r.json();
  allMetrics=d.metrics; allFeatImp=d.feat_imp; bestModel=d.best;

  document.getElementById('bestModelLabel').textContent=`Best: ${bestModel} · R² ${d.metrics[bestModel].r2}`;

  // populate model select
  const sel=document.getElementById('modelSelect');
  Object.keys(allMetrics).forEach(n=>{
    const o=document.createElement('option');
    o.value=n; o.textContent=n+' (R² '+allMetrics[n].r2+')'; sel.appendChild(o);
  });

  // tabs
  const tabs=document.getElementById('modelTabs');
  Object.keys(allMetrics).forEach(n=>{
    const b=document.createElement('button');
    b.className='tab-btn'+(n===bestModel?' active':'');
    b.textContent=n+(n===bestModel?' ★':'');
    b.onclick=()=>switchTab(n);
    tabs.appendChild(b);
  });
  activeTab=bestModel;
  renderMetrics(bestModel);
  renderFeatureChart(bestModel);
  renderDistChart(d.dist);
  renderNeighborhood(d.neighborhoods);
}

function switchTab(name){
  activeTab=name;
  document.querySelectorAll('.tab-btn').forEach((b,i)=>{
    b.classList.toggle('active', Object.keys(allMetrics)[i]===name);
  });
  renderMetrics(name);
  renderFeatureChart(name);
}

function renderMetrics(name){
  const m=allMetrics[name];
  const g=document.getElementById('metricsGrid');
  g.innerHTML=`
    <div class="metric-card"><div class="metric-name">R² Score</div><div class="metric-val good">${m.r2}</div></div>
    <div class="metric-card"><div class="metric-name">CV R² (5-fold)</div><div class="metric-val good">${m.cv_r2}</div></div>
    <div class="metric-card"><div class="metric-name">MAE</div><div class="metric-val warn">${fmt(m.mae)}</div></div>
    <div class="metric-card"><div class="metric-name">RMSE</div><div class="metric-val warn">${fmt(m.rmse)}</div></div>
  `;
}

function renderFeatureChart(name){
  const fi=allFeatImp[name];
  const labels=Object.keys(fi);
  const vals=Object.values(fi);
  const paired=labels.map((l,i)=>({l,v:vals[i]})).sort((a,b)=>b.v-a.v);

  if(featureChart) featureChart.destroy();
  const ctx=document.getElementById('featureChart').getContext('2d');
  featureChart=new Chart(ctx,{
    type:'bar',
    data:{
      labels:paired.map(p=>p.l),
      datasets:[{
        data:paired.map(p=>p.v),
        backgroundColor:paired.map((_,i)=>`hsla(${45+i*18},80%,60%,0.75)`),
        borderRadius:6,borderSkipped:false
      }]
    },
    options:{
      indexAxis:'y',responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>(c.raw*100).toFixed(1)+'%'}}},
      scales:{
        x:{grid:{color:'rgba(255,255,255,.05)'},ticks:{color:'#7a7994',callback:v=>(v*100).toFixed(0)+'%'}},
        y:{grid:{display:false},ticks:{color:'#c8c5f0',font:{size:11}}}
      }
    }
  });
}

function renderDistChart(dist){
  if(distChart) distChart.destroy();
  const ctx=document.getElementById('distChart').getContext('2d');
  distChart=new Chart(ctx,{
    type:'bar',
    data:{
      labels:dist.labels,
      datasets:[{
        label:'Properties',data:dist.counts,
        backgroundColor:'rgba(232,197,71,0.55)',
        borderColor:'rgba(232,197,71,0.9)',
        borderWidth:1,borderRadius:4
      }]
    },
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false}},
      scales:{
        x:{grid:{display:false},ticks:{color:'#7a7994',maxRotation:45}},
        y:{grid:{color:'rgba(255,255,255,.05)'},ticks:{color:'#7a7994'}}
      }
    }
  });
}

function renderNeighborhood(data){
  const max=Math.max(...data.map(d=>d.avg));
  const tbody=document.getElementById('neighborhoodTable');
  tbody.innerHTML=data.map(d=>`
    <tr>
      <td>${d.name}<div class="n-bar"><div class="n-bar-fill" style="width:${(d.avg/max*100).toFixed(0)}%"></div></div></td>
      <td>${fmt(d.avg)}</td>
    </tr>`).join('');
}

async function predict(){
  const btn=document.querySelector('.predict-btn');
  btn.classList.add('loading'); btn.textContent='⏳ Predicting…';

  const payload={
    sqft:+document.getElementById('sqft').value,
    bedrooms:+document.getElementById('bedrooms').value,
    bathrooms:+document.getElementById('bathrooms').value,
    floors:+document.getElementById('floors').value,
    garage:+document.getElementById('garage').value,
    pool:document.getElementById('pool').checked?1:0,
    year_built:+document.getElementById('year_built').value,
    lot_size:+document.getElementById('lot_size').value,
    school_dist:+document.getElementById('school_dist').value,
    renovation:document.getElementById('renovation').checked?1:0,
    fireplace:document.getElementById('fireplace').checked?1:0,
    basement:document.getElementById('basement').checked?1:0,
    condition:+document.getElementById('condition').value,
    neighborhood:+document.getElementById('neighborhood').value,
    model:document.getElementById('modelSelect').value||null,
  };

  const r=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const d=await r.json();

  btn.classList.remove('loading'); btn.textContent='⚡ Predict Price';

  const box=document.getElementById('resultBox');
  box.classList.remove('show');
  void box.offsetWidth;
  box.classList.add('show');

  document.getElementById('resultPrice').textContent=fmt(d.price);
  document.getElementById('resultModelName').textContent='via '+d.model;
  document.getElementById('rangeLow').textContent=fmt(d.price*0.92);
  document.getElementById('rangeMid').textContent=fmt(d.price);
  document.getElementById('rangeHigh').textContent=fmt(d.price*1.08);
}

init();
</script>
</body>
</html>
"""

# ─── FLASK APP ────────────────────────────────────────────────────────────────

app  = Flask(__name__)
hub  = ModelHub()

NEIGHBORHOODS = ["Downtown","Midtown","Suburbs","East Side",
                 "West Hills","Riverside","University","Old Quarter"]

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/api/info")
def api_info():
    dist_data = []
    prices     = hub._df["price"].values
    bins       = np.linspace(prices.min(), prices.max(), 18)
    counts, edges = np.histogram(prices, bins=bins)
    labels = [f"${int(e/1000)}k" for e in edges[:-1]]

    neighborhoods_avg = []
    for i, name in enumerate(NEIGHBORHOODS):
        mask = hub._df["neighborhood"] == i
        avg  = hub._df.loc[mask, "price"].mean()
        neighborhoods_avg.append({"name": name, "avg": round(avg)})

    return jsonify({
        "best":    hub.best,
        "metrics": hub.metrics,
        "feat_imp":hub.feat_imp,
        "dist":    {"labels": labels, "counts": counts.tolist()},
        "neighborhoods": sorted(neighborhoods_avg, key=lambda x: -x["avg"]),
    })

@app.route("/api/predict", methods=["POST"])
def api_predict():
    body  = request.get_json(force=True)
    model = body.pop("model", None) or hub.best
    price, used = hub.predict(body, model)
    return jsonify({"price": round(price), "model": used})

# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 5050))

    print("\n" + "═"*54)
    print("  🏠  HOMEVAL — AI House Price Predictor")
    print("═"*54)
    print("  ⚙️  Generating dataset (3,000 properties)…")
    df = generate_dataset(n=3000)

    print("  🧠  Training 5 ML models…")
    hub._df = df
    hub.train(df)

    print(f"\n  🌐  Running at → http://localhost:{PORT}")
    print("  Press Ctrl+C to stop\n" + "═"*54 + "\n")
    app.run(host="0.0.0.0", port=PORT, debug=False)
