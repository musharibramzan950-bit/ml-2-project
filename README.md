# 🏠 HOMEVAL — AI House Price Predictor

> A production-grade, single-file machine learning web app that predicts house prices in real time using an ensemble of 5 ML models.

**Made by [Musharib](https://linktr.ee/Musharib_)**

[![GitHub](https://img.shields.io/badge/GitHub-musharibramzan950--bit-181717?style=flat&logo=github)](https://github.com/musharibramzan950-bit)
[![Linktree](https://img.shields.io/badge/Linktree-Musharib__-43E55E?style=flat&logo=linktree)](https://linktr.ee/Musharib_)

---

## ✨ Features

- **5 ML Models trained simultaneously** — Gradient Boosting, Random Forest, Extra Trees, Ridge (Polynomial), ElasticNet
- **Auto model selection** — picks the best model by R² score automatically
- **14 input features** — sqft, bedrooms, bathrooms, lot size, neighborhood, condition, year built, and more
- **Live web UI** — interactive form with instant price prediction
- **Price range bands** — conservative / most likely / optimistic estimates
- **Model comparison dashboard** — R², MAE, RMSE, 5-fold CV for every model
- **Feature importance chart** — per-model bar chart
- **Neighborhood price index** — ranked table across 8 neighborhoods
- **Price distribution histogram** — full dataset visualization
- **Auto-installs dependencies** — just run the file, it handles everything

---

## 🚀 Quick Start

### 1. Clone or download

```bash
git clone https://github.com/musharibramzan950-bit/homeval.git
cd homeval
```

Or simply download `house_predictor.py`.

### 2. Run

```bash
python house_predictor.py
```

That's it. The script auto-installs any missing packages, trains the models, and starts the server.

### 3. Open in browser

```
http://localhost:5050
```

---

## 📦 Dependencies

Auto-installed on first run. You can also install manually:

```bash
pip install flask scikit-learn numpy pandas
```

**Python 3.8+** required.

---

## 🧠 ML Architecture

| Model | Algorithm | Notes |
|---|---|---|
| Gradient Boost | `GradientBoostingRegressor` | 300 trees, lr=0.08, depth=5 |
| Random Forest | `RandomForestRegressor` | 200 trees, parallel |
| Extra Trees | `ExtraTreesRegressor` | 200 trees, faster training |
| Ridge (Poly2) | `Pipeline` → `PolynomialFeatures` + `Ridge` | Degree-2 interactions |
| ElasticNet | `Pipeline` → `StandardScaler` + `ElasticNet` | L1+L2 regularization |

- **Dataset**: 3,000 synthetic properties with realistic noise
- **Split**: 80% train / 20% test + 5-fold cross-validation
- **Evaluation**: R², MAE, RMSE, CV-R²
- **Feature importance**: native `feature_importances_` or permutation importance fallback

---

## 🏘️ Neighborhoods

| Neighborhood | Base Price | Profile |
|---|---|---|
| Downtown | $520,000 | Dense urban, walkable |
| West Hills | $450,000 | Suburban premium, low crime |
| Old Quarter | $410,000 | Historic charm |
| Midtown | $380,000 | Mid-density, mixed use |
| University | $340,000 | Student area |
| Suburbs | $310,000 | Family-friendly |
| Riverside | $290,000 | Scenic, moderate |
| East Side | $260,000 | High density, affordable |

---

## 📁 File Structure

```
house_predictor.py   ← entire app (single file)
README.md            ← this file
```

---

## 🛠️ Configuration

| Variable | Default | Description |
|---|---|---|
| `PORT` | `5050` | Set env var `PORT` to change |
| `n` in `generate_dataset` | `3000` | Increase for better accuracy |
| `seed` | `42` | Reproducibility seed |

Change port:
```bash
PORT=8080 python house_predictor.py
```

---

## 📸 What You'll See

```
══════════════════════════════════════════════════════
  🏠  HOMEVAL — AI House Price Predictor
══════════════════════════════════════════════════════
  ⚙️  Generating dataset (3,000 properties)…
  🧠  Training 5 ML models…

     Gradient Boost        R²=0.9612  MAE=$18,204  RMSE=$24,891 ◄ BEST
     Random Forest         R²=0.9574  MAE=$19,340  RMSE=$26,113
     Extra Trees           R²=0.9531  MAE=$20,102  RMSE=$27,450
     Ridge (Poly2)         R²=0.8920  MAE=$31,400  RMSE=$42,300
     ElasticNet            R²=0.8540  MAE=$38,200  RMSE=$50,100

  🌐  Running at → http://localhost:5050
══════════════════════════════════════════════════════
```

---

## 👤 Author

**Musharib Ramzan**

- 🌐 [Linktree](https://linktr.ee/Musharib_)
- 💻 [GitHub](https://github.com/musharibramzan950-bit)

---
---

*Built with Python · Flask · scikit-learn · Chart.js*
