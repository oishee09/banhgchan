# 🦈 Shark Tank India — Startup Valuation Predictor

> *A production-grade Machine Learning web application that predicts Shark Tank India deal valuations using Multiple Linear Regression, trained on real pitch data from Seasons 1–3.*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Model](https://img.shields.io/badge/Model-OLS%20Regression-teal)](https://www.statsmodels.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Live Demo

**[ STARTUP VALUATION PREDICTOR]([[https://your-app-name.streamlit.app](https://shark-tank-india-valuation-w2cqtqtgfsnkyjtbe9fwrh.streamlit.app/))]**


---

## 📖 Project Overview

This project answers a real business question:

> **"Given a startup's financials and business profile, what valuation would Shark Tank India investors offer for a deal?"**

Using a curated dataset of 423 Shark Tank India pitches (after cleaning and outlier removal), a Multiple Linear Regression model was trained to predict deal valuations. The final app lets anyone enter their startup's metrics and receive an instant, model-backed valuation estimate.

### Key Features
- **Interactive Valuation Engine** — 3-column input form with real-time implied valuation calculator
- **Pitch Analytics Dashboard** — benchmark charts comparing your startup against 18 industry medians
- **Startup Health Radar** — 7-dimension radar chart and gauge scoring your business health
- **Prediction History Log** — every prediction saved to CSV for cross-session analysis
- **Shark Animation** — custom SVG shark swim + deal animation on every successful prediction
- **Dark Navy Theme** — Shark Tank India–inspired design with animated ocean background

---

## 🗂️ Project Structure

```
shark-tank-india-valuation/
│
├── app.py                          # Streamlit application (main entry point)
├── requirements.txt                # Python dependencies for deployment
├── .gitignore                      # Files excluded from version control
│
├── data/
│   ├── Shark Tank India.csv                        # Raw dataset (original)
│   ├── Cleaned_Shark_Tank_India.xlsx               # After data_cleaning.ipynb
│   ├── MLR_Friendly_Shark_Tank_India.csv           # After eda.ipynb (model-ready)
│   └── prediction_history.csv                      # Auto-created; logs app predictions
│
├── models/
│   └── valuation_model_1.pkl                       # Trained OLS model (statsmodels)
│
├── data_cleanning.ipynb            # Step 1: Raw data cleaning pipeline
├── eda.ipynb                       # Step 2: EDA, encoding, scaling, feature selection
├── log_valuation.ipynb             # Step 3: Model training, evaluation, serialisation
│
├── mlr dummy reserch paper/        # Supporting research and references
└── read me/                        # Additional documentation assets
```

---

## 🧠 Machine Learning Pipeline

### Data Flow

```
Raw CSV  →  data_cleaning.ipynb  →  Cleaned Excel
                                         │
                                    eda.ipynb
                                         │
                               ┌─────────▼──────────┐
                               │  Feature Engineering │
                               │  • One-Hot Encoding  │
                               │  • log1p Transform   │
                               │  • StandardScaler    │
                               └─────────┬────────────┘
                                         │
                              log_valuation.ipynb
                                         │
                               ┌─────────▼──────────┐
                               │   OLS Regression    │
                               │   model_1.save()    │
                               └─────────┬────────────┘
                                         │
                                    app.py (Streamlit)
```

### Preprocessing Steps (exact pipeline reproduced in `app.py`)

| Step | Operation | Columns |
|------|-----------|---------|
| 1 | One-Hot Encoding (Hardware = baseline, dropped) | `Industry` → 17 dummies |
| 2 | `np.log1p(x)` transform | `Yearly Revenue`, `SKUs`, `Original Ask Amount` |
| 3 | Raw values (no log) | `Gross Margin %`, `Net Margin %`, `EBITDA`, `Equity %`, `Company Age` |
| 4 | Z-score standardisation `(x−μ)/σ` | All 8 continuous columns |
| 5 | Ordinal encoding | `Pitchers Average Age` → Young=0, Middle=1, Old=2 |
| 6 | Binary pass-through | `Has Patents`, `Bootstrapped` |

### Scaler Parameters (extracted from training data, 423 samples)

| Feature | Mean (μ) | Std (σ) |
|---------|----------|---------|
| Yearly Revenue *(after log1p)* | 4.7539 | 2.1766 |
| Gross Margin % | 55.500 | 15.375 |
| Net Margin % | 19.781 | 6.761 |
| EBITDA ₹L | 13.402 | 9.087 |
| SKUs *(after log1p)* | 3.999 | 1.202 |
| Original Ask Amount *(after log1p)* | 4.192 | 0.586 |
| Offered Equity % | 3.131 | 2.963 |
| Company Age (years) | 3.903 | 2.356 |

### Target Variable
- **Trained on:** `log1p(Deal Valuation in ₹ Lakhs)`
- **App output:** `np.expm1(model prediction)` → readable ₹ Lakhs

### Model Performance
| Metric | Value |
|--------|-------|
| Model | OLS Multiple Linear Regression |
| Library | `statsmodels` |
| Training samples | 423 (after outlier removal) |
| Outlier method | Externally studentized residuals (\|r\| > 2.5) |
| Features | 28 (17 industry dummies + 11 numeric) |

## 📊 App Pages

### 🦈 Page 1 — Valuation Engine
Enter your startup details across three input columns:
- **Identity:** Industry, founder age group, company age
- **Ask Details:** Investment ask (₹ Lakhs), equity offered (%), IP status, funding type
- **Financials:** Revenue, gross margin, net margin, EBITDA, SKUs

Click **"CALCULATE VALUATION"** to:
- See the predicted deal valuation in ₹ Lakhs
- Compare it against your implied valuation (Ask ÷ Equity)
- Watch the shark swim animation with a "DEAL STRUCK!" flash

### 📊 Page 2 — Pitch Analytics
- **5 benchmark tabs:** Revenue, Gross Margin, Net Margin, EBITDA, Deal Valuation — all comparing your startup against 18 industry medians
- **Implied vs Predicted chart:** Side-by-side bar chart of your ask-implied valuation vs the model's prediction
- **Health Radar:** 7-dimension spider chart scoring Gross Margin, Net Margin, Revenue, EBITDA, IP, Funding, Product Breadth
- **Gauge + Score Bars:** Overall health score (0–10) with breakdown
- **Model explainer expander:** Full preprocessing pipeline documentation

### 🗂️ Page 3 — Prediction History
- Full table of every prediction made in the session
- Trend line chart (Predicted vs Implied across runs)
- Industry distribution bar chart
- Export as CSV or Excel
- Downloads the full persistent `data/prediction_history.csv`

---

## 📁 Data Sources

| File | Description | Records |
|------|-------------|---------|
| `Shark Tank India.csv` | Raw scraped data | ~789 |
| `Cleaned_Shark_Tank_India.xlsx` | After cleaning pipeline | 789 |
| `MLR_Friendly_Shark_Tank_India.csv` | After EDA + encoding + scaling | 423 |

**Data collection:** Shark Tank India Seasons 1, 2, and 3 pitch data.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| App Framework | Streamlit |
| ML Model | statsmodels OLS |
| Data Processing | pandas, numpy |
| Visualisation | Plotly |
| Model Persistence | statsmodels `.save()` / `sm.load()` |
| Animation | CSS keyframes + inline SVG |
| Deployment | Streamlit Community Cloud |

---

## 📓 Notebooks

| Notebook | Purpose |
|----------|---------|
| `data_cleanning.ipynb` | Raw data cleaning: missing values, encoding, outlier handling |
| `eda.ipynb` | Exploratory analysis, log transforms, feature selection, scaling |
| `log_valuation.ipynb` | OLS model training, residual analysis, model evaluation, serialisation |

---

## 🔧 Troubleshooting

**Model not loading**
> Ensure `models/valuation_model_1.pkl` exists in your repo. The app looks for it at this exact path relative to `app.py`. Re-save it using:
> ```python
> model_1.save('models/valuation_model_1.pkl')
> ```

**Streamlit Cloud deployment fails**
> - Check that `requirements.txt` is at the repo root
> - Ensure the main file path is set to `app.py` in the Streamlit Cloud dashboard
> - Check the cloud logs for specific error messages

**Prediction gives unexpected results**
> Do not modify `SCALER_PARAMS` in `app.py` — these values are extracted from the training data and must exactly match what `StandardScaler` used during training.

**History CSV not saving**
> The `data/` folder must exist in the repo. Streamlit Cloud has an ephemeral filesystem — the CSV resets on each app restart. For persistent cloud storage, consider integrating a database (e.g. Supabase or Google Sheets via `gspread`).

---

## 🤝 Contributing

Contributions, issues and feature requests are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Amber Agrawal**
- GitHub: [Amber Agrawal](https://github.com/Amber-s-art)
- LinkedIn: [Amber Agrawal](https://www.linkedin.com/in/amber-agrawal/www.linkedin.com/in/amber-agrawal)

*Aditya Nagsen**
- GitHub: [Aditya Nagsen](https://github.com/Amber-s-art)
- LinkedIn: [Aditya Nagsen](https://www.linkedin.com/in/amber-agrawal/www.linkedin.com/in/amber-agrawal)

*Oishee **
- GitHub: [Amber Agrawal](https://github.com/Amber-s-art)
- LinkedIn: [Amber Agrawal](https://www.linkedin.com/in/amber-agrawal/www.linkedin.com/in/amber-agrawal)


*Kalyani**
- GitHub: [kalu](https://github.com/Amber-s-art)
- LinkedIn: [kalul](https://www.linkedin.com/in/amber-agrawal/www.linkedin.com/in/amber-agrawal)

---

## 🙏 Acknowledgements

- **Shark Tank India** for the inspiration and publicly available pitch data
- **Streamlit** for making ML app deployment effortless
- **statsmodels** for a production-grade OLS implementation
- The open-source Python data science ecosystem

---

*Built with ❤️ and a lot of shark puns 🦈*
