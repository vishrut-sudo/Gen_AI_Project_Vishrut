# ML Projects — Uni Portfolio

three machine learning projects built for uni. each one does EDA, model comparison, and evaluation.

---

## Projects

- [1. Delhi House Price Predictor](#1-delhi-house-price-predictor) — the main one
- [2. Car Price Prediction](#2-car-price-prediction)
- [3. Calories Burnt Prediction](#3-calories-burnt-prediction)

---

## 1. Delhi House Price Predictor

the flagship project. predicts Delhi house prices using ML and then does quant finance stuff on top of it because normal regression was too boring. VaR, CVaR, prediction intervals — treating flats like they are stock portfolios. unhinged? yes. but it actually works.

### what it does

takes ~1,259 Delhi housing listings from 99acres and runs them through three models — Linear Regression as the baseline, Random Forest for the serious stuff, and XGBoost which wins every time. then layers quant risk metrics on top to quantify prediction uncertainty in a way that actually means something.

the core insight: prediction errors follow a fat-tailed distribution, not a normal one. so vanilla Gaussian VaR understimates tail risk. basically the same lesson from 2008 but for a 2BHK in Dwarka.

### models used

| model | purpose |
|---|---|
| Linear Regression | baseline |
| Random Forest (300 trees, cross-validated) | solid performer |
| XGBoost | best results |

### risk metrics

| metric | what it means |
|---|---|
| Prediction Volatility | std dev of residuals, basically sigma |
| VaR 95% | max expected prediction error at 95% confidence |
| CVaR | avg error in the worst 5% of cases |
| Prediction Intervals | plus/minus 1 and 2 sigma bands |
| Quantile Regression | P10/P50/P90 price ranges so you get a spread not just one number |

### the streamlit app

three tabs —

- price predictor — input property details, get a predicted price, a confidence band, and a risk meter (Low / Medium / High)
- EDA dashboard — interactive plotly charts to explore the market data
- risk analytics — VaR/CVaR numbers, residual distribution, feature importance. screenshot this tab for linkedin

### how to run

```bash
pip install -r requirements.txt
jupyter notebook "Delhi house pricing.ipynb"
streamlit run app.py
```

### project structure

```
Delhi house pricing.ipynb   main analysis and model training
app.py                      streamlit dashboard
models/                     saved model files (generated after running notebook)
csv(s)/                     raw dataset
plots_*.png                 auto-generated EDA charts
```

### data

Delhi real estate listings scraped from 99acres.com. used for academic purposes only.

---

## 2. Car Price Prediction

predicts the selling price of used cars based on features like car age, fuel type, transmission, and mileage. standard ML pipeline with three models compared.

### what it does

loads a dataset of 301 used car listings, cleans it up (handles missing values, removes duplicates, engineers a Car_Age feature from the Year column), encodes categoricals, and trains three regression models. compares them using R2, MAE, and RMSE.

### models used

| model | notes |
|---|---|
| Linear Regression | baseline |
| Decision Tree | captures non-linearity |
| Random Forest (100 trees) | best performer |

### how to run

```bash
pip install -r requirements.txt
jupyter notebook Car_Price_Prediction000.ipynb
```

### data

used car listings dataset (`car data.csv`), 301 rows, 9 columns. features include year, present price, kms driven, fuel type, seller type, transmission, and owner history.

---

## 3. Calories Burnt Prediction

predicts how many calories a person burns during exercise based on biometric and workout data. merges two separate datasets and runs four models.

### what it does

takes two CSVs — `exercise.csv` (workout session data) and `calories.csv` (calories burnt) — merges them on User_ID, does EDA, and trains four regression models. evaluates on R2, MAE, and RMSE.

### models used

| model | notes |
|---|---|
| Linear Regression | baseline |
| Decision Tree | decent but overfits |
| Random Forest (100 trees) | strong performer |
| Gradient Boosting (100 estimators) | best results |

### how to run

```bash
pip install -r requirements.txt
jupyter notebook Calories_Burnt_Prediction.ipynb
```

### data

two CSVs merged on User_ID. features include gender, age, height, weight, duration, heart rate, and body temperature.

---

## requirements

```bash
pip install -r requirements.txt
```

all three projects share the same dependencies — pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, scipy, joblib, streamlit, plotly, statsmodels.
