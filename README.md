# Lahore House Price Predictor

A beginner-friendly machine learning project that predicts Lahore house prices in PKR using property features, society information, and live market rate calibration.

This project combines:
- A trained batch ML model for baseline prediction
- Online learning for continuous adaptation
- Live market indices integration for realistic per-marla anchoring
- A professional Streamlit web interface

## 1) What This Project Does

- Predicts house price based on:
  - Society
  - Marla size
  - Bedrooms / bathrooms
  - Garage / terrace / kitchens / drawing rooms
  - House age and condition
- Refreshes society rates hourly (live market mode)
- Calibrates predictions against market-per-marla behavior
- Supports incremental model updates without full retraining

## 2) ML Type, Model Type, and Sub-Type

### Primary Problem Type
- **Supervised Learning**
- **Regression** (predicting continuous numeric value: price)

### Offline/Batch Model
- **Model class:** `GradientBoostingRegressor` (scikit-learn)
- **Type:** Ensemble tree-based boosting regression
- **Use:** Strong baseline accuracy from generated/training dataset

### Online Learning Model
- **Model class:** `SGDRegressor` (scikit-learn)
- **Type:** Linear model trained with Stochastic Gradient Descent
- **Configured subtype:** Huber loss + L2 regularization + adaptive learning rate
- **Use:** Incremental updates via `partial_fit` as market state changes

### Encoding
- **Encoder:** `LabelEncoder`
- **Use:** Converts society names to numeric IDs for model input

## 3) Libraries Used (and Why)

- `pandas` → data tables and preprocessing
- `numpy` → numeric operations
- `scikit-learn` → ML models, splitting, metrics, encoders
- `streamlit` → web app UI
- `joblib` → save/load model and encoder files
- `requests` → fetch live market indices
- `streamlit-keyup` → responsive search input UX
- `streamlit-autorefresh` → periodic UI refresh for market updates

See full dependency list in `requirements.txt`.

## 4) Project Files Explained

- `app.py`
  - Streamlit frontend/controller
  - User input form, prediction display, market status UI
  - Hourly refresh and cache handling

- `predictor_core.py`
  - Core backend logic
  - Live market data fetch + normalization + mapping
  - Market-anchored pricing logic
  - Online model initialization and update routines

- `model.py`
  - Training pipeline script
  - Generates dataset, trains batch model, evaluates, and exports artifacts

- `society_data.json`
  - Baseline per-marla rates + society tiers

- `lahore_housing_data.csv`
  - Generated/reference training dataset

- `model.pkl`
  - Saved batch prediction model (Gradient Boosting)

- `label_encoder.pkl`
  - Saved encoder for society names

- `online_model.pkl`
  - Saved online learning model (SGDRegressor)

- `online_learning_state.json`
  - Online model state/version/signature metadata

## 5) Market Data Source Logic

- If `MARKET_RATES_URL` is provided, app first tries that configured feed.
- If unavailable, the app uses live Zameen indices mapping as fallback.
- Current mapping priority for Lahore per-marla behavior:
  1. Zameen residential plot indices (`type_id=12`) (primary)
  2. Zameen house indices (`type_id=9`) (fallback when needed)

This helps keep society rates closer to official market behavior where data is available.

## 6) How to Run

## Step A: Install dependencies

```bash
pip install -r requirements.txt
```

## Step B: (Optional) Retrain base model

```bash
python model.py
```

## Step C: Run the app

```bash
streamlit run app.py
```

## Optional environment variable

If you have your own live rate endpoint:

```bash
set MARKET_RATES_URL=https://your-feed-url
```

Then run Streamlit.

## 7) Prediction Flow (High Level)

1. Load model + encoder + society data
2. Load live market rates (configured feed or Zameen fallback)
3. Build feature vector from user inputs
4. Get baseline model prediction
5. Compute market-anchored estimate
6. Blend predictions with clipping/guardrails
7. Display final price and confidence-oriented context
8. Perform online update cycle for adaptation

## 8) Beginner Notes

- This project is practical for learning end-to-end ML app development:
  - Data preparation
  - Regression modeling
  - Model persistence
  - Live-data-aware inference
  - Web deployment with Streamlit

- It is intentionally readable and modular:
  - UI/controller in `app.py`
  - Core logic in `predictor_core.py`