"""
Lahore House Price Prediction - Model Training Script
=====================================================
This script trains a machine learning model using realistic
property price data from Lahore, Punjab, Pakistan.

Societies covered:
  DHA, Bahria Town, Johar Town, Model Town, Gulberg,
  Garden Town, Faisal Town, Cavalry Ground, Wapda Town,
  Iqbal Town, Canal Garden, Askari, Valencia, Lake City,
  State Life, Township, Sui Gas Society, PCSIR, EME Society,
  Architects Engineers Society, Punjab Govt. Servants Society,
  Sabzazar, Samanabad, Shadman, Ichhra, Muslim Town,
  Muhafiz Town, Tajpura, Green Town, Pak Arab Society,
  Revenue Society, Nasheman-e-Iqbal, Paragon City,
  Citi Housing, Etihad Town, Al-Kabir Town, NFC Society,
  Khayaban-e-Amin, Eden, Izmir Town, Audit & Accounts Society

Run this script FIRST:  python model.py
Then run the web app:   streamlit run app.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json

# ==================================================
# Step 1: Define Real Lahore Society Data
# ==================================================
# Average price per marla (in PKR Lakhs) for each society
# Based on 2024-2025 market research from Zameen.com, OLX,
# Graana.com, and PakObserver data.

SOCIETY_DATA = {
    # --- Premium / High-End Societies ---
    "Gulberg":                      {"avg_price_per_marla_lakhs": 130, "tier": "premium"},
    "DHA Phase 1-4":                {"avg_price_per_marla_lakhs": 100, "tier": "premium"},
    "DHA Phase 5":                  {"avg_price_per_marla_lakhs": 85,  "tier": "premium"},
    "DHA Phase 6":                  {"avg_price_per_marla_lakhs": 75,  "tier": "premium"},
    "DHA Phase 7-8":                {"avg_price_per_marla_lakhs": 55,  "tier": "premium"},
    "DHA Phase 9 (Prism)":          {"avg_price_per_marla_lakhs": 40,  "tier": "premium"},
    "DHA Phase 10-11":              {"avg_price_per_marla_lakhs": 25,  "tier": "premium"},
    "DHA Phase 13":                 {"avg_price_per_marla_lakhs": 18,  "tier": "mid"},
    "Cavalry Ground":               {"avg_price_per_marla_lakhs": 90,  "tier": "premium"},
    "Model Town":                   {"avg_price_per_marla_lakhs": 70,  "tier": "premium"},

    # --- Upper-Mid Societies ---
    "Garden Town":                  {"avg_price_per_marla_lakhs": 60,  "tier": "upper_mid"},
    "Faisal Town":                  {"avg_price_per_marla_lakhs": 50,  "tier": "upper_mid"},
    "Johar Town":                   {"avg_price_per_marla_lakhs": 45,  "tier": "upper_mid"},
    "Askari 10":                    {"avg_price_per_marla_lakhs": 48,  "tier": "upper_mid"},
    "Askari 11":                    {"avg_price_per_marla_lakhs": 42,  "tier": "upper_mid"},
    "Valencia Housing":             {"avg_price_per_marla_lakhs": 38,  "tier": "upper_mid"},
    "Shadman":                      {"avg_price_per_marla_lakhs": 65,  "tier": "upper_mid"},

    # --- Mid-Range Societies ---
    "Bahria Town (Sector A-D)":     {"avg_price_per_marla_lakhs": 35,  "tier": "mid"},
    "Bahria Town (Sector E-F)":     {"avg_price_per_marla_lakhs": 30,  "tier": "mid"},
    "Wapda Town":                   {"avg_price_per_marla_lakhs": 32,  "tier": "mid"},
    "Iqbal Town":                   {"avg_price_per_marla_lakhs": 35,  "tier": "mid"},
    "Muslim Town":                  {"avg_price_per_marla_lakhs": 38,  "tier": "mid"},
    "Township":                     {"avg_price_per_marla_lakhs": 28,  "tier": "mid"},
    "EME Society":                  {"avg_price_per_marla_lakhs": 30,  "tier": "mid"},
    "Sui Gas Society":              {"avg_price_per_marla_lakhs": 34,  "tier": "mid"},
    "Lake City":                    {"avg_price_per_marla_lakhs": 22,  "tier": "mid"},
    "State Life Housing":           {"avg_price_per_marla_lakhs": 26,  "tier": "mid"},
    "PCSIR Housing Society":        {"avg_price_per_marla_lakhs": 25,  "tier": "mid"},
    "Paragon City":                 {"avg_price_per_marla_lakhs": 22,  "tier": "mid"},
    "Nasheman-e-Iqbal":             {"avg_price_per_marla_lakhs": 20,  "tier": "mid"},

    # --- Affordable / Budget Societies ---
    "Canal Garden":                 {"avg_price_per_marla_lakhs": 18,  "tier": "affordable"},
    "Citi Housing":                 {"avg_price_per_marla_lakhs": 20,  "tier": "affordable"},
    "Etihad Town":                  {"avg_price_per_marla_lakhs": 14,  "tier": "affordable"},
    "Al-Kabir Town":                {"avg_price_per_marla_lakhs": 12,  "tier": "affordable"},
    "Khayaban-e-Amin":              {"avg_price_per_marla_lakhs": 16,  "tier": "affordable"},
    "NFC Society":                  {"avg_price_per_marla_lakhs": 22,  "tier": "affordable"},
    "Green Town":                   {"avg_price_per_marla_lakhs": 18,  "tier": "affordable"},
    "Sabzazar":                     {"avg_price_per_marla_lakhs": 20,  "tier": "affordable"},
    "Samanabad":                    {"avg_price_per_marla_lakhs": 22,  "tier": "affordable"},
    "Ichhra":                       {"avg_price_per_marla_lakhs": 28,  "tier": "affordable"},
    "Muhafiz Town":                 {"avg_price_per_marla_lakhs": 16,  "tier": "affordable"},
    "Tajpura":                      {"avg_price_per_marla_lakhs": 15,  "tier": "affordable"},
    "Pak Arab Society":             {"avg_price_per_marla_lakhs": 14,  "tier": "affordable"},
    "Revenue Society":              {"avg_price_per_marla_lakhs": 16,  "tier": "affordable"},
    "Eden City":                    {"avg_price_per_marla_lakhs": 14,  "tier": "affordable"},
    "Izmir Town":                   {"avg_price_per_marla_lakhs": 12,  "tier": "affordable"},
    "Audit & Accounts Society":     {"avg_price_per_marla_lakhs": 16,  "tier": "affordable"},
    "Architects Engineers Society": {"avg_price_per_marla_lakhs": 18,  "tier": "affordable"},
    "Punjab Govt Servants Society": {"avg_price_per_marla_lakhs": 18,  "tier": "affordable"},
}

# ==================================================
# Step 2: Generate Realistic Training Data
# ==================================================
np.random.seed(42)
rows = []

for society, info in SOCIETY_DATA.items():
    base_price = info["avg_price_per_marla_lakhs"]  # in Lakhs per marla
    tier = info["tier"]

    # Generate multiple samples per society for different house sizes
    # Each society gets 40-60 samples with varied house sizes
    n = np.random.randint(40, 61)

    for _ in range(n):
        # House size in marla (common in Pakistan: 3, 5, 7, 10, 15, 20)
        marla = np.random.choice([3, 5, 7, 10, 15, 20],
                                  p=[0.10, 0.30, 0.15, 0.25, 0.10, 0.10])

        # Bedrooms depend on house size
        if marla <= 3:
            bedrooms = np.random.choice([2, 3])
        elif marla <= 5:
            bedrooms = np.random.choice([2, 3, 4])
        elif marla <= 7:
            bedrooms = np.random.choice([3, 4, 5])
        elif marla <= 10:
            bedrooms = np.random.choice([3, 4, 5, 6])
        else:
            bedrooms = np.random.choice([4, 5, 6, 7])

        # Bathrooms depend on bedrooms
        bathrooms = max(1, bedrooms - np.random.choice([0, 1]))

        # Additional layout features
        # Garage capacity (number of cars)
        if marla <= 3:
            garage = np.random.choice([0, 1], p=[0.85, 0.15])
            terrace = np.random.choice([0, 1], p=[0.75, 0.25])
        elif marla <= 5:
            garage = np.random.choice([0, 1], p=[0.35, 0.65])
            terrace = np.random.choice([0, 1], p=[0.45, 0.55])
        elif marla <= 10:
            garage = np.random.choice([1, 2], p=[0.70, 0.30])
            terrace = np.random.choice([0, 1, 2], p=[0.20, 0.65, 0.15])
        else:
            garage = np.random.choice([1, 2, 3], p=[0.20, 0.60, 0.20])
            terrace = np.random.choice([1, 2], p=[0.60, 0.40])

        # Kitchens and drawing rooms scale with layout/size
        kitchens = int(np.clip(np.round((bedrooms / 3) + (marla / 12)), 1, 3))
        kitchens = max(1, kitchens + np.random.choice([-1, 0, 0, 1]))
        kitchens = int(np.clip(kitchens, 1, 3))

        drawing_rooms = 0
        if marla >= 5:
            drawing_rooms = 1
        if marla >= 12 and np.random.rand() > 0.45:
            drawing_rooms = 2
        if marla >= 20 and np.random.rand() > 0.75:
            drawing_rooms = 3

        # Age of house (0 = brand new)
        age = np.random.randint(0, 30)

        # Condition: 1 (Poor) to 5 (Excellent)
        condition = np.random.randint(1, 6)

        # ---- Calculate Price in PKR ----
        # Base price = price_per_marla * marla (in Lakhs)
        total_price_lakhs = base_price * marla

        # Add premium for more bedrooms & bathrooms
        total_price_lakhs += bedrooms * 3
        total_price_lakhs += bathrooms * 2

        # Add layout premiums for additional amenities
        total_price_lakhs += garage * (base_price * 0.30)
        total_price_lakhs += terrace * (base_price * 0.18)
        total_price_lakhs += kitchens * 1.8
        total_price_lakhs += drawing_rooms * 2.5

        # Deduct for age (older houses lose value)
        age_penalty = age * (base_price * 0.008)  # ~0.8% per year
        total_price_lakhs -= age_penalty

        # Add bonus for condition
        condition_bonus = (condition - 3) * (base_price * 0.05)
        total_price_lakhs += condition_bonus

        # Add realistic noise (±10%)
        noise = np.random.uniform(-0.10, 0.10)
        total_price_lakhs *= (1 + noise)

        # Convert to PKR (1 Lakh = 100,000)
        price_pkr = max(total_price_lakhs * 100000, 500000)  # Floor at 5 Lakh

        rows.append({
            "Society": society,
            "Marla": int(marla),
            "Bedrooms": int(bedrooms),
            "Bathrooms": int(bathrooms),
            "Garage": int(garage),
            "Terrace": int(terrace),
            "Kitchens": int(kitchens),
            "Drawing Rooms": int(drawing_rooms),
            "Age (years)": int(age),
            "Condition (1-5)": int(condition),
            "Price (PKR)": round(price_pkr)
        })

data = pd.DataFrame(rows)

print("=" * 60)
print("🏘️  LAHORE HOUSE PRICE PREDICTION - Model Training")
print("=" * 60)
print(f"\n📊 Total Training Samples: {len(data)}")
print(f"📍 Total Societies: {data['Society'].nunique()}")
print(f"\n📋 Dataset Preview:")
print(data.head(10).to_string(index=False))
print(f"\n💰 Price Range:")
print(f"   Min:  PKR {data['Price (PKR)'].min():,.0f}")
print(f"   Max:  PKR {data['Price (PKR)'].max():,.0f}")
print(f"   Mean: PKR {data['Price (PKR)'].mean():,.0f}")

# ==================================================
# Step 3: Prepare Features for Training
# ==================================================
# Encode society names as numbers (the model needs numbers, not text)
label_encoder = LabelEncoder()
data["Society_Encoded"] = label_encoder.fit_transform(data["Society"])

# Features: Society (encoded), core specs, and layout amenities
X = data[[
    "Society_Encoded",
    "Marla",
    "Bedrooms",
    "Bathrooms",
    "Garage",
    "Terrace",
    "Kitchens",
    "Drawing Rooms",
    "Age (years)",
    "Condition (1-5)",
]]
y = data["Price (PKR)"]

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n🔀 Train/Test Split:")
print(f"   Training: {len(X_train)} samples")
print(f"   Testing:  {len(X_test)} samples")

# ==================================================
# Step 4: Train the Model (Gradient Boosting for accuracy)
# ==================================================
print(f"\n⏳ Training model... (this may take a few seconds)")

model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

print(f"✅ Model trained successfully!")

# ==================================================
# Step 5: Evaluate the Model
# ==================================================
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n{'=' * 60}")
print(f"📏 Model Evaluation:")
print(f"{'=' * 60}")
print(f"   MAE  (Mean Absolute Error): PKR {mae:,.0f}")
print(f"   R² Score:                   {r2:.4f}")
print(f"   (R² closer to 1.0 = better predictions)")

# ==================================================
# Step 6: Save Everything
# ==================================================
# Save the trained model
joblib.dump(model, 'model.pkl')
print(f"\n💾 Saved: model.pkl")

# Save the label encoder (needed by the web app to convert society names)
joblib.dump(label_encoder, 'label_encoder.pkl')
print(f"💾 Saved: label_encoder.pkl")

# Save society list and data for the web app
society_info = {}
for society, info in SOCIETY_DATA.items():
    society_info[society] = {
        "avg_price_per_marla_lakhs": info["avg_price_per_marla_lakhs"],
        "tier": info["tier"]
    }

with open("society_data.json", "w", encoding="utf-8") as f:
    json.dump(society_info, f, indent=2, ensure_ascii=False)
print(f"💾 Saved: society_data.json")

# Save the dataset to CSV for reference
data.to_csv("lahore_housing_data.csv", index=False)
print(f"💾 Saved: lahore_housing_data.csv")

print(f"\n{'=' * 60}")
print(f"✅ All done! Now run:  streamlit run app.py")
print(f"{'=' * 60}")
