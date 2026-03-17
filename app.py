"""
Lahore House Price Prediction - Streamlit Web App
===================================================
A web interface for predicting house prices across
Lahore's housing societies in PKR.

Run with: streamlit run app.py
"""

import json
import os
import joblib
import numpy as np
import streamlit as st

try:
    from st_keyup import st_keyup
except ImportError:
    st_keyup = None

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

from predictor_core import (
    LATEST_BASE_FEATURE_COUNT,
    LATEST_ONLINE_FEATURE_COUNT,
    PKR_PER_LAKH,
    apply_online_update,
    build_sidebar_society_matches,
    calculate_market_anchored_price,
    format_update_timestamp,
    initialize_online_model,
    load_live_society_data_core,
    resolve_feature_vector,
    safe_float,
)

# Page setup
st.set_page_config(page_title="Lahore House Price Predictor", page_icon="🏠", layout="wide")

# Theme CSS
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    :root {
        --bg-1: #07120e;
        --bg-2: #0a2118;
        --text: #e9fff5;
        --muted: #b8dbc8;
        --card: rgba(15, 43, 31, 0.67);
        --card-strong: rgba(17, 54, 39, 0.84);
        --border: rgba(117, 227, 176, 0.24);
        --accent-1: #2f8f5b;
        --accent-2: #4cd08b;
        --accent-3: #1f6f47;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(1000px 520px at 6% -10%, rgba(72, 207, 140, 0.20), transparent 62%),
            radial-gradient(980px 520px at 100% -6%, rgba(39, 133, 90, 0.24), transparent 58%),
            linear-gradient(135deg, var(--bg-1), var(--bg-2));
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stToolbar"] {
        right: 0.8rem;
    }

    .block-container {
        max-width: 1260px;
        padding-top: 1rem;
        padding-bottom: 2.2rem;
    }

    .main-header {
        text-align: center;
        padding: 1.8rem 1.1rem;
        margin-bottom: 1rem;
        color: white;
        border-radius: 24px;
        border: 1px solid rgba(152, 255, 215, 0.30);
        background:
            radial-gradient(circle at 16% 18%, rgba(173, 255, 220, 0.22), transparent 34%),
            linear-gradient(135deg, #195637 0%, #2a8756 52%, #3bb174 100%);
        box-shadow:
            0 20px 48px rgba(0, 0, 0, 0.28),
            inset 0 1px 0 rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(4px);
    }

    .main-header h1 {
        color: #ffffff !important;
        font-size: 2.35rem;
        letter-spacing: 0.2px;
        margin: 0;
        font-weight: 700;
    }

    .main-header p {
        color: rgba(255, 255, 255, 0.90);
        font-size: 1.03rem;
        margin: 0.3rem 0 0;
    }

    .status-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.7rem;
        margin: 0.15rem 0 1.2rem;
    }

    .status-card {
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.72rem 0.85rem;
        background: var(--card);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.16);
    }

    .status-card .label {
        color: var(--muted);
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.2rem;
        font-weight: 600;
    }

    .status-card .value {
        color: var(--text);
        font-size: 0.93rem;
        font-weight: 600;
        line-height: 1.35;
        word-break: break-word;
    }

    .society-info {
        background: var(--card);
        border: 1px solid var(--border);
        border-left: 4px solid var(--accent-2);
        border-radius: 14px;
        padding: 1rem 1.05rem;
        margin: 0.55rem 0;
        color: var(--text);
        box-shadow: 0 10px 26px rgba(0, 0, 0, 0.18);
    }

    .prediction-box {
        background:
            radial-gradient(circle at 100% 0%, rgba(170, 255, 223, 0.20), transparent 36%),
            linear-gradient(135deg, #154a31 0%, #226e47 62%, #2f8e5d 100%);
        border: 1px solid rgba(146, 255, 212, 0.24);
        border-radius: 22px;
        padding: 2rem 1.6rem;
        text-align: center;
        color: #fff;
        margin: 1.3rem 0;
        box-shadow:
            0 24px 45px rgba(0, 0, 0, 0.25),
            inset 0 1px 0 rgba(255, 255, 255, 0.14);
    }

    .prediction-box h2 {
        color: #fff;
        font-size: 2.7rem;
        letter-spacing: 0.4px;
        margin: 0.45rem 0;
    }

    .prediction-box p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1rem;
        margin: 0.27rem 0;
    }

    [data-testid="metric-container"] {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.7rem 0.8rem;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
    }

    [data-testid="metric-container"] [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
    }

    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: var(--text) !important;
    }

    .stSelectbox label,
    .stTextInput label,
    .stCaption,
    .stSubheader,
    .stHeader,
    .stMarkdown,
    .stAlert {
        color: var(--text) !important;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="base-input"] > div,
    .stTextInput > div > div > input {
        background: var(--card-strong) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        color: var(--text) !important;
        transition: all 0.2s ease;
    }

    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="base-input"] > div:hover,
    .stTextInput > div > div > input:hover {
        border-color: rgba(168, 255, 221, 0.46) !important;
    }

    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="base-input"] > div:focus-within,
    .stTextInput > div > div > input:focus {
        border-color: rgba(172, 255, 226, 0.72) !important;
        box-shadow: 0 0 0 0.18rem rgba(72, 203, 140, 0.18) !important;
    }

    .stButton > button {
        border-radius: 14px !important;
        border: 1px solid rgba(169, 255, 221, 0.34) !important;
        background: linear-gradient(135deg, var(--accent-1), var(--accent-2)) !important;
        color: #fff !important;
        font-weight: 700 !important;
        min-height: 3rem;
        box-shadow: 0 12px 28px rgba(27, 105, 70, 0.42);
        transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 15px 30px rgba(27, 105, 70, 0.52);
        filter: brightness(1.03);
    }

    .stMarkdown table {
        width: 100%;
        border-collapse: collapse;
        border-radius: 12px;
        overflow: hidden;
        background: rgba(15, 45, 32, 0.64);
        border: 1px solid var(--border);
    }

    .stMarkdown table thead tr,
    .stMarkdown table tbody tr:nth-child(even) {
        background: rgba(36, 89, 65, 0.35);
    }

    .stMarkdown table th,
    .stMarkdown table td {
        border-color: rgba(118, 222, 173, 0.2) !important;
        color: var(--text) !important;
        padding: 0.62rem 0.7rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(12, 34, 25, 0.96) 0%, rgba(14, 39, 28, 0.92) 100%);
        border-right: 1px solid rgba(87, 199, 150, 0.22);
    }

    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #e8fff4 !important;
    }

    [data-testid="stSidebar"] .stAlert {
        background: rgba(81, 35, 15, 0.5) !important;
        border: 1px solid rgba(255, 185, 122, 0.34) !important;
        border-radius: 12px;
    }

    .society-rate-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.7rem;
        padding: 0.45rem 0.55rem;
        margin: 0.25rem 0;
        border-radius: 10px;
        background: rgba(41, 94, 67, 0.28);
        border: 1px solid rgba(112, 223, 171, 0.2);
    }

    .society-rate-name {
        color: #eafff4;
        font-size: 0.87rem;
        font-weight: 600;
        line-height: 1.3;
    }

    .society-rate-value {
        color: #8cf4c3;
        font-size: 0.84rem;
        font-weight: 700;
        white-space: nowrap;
    }

    .footer {
        text-align: center;
        color: #c4e8d9;
        padding: 1rem;
        font-size: 0.9rem;
        background: rgba(12, 34, 25, 0.5);
        border: 1px solid var(--border);
        border-radius: 14px;
    }

    hr {
        border-color: rgba(102, 205, 154, 0.22) !important;
    }

    @media (max-width: 992px) {
        .block-container {
            padding-top: 0.7rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .main-header {
            border-radius: 16px;
            padding: 1.3rem 0.8rem;
        }

        .main-header h1 {
            font-size: 1.95rem;
        }

        .prediction-box h2 {
            font-size: 2.2rem;
        }

        .status-grid {
            grid-template-columns: 1fr;
            gap: 0.55rem;
        }
    }

    @media (max-width: 640px) {
        .block-container {
            padding-top: 0.55rem;
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }

        .main-header h1 {
            font-size: 1.62rem;
            line-height: 1.25;
        }

        .main-header p {
            font-size: 0.92rem;
        }

        .prediction-box {
            padding: 1.4rem 1rem;
            border-radius: 16px;
        }

        .prediction-box h2 {
            font-size: 1.82rem;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)

# Runtime config
MARKET_REFRESH_SECONDS = 3600
MARKET_RATES_URL = os.getenv("MARKET_RATES_URL", "").strip()
ENABLE_ZAMEEN_AUTO_RATES = os.getenv("ENABLE_ZAMEEN_AUTO_RATES", "1").strip().lower() not in {"0", "false", "no"}
ZAMEEN_TIMEOUT_SECONDS = int(os.getenv("ZAMEEN_TIMEOUT_SECONDS", "12"))

base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "model.pkl")
encoder_path = os.path.join(base_dir, "label_encoder.pkl")
society_path = os.path.join(base_dir, "society_data.json")
training_data_path = os.path.join(base_dir, "lahore_housing_data.csv")
online_model_path = os.path.join(base_dir, "online_model.pkl")
online_state_path = os.path.join(base_dir, "online_learning_state.json")


def format_pkr(amount):
    if amount >= 10000000:
        return f"PKR {amount / 10000000:,.2f} Crore"
    if amount >= 100000:
        return f"PKR {amount / 100000:,.2f} Lakh"
    return f"PKR {amount:,.0f}"


@st.cache_data(ttl=MARKET_REFRESH_SECONDS, show_spinner=False)
def load_live_society_data(society_file_path, market_rates_url, enable_zameen_auto_rates, zameen_timeout_seconds):
    with open(society_file_path, "r", encoding="utf-8") as f:
        base_societies = json.load(f)
    return load_live_society_data_core(
        base_societies,
        market_rates_url,
        enable_zameen_auto_rates=enable_zameen_auto_rates,
        zameen_timeout_seconds=zameen_timeout_seconds,
    )


# Auto refresh
if st_autorefresh is not None:
    st_autorefresh(interval=MARKET_REFRESH_SECONDS * 1000, key="market_rate_refresh")

# Required files
required_assets = [
    (model_path, "⚠️ Model file not found! Please run `python model.py` first.", "python model.py"),
    (encoder_path, "⚠️ Label encoder not found! Please run `python model.py` first.", None),
    (society_path, "⚠️ Society data not found! Please run `python model.py` first.", None),
]
for file_path, message, command in required_assets:
    if not os.path.exists(file_path):
        st.error(message)
        if command:
            st.code(command, language="bash")
        st.stop()

# Load models and data
model = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)
with open(society_path, "r", encoding="utf-8") as f:
    baseline_society_data = json.load(f)

society_data, market_source, market_updated_at, market_warning = load_live_society_data(
    society_path,
    MARKET_RATES_URL,
    ENABLE_ZAMEEN_AUTO_RATES,
    ZAMEEN_TIMEOUT_SECONDS,
)

online_model = initialize_online_model(
    online_model_path,
    training_data_path,
    label_encoder,
    society_data,
    online_state_path,
)
online_model_refreshed, online_model_updated_at = apply_online_update(
    online_model,
    society_data,
    label_encoder,
    online_state_path,
    online_model_path,
)

# Header
st.markdown(
    """
<div class="main-header">
    <h1>🏠 Lahore House Price Predictor</h1>
    <p>🇵🇰 Predict property prices across Lahore's housing societies</p>
    <p>Powered by Machine Learning | Prices in PKR</p>
</div>
""",
    unsafe_allow_html=True,
)

online_status_label = "Online model active"
if online_model is not None and online_model_refreshed:
    online_status_label = "Model refreshed with latest market snapshot"
elif online_model is None:
    online_status_label = "Online model unavailable (rate-anchored mode)"

market_health = "Live feed healthy" if not market_warning else "Live feed fallback active"

st.markdown(
    f"""
<div class="status-grid">
    <div class="status-card">
        <div class="label">Market Source</div>
        <div class="value">{market_source}</div>
    </div>
    <div class="status-card">
        <div class="label">Last Market Update</div>
        <div class="value">{format_update_timestamp(market_updated_at)}</div>
    </div>
    <div class="status-card">
        <div class="label">Model Status</div>
        <div class="value">{online_status_label}<br><span style='color:#b7e7cf'>{market_health}</span></div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# Society lists
tier_labels = {
    "premium": "🏆 Premium Societies",
    "upper_mid": "🌟 Upper-Mid Societies",
    "mid": "🏘️ Mid-Range Societies",
    "affordable": "🏡 Affordable Societies",
}
tier_display = {
    "premium": "🏆 Premium",
    "upper_mid": "🌟 Upper-Mid",
    "mid": "🏘️ Mid-Range",
    "affordable": "🏡 Affordable",
}

tier_groups = {}
for name, info in society_data.items():
    tier_groups.setdefault(info["tier"], []).append(name)
for tier in tier_groups:
    tier_groups[tier].sort()

ordered_tiers = ["premium", "upper_mid", "mid", "affordable"]
all_societies = []
for tier_key in ordered_tiers + sorted([tier for tier in tier_groups if tier not in ordered_tiers]):
    if tier_key in tier_groups:
        all_societies.extend(tier_groups[tier_key])

# Input section
st.subheader("📍 Select Society & House Details")
selected_society = st.selectbox(
    "🏘️ Housing Society / Area",
    options=all_societies,
    index=all_societies.index("Johar Town") if "Johar Town" in all_societies else 0,
    help="Select the housing society in Lahore",
)

soc_info = society_data[selected_society]
st.markdown(
    f"""
<div class="society-info">
    <strong>{selected_society}</strong><br>
    📊 Category: {tier_display.get(soc_info['tier'], soc_info['tier'])}<br>
    💰 Avg. Rate: ~PKR {soc_info['avg_price_per_marla_lakhs']} Lakh / Marla
</div>
""",
    unsafe_allow_html=True,
)

st.divider()
st.subheader("📝 House Specifications")

row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    marla = st.selectbox(
        "📐 House Size (Marla)",
        options=[3, 5, 7, 10, 15, 20],
        index=1,
        help="Common sizes: 3, 5, 7, 10 Marla or 1 Kanal (20 Marla)",
    )
with row1_col2:
    bedrooms = st.selectbox("🛏️ Bedrooms", options=[2, 3, 4, 5, 6, 7], index=1, help="Number of bedrooms")

row2_col1, row2_col2 = st.columns(2)
with row2_col1:
    bathrooms = st.selectbox("🚿 Bathrooms", options=[1, 2, 3, 4, 5, 6], index=1, help="Number of bathrooms")
with row2_col2:
    kitchens = st.selectbox("🍽️ Kitchens", options=[1, 2, 3], index=0, help="Total kitchens in the house")

row3_col1, row3_col2 = st.columns(2)
with row3_col1:
    garage = st.selectbox("🚗 Garage (Cars)", options=[0, 1, 2, 3], index=1, help="Number of cars that can be parked")
with row3_col2:
    terrace = st.selectbox("🌤️ Terrace", options=[0, 1, 2], index=1, help="Number of terrace areas")

row4_col1, row4_col2 = st.columns(2)
with row4_col1:
    drawing_rooms = st.selectbox("🛋️ Drawing Rooms", options=[0, 1, 2, 3], index=1, help="Total drawing/living rooms")
with row4_col2:
    condition = st.selectbox(
        "🏗️ House Condition",
        options=[1, 2, 3, 4, 5],
        index=2,
        format_func=lambda x: {
            1: "1 - Poor (Needs major renovation)",
            2: "2 - Below Average",
            3: "3 - Average (Liveable)",
            4: "4 - Good (Well maintained)",
            5: "5 - Excellent (Like new)",
        }[x],
        help="Rate the overall condition of the house",
    )

age = st.selectbox("📅 House Age (Years)", options=list(range(0, 31)), index=5, help="0 = Brand New, 30 = Very Old")

# Live preview KPIs
live_rate_lakh = safe_float(soc_info.get("avg_price_per_marla_lakhs"), 0.0)
market_base_preview = live_rate_lakh * marla * PKR_PER_LAKH
amenity_score = int(garage + terrace + kitchens + drawing_rooms)
condition_age_summary = f"{condition}/5 • {age} years"

preview_col1, preview_col2, preview_col3 = st.columns(3)
preview_col1.metric("Live Market Base", format_pkr(market_base_preview))
preview_col2.metric("Amenity Score", f"{amenity_score}/10")
preview_col3.metric("Condition & Age", condition_age_summary)
st.caption("Live preview updates as you change inputs. Final price appears after prediction.")

st.divider()

# Prediction
if "show_prediction" not in st.session_state:
    st.session_state.show_prediction = False
if st.button("🔮 Predict House Price", use_container_width=True, type="primary"):
    st.session_state.show_prediction = True

if st.session_state.show_prediction:
    try:
        society_encoded = label_encoder.transform([selected_society])[0]
    except ValueError:
        st.error(f"⚠️ Society '{selected_society}' not recognized. Please retrain the model.")
        st.stop()

    current_rate_lakh = safe_float(soc_info.get("avg_price_per_marla_lakhs"), 0.0)
    market_anchor_price = calculate_market_anchored_price(
        current_rate_lakh,
        marla,
        bedrooms,
        bathrooms,
        garage,
        terrace,
        kitchens,
        drawing_rooms,
        age,
        condition,
    )

    latest_base_vector = [
        society_encoded,
        marla,
        bedrooms,
        bathrooms,
        garage,
        terrace,
        kitchens,
        drawing_rooms,
        age,
        condition,
    ]
    legacy_base_vector = [society_encoded, marla, bedrooms, bathrooms, age, condition]
    base_features = resolve_feature_vector(model, latest_base_vector, legacy_base_vector, LATEST_BASE_FEATURE_COUNT)
    base_prediction = model.predict(np.array([base_features], dtype=float))[0]
    prediction_notes = []

    if online_model is not None:
        latest_online_vector = [
            society_encoded,
            marla,
            bedrooms,
            bathrooms,
            garage,
            terrace,
            kitchens,
            drawing_rooms,
            age,
            condition,
            current_rate_lakh,
        ]
        legacy_online_vector = [society_encoded, marla, bedrooms, bathrooms, age, condition, current_rate_lakh]
        online_features = resolve_feature_vector(
            online_model,
            latest_online_vector,
            legacy_online_vector,
            LATEST_ONLINE_FEATURE_COUNT,
        )

        online_prediction_lakh = float(online_model.predict(np.array([online_features], dtype=float))[0])
        online_prediction = online_prediction_lakh * PKR_PER_LAKH
        online_ratio = online_prediction / market_anchor_price if market_anchor_price > 0 else 1.0

        if np.isfinite(online_prediction) and online_prediction > 0 and 0.60 <= online_ratio <= 1.55:
            predicted_price = (online_prediction * 0.10) + (base_prediction * 0.15) + (market_anchor_price * 0.75)
        else:
            predicted_price = (base_prediction * 0.20) + (market_anchor_price * 0.80)
            prediction_notes.append("Online update skipped for this input (outlier-safe fallback in use).")
    else:
        baseline_rate = safe_float(
            baseline_society_data.get(selected_society, {}).get("avg_price_per_marla_lakhs", current_rate_lakh),
            current_rate_lakh,
        )
        rate_shift = current_rate_lakh / baseline_rate if baseline_rate and baseline_rate > 0 else 1.0
        adjusted_base = base_prediction * rate_shift
        predicted_price = (adjusted_base * 0.20) + (market_anchor_price * 0.80)

    predicted_price = float(np.clip(predicted_price, market_anchor_price * 0.82, market_anchor_price * 1.28))
    predicted_price = max(predicted_price, 0)
    price_formatted = format_pkr(predicted_price)
    price_per_marla = format_pkr(predicted_price / marla)

    st.markdown(
        f"""
    <div class="prediction-box">
        <p>🏠 Estimated House Price</p>
        <h2>{price_formatted}</h2>
        <p>📍 {selected_society} | {marla} Marla | {bedrooms} Bed | {bathrooms} Bath</p>
        <p>🚗 Garage: {garage} | 🌤️ Terrace: {terrace} | 🍽️ Kitchens: {kitchens} | 🛋️ Drawing: {drawing_rooms}</p>
        <p>💹 Current Rate Used: ~PKR {current_rate_lakh:.2f} Lakh/Marla</p>
        <p>📐 ≈ {price_per_marla} per Marla</p>
        <p>🏗️ Age: {age} years | Condition: {condition}/5</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if prediction_notes:
        for note in prediction_notes:
            st.caption(f"ℹ️ {note}")

    st.subheader("📋 Prediction Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Society", selected_society)
    col2.metric("Size", f"{marla} Marla")
    col3.metric("Total Price", price_formatted)

    col4, col5, col6 = st.columns(3)
    col4.metric("Bedrooms", f"{bedrooms}")
    col5.metric("Bathrooms", f"{bathrooms}")
    col6.metric("Per Marla", price_per_marla)

    col7, col8, col9, col10 = st.columns(4)
    col7.metric("Garage", f"{garage}")
    col8.metric("Terrace", f"{terrace}")
    col9.metric("Kitchens", f"{kitchens}")
    col10.metric("Drawing Rooms", f"{drawing_rooms}")

    st.subheader("📊 Quick Market Comparison")
    st.markdown(
        f"""
    | Detail | Value |
    |--------|-------|
    | **Society** | {selected_society} |
    | **Market Avg. Rate** | ~PKR {soc_info['avg_price_per_marla_lakhs']} Lakh/Marla |
    | **Predicted Total** | {price_formatted} |
    | **Predicted Rate** | {price_per_marla}/Marla |
    | **House Size** | {marla} Marla |
    | **Garage** | {garage} |
    | **Terrace** | {terrace} |
    | **Kitchens** | {kitchens} |
    | **Drawing Rooms** | {drawing_rooms} |
    | **Category** | {tier_display.get(soc_info['tier'], soc_info['tier'])} |
    """
    )

# Sidebar guide
with st.sidebar:
    st.header("📍 Lahore Society Guide")
    st.markdown("Average rates per Marla:")
    st.caption("⏱️ Auto-refresh: every hour")
    st.caption(f"📡 Rate source: {market_source}")
    st.caption(f"🕒 Last market update: {format_update_timestamp(market_updated_at)}")

    if market_warning:
        st.warning(
            "Live rate refresh issue. Using best available rates with baseline fallback where needed.\n\n"
            f"Details: {market_warning}"
        )

    if online_model is not None:
        status = "Updated with latest market snapshot" if online_model_refreshed else "Online model active"
        if online_model_updated_at:
            status = f"{status} ({format_update_timestamp(online_model_updated_at)})"
        st.caption(f"🧠 {status}")
    else:
        st.caption("🧠 Online model unavailable (using rate-anchored mode).")

    if st_keyup is not None:
        search_query = st_keyup(
            "Search Society",
            key="sidebar_society_search",
            placeholder="Type society name...",
            debounce=100,
        ).strip()
    else:
        search_query = st.text_input(
            "Search Society",
            placeholder="Type society name...",
            help="Search from the societies listed below",
        ).strip()

    if search_query:
        query_lower = search_query.lower()

        # Rank: startswith, word-prefix, then contains.
        filtered_societies = build_sidebar_society_matches(all_societies, query_lower)

        st.markdown("### 🎯 Matching Societies")
        if filtered_societies:
            for society in filtered_societies:
                rate = society_data[society]["avg_price_per_marla_lakhs"]
                st.markdown(
                    f"""
<div class="society-rate-item">
    <span class="society-rate-name">{society}</span>
    <span class="society-rate-value">~{rate} Lakh/Marla</span>
</div>
""",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No matching society found.")
    else:
        sidebar_tier_order = ordered_tiers + sorted([tier for tier in tier_groups.keys() if tier not in ordered_tiers])
        for tier_key in sidebar_tier_order:
            if tier_key not in tier_groups:
                continue
            section_title = tier_labels.get(tier_key, f"📍 {tier_key.replace('_', ' ').title()} Societies")
            st.markdown(f"### {section_title}")
            for society in tier_groups[tier_key]:
                rate = society_data[society]["avg_price_per_marla_lakhs"]
                st.markdown(
                    f"""
<div class="society-rate-item">
    <span class="society-rate-name">{society}</span>
    <span class="society-rate-value">~{rate} Lakh/Marla</span>
</div>
""",
                    unsafe_allow_html=True,
                )

    st.divider()
    st.markdown(
        """
    **📌 Note:** Prices are refreshed hourly using
    live 2026 market feeds (including Zameen indices)
    with baseline fallback for unmatched areas.
    Actual prices can still vary by exact location,
    plot number, road access, and market conditions.
    """
    )

# Footer
st.divider()
st.markdown(
    """
<div class="footer">
    🏠 Lahore House Price Predictor | Built with ❤️ using Streamlit & scikit-learn<br>
    📊 Historical market data + hourly live 2026 rate refresh<br>
    🇵🇰 Designed for Pakistan's property market
</div>
""",
    unsafe_allow_html=True,
)
