import hashlib
import json
import os
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from statistics import median

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import SGDRegressor

PKR_PER_LAKH = 100000.0
LATEST_BASE_FEATURE_COUNT = 10
LATEST_ONLINE_FEATURE_COUNT = 11
ONLINE_CALIBRATION_VERSION = "2026_03_expected_marla_v2"

ZAMEEN_INDICES_API = "https://legion.zameen.com/api/public/indices"
ZAMEEN_LAHORE_LOCATION_ID = 1
ZAMEEN_TYPE_ID_PLOT = 12
ZAMEEN_TYPE_ID_HOUSE = 9
ZAMEEN_FETCH_TYPE_PRIORITY = (ZAMEEN_TYPE_ID_PLOT, ZAMEEN_TYPE_ID_HOUSE)
ZAMEEN_PURPOSE_ID_BUY = 1
ZAMEEN_PER_PAGE = 100
MARLA_TO_SQFT = 225.0

GENERIC_NAME_TOKENS = {
    "and", "housing", "society", "scheme", "city", "town", "phase", "sector", "block",
    "defence", "road", "extension", "cooperative", "co", "government", "foundation",
}

ZAMEEN_SOCIETY_ALIASES = {
    "DHA Phase 1-4": ["DHA Defence"],
    "DHA Phase 5": ["DHA Defence"],
    "DHA Phase 6": ["DHA Defence"],
    "DHA Phase 7-8": ["DHA Defence"],
    "DHA Phase 9 (Prism)": ["DHA Defence"],
    "DHA Phase 10-11": ["DHA 11 Rahbar", "DHA Defence"],
    "DHA Phase 13": ["DHA City", "DHA Defence"],
    "Askari 10": ["Askari"],
    "Askari 11": ["Askari"],
    "Bahria Town (Sector A-D)": ["Bahria Town"],
    "Bahria Town (Sector E-F)": ["Bahria Town"],
    "Iqbal Town": ["Allama Iqbal Town"],
    "Lake City": ["Lake City Meadows"],
    "NFC Society": ["NFC 1", "NFC 2"],
    "Punjab Govt Servants Society": ["Punjab Government Servant Housing Foundation"],
    "Green Town": ["Green Town Sector D2"],
    "Valencia Housing": ["Valencia Housing Society"],
    "Sui Gas Society": ["Sui Gas Housing Society"],
    "PCSIR Housing Society": ["PCSIR Housing Scheme"],
    "Audit & Accounts Society": ["Audit & Accounts Housing Society"],
    "Architects Engineers Society": ["Architects Engineers Housing Society"],
}


def safe_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def resolve_feature_vector(model_obj, latest_vector, legacy_vector, default_feature_count):
    feature_count = int(getattr(model_obj, "n_features_in_", default_feature_count))
    if feature_count == len(latest_vector):
        return latest_vector
    if feature_count == len(legacy_vector):
        return legacy_vector
    if feature_count < len(latest_vector):
        return latest_vector[:feature_count]
    return latest_vector + ([0.0] * (feature_count - len(latest_vector)))


def build_sidebar_society_matches(societies, query_lower):
    prefix_matches = [society for society in societies if society.lower().startswith(query_lower)]
    word_prefix_matches = [
        society for society in societies
        if society not in prefix_matches and any(word.startswith(query_lower) for word in society.lower().split())
    ]
    contains_matches = [
        society for society in societies
        if society not in prefix_matches and society not in word_prefix_matches and query_lower in society.lower()
    ]
    return prefix_matches + word_prefix_matches + contains_matches


def expected_house_profile_by_marla(marla):
    marla = safe_float(marla, 5)
    if marla <= 3:
        return 2, 2, 0, 0, 1, 0
    if marla <= 5:
        return 3, 3, 1, 1, 1, 1
    if marla <= 7:
        return 4, 4, 1, 1, 1, 1
    if marla <= 10:
        return 5, 4, 2, 1, 2, 1
    if marla <= 15:
        return 5, 5, 2, 2, 2, 2
    return 6, 6, 3, 2, 2, 2


def estimate_layout_features(marla, bedrooms):
    _, _, garage, terrace, kitchens, drawing_rooms = expected_house_profile_by_marla(marla)
    if safe_float(bedrooms, 3) >= 6:
        kitchens = min(3, kitchens + 1)
        drawing_rooms = min(3, drawing_rooms + 1)
    return garage, terrace, kitchens, drawing_rooms


def calculate_market_anchored_price(rate_lakh, marla, bedrooms, bathrooms, garage, terrace, kitchens, drawing_rooms, age, condition):
    rate_lakh = max(safe_float(rate_lakh, 0.0), 0.0)
    marla = max(safe_float(marla, 0.0), 0.0)
    base_lakhs = rate_lakh * marla

    expected_bedrooms, expected_bathrooms, expected_garage, expected_terrace, expected_kitchens, expected_drawing = expected_house_profile_by_marla(marla)

    adjustment_ratio = 0.0
    adjustment_ratio += (safe_float(bedrooms, expected_bedrooms) - expected_bedrooms) * 0.025
    adjustment_ratio += (safe_float(bathrooms, expected_bathrooms) - expected_bathrooms) * 0.020
    adjustment_ratio += (safe_float(garage, expected_garage) - expected_garage) * 0.028
    adjustment_ratio += (safe_float(terrace, expected_terrace) - expected_terrace) * 0.017
    adjustment_ratio += (safe_float(kitchens, expected_kitchens) - expected_kitchens) * 0.015
    adjustment_ratio += (safe_float(drawing_rooms, expected_drawing) - expected_drawing) * 0.020

    adjustment_ratio += (safe_float(condition, 3) - 3.0) * 0.030
    adjustment_ratio -= max(safe_float(age, 0), 0) * 0.0045
    adjustment_ratio = float(np.clip(adjustment_ratio, -0.35, 0.30))

    total_lakhs = base_lakhs * (1.0 + adjustment_ratio)
    return max(total_lakhs * PKR_PER_LAKH, 500000)


def format_update_timestamp(updated_at_iso):
    if not updated_at_iso:
        return "Unknown"
    try:
        parsed = datetime.fromisoformat(str(updated_at_iso).replace("Z", "+00:00"))
        return parsed.astimezone().strftime("%d-%b-%Y %I:%M:%S %p")
    except ValueError:
        return str(updated_at_iso)


def normalize_society_name(name):
    if name is None:
        return ""
    text = str(name).lower().replace("&", " and ").replace("govt", "government")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def extract_society_tokens(name):
    tokens = []
    for token in normalize_society_name(name).split():
        if len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        if token not in GENERIC_NAME_TOKENS:
            tokens.append(token)
    return set(tokens)


def select_best_society_match(base_name, available_names):
    base_tokens = extract_society_tokens(base_name)
    if not base_tokens:
        return None

    normalized_base = normalize_society_name(base_name)
    best_name, best_score = None, 0.0
    for candidate_name in available_names:
        candidate_tokens = extract_society_tokens(candidate_name)
        if not candidate_tokens:
            continue

        overlap = len(base_tokens & candidate_tokens) / len(base_tokens)
        if overlap < 0.75:
            continue

        similarity = SequenceMatcher(None, normalized_base, normalize_society_name(candidate_name)).ratio()
        score = (overlap * 0.65) + (similarity * 0.35)
        if score > best_score:
            best_score, best_name = score, candidate_name

    return best_name if best_score >= 0.80 else None


def fetch_zameen_lahore_rates_per_marla_lakhs(timeout_seconds=12, type_id=ZAMEEN_TYPE_ID_PLOT):
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; LahoreHousePricePredictor/1.0)"})

    page = 1
    location_rate_buckets = {}
    while True:
        params = {
            "q[location_p_eq]": ZAMEEN_LAHORE_LOCATION_ID,
            "q[purpose_id_eq]": ZAMEEN_PURPOSE_ID_BUY,
            "q[type_id_eq]": type_id,
            "per_page": ZAMEEN_PER_PAGE,
            "page": page,
        }
        response = session.get(ZAMEEN_INDICES_API, params=params, timeout=timeout_seconds)
        response.raise_for_status()
        payload = response.json()

        rows = payload.get("indices", [])
        if not isinstance(rows, list) or not rows:
            break

        for row in rows:
            if row.get("land_size") is not None:
                continue

            location = row.get("location") or {}
            if int(location.get("p", 0) or 0) != ZAMEEN_LAHORE_LOCATION_ID:
                continue

            location_title = str(location.get("title") or "").strip()
            price_per_sqft = safe_float(row.get("avg_price_per_sqft"))
            if not location_title or price_per_sqft is None or price_per_sqft <= 0:
                continue

            rate_lakh_per_marla = (price_per_sqft * MARLA_TO_SQFT) / PKR_PER_LAKH
            location_rate_buckets.setdefault(location_title, []).append(rate_lakh_per_marla)

        next_page = (payload.get("pagination") or {}).get("next_page")
        if not next_page:
            break

        page = int(next_page)
        if page > 200:
            break

    return {
        location_name: round(float(median(rate_values)), 2)
        for location_name, rate_values in location_rate_buckets.items() if rate_values
    }


def fetch_and_map_zameen_rates_by_priority(base_societies, timeout_seconds=12):
    mapped_rates = {}
    matched_type_ids = {}
    unmatched_societies = list(base_societies.keys())

    for zameen_type_id in ZAMEEN_FETCH_TYPE_PRIORITY:
        if not unmatched_societies:
            break

        zameen_rates_by_location = fetch_zameen_lahore_rates_per_marla_lakhs(
            timeout_seconds=timeout_seconds,
            type_id=zameen_type_id,
        )
        if not zameen_rates_by_location:
            continue

        subset_societies = {name: base_societies[name] for name in unmatched_societies if name in base_societies}
        currently_mapped, _ = map_zameen_rates_to_baseline_societies(subset_societies, zameen_rates_by_location)

        if not currently_mapped:
            continue

        mapped_rates.update(currently_mapped)
        matched_type_ids.update({society_name: zameen_type_id for society_name in currently_mapped.keys()})
        unmatched_societies = [name for name in unmatched_societies if name not in currently_mapped]

    return mapped_rates, unmatched_societies, matched_type_ids


def map_zameen_rates_to_baseline_societies(base_societies, zameen_rates_by_location):
    if not zameen_rates_by_location:
        return {}, []

    lower_lookup = {name.lower(): name for name in zameen_rates_by_location.keys()}
    normalized_lookup = {}
    for location_name in zameen_rates_by_location.keys():
        normalized_lookup.setdefault(normalize_society_name(location_name), []).append(location_name)

    mapped_rates, unmatched_societies = {}, []
    for base_society_name in base_societies.keys():
        candidate_names = [base_society_name] + ZAMEEN_SOCIETY_ALIASES.get(base_society_name, [])
        selected_location_names = []

        for candidate_name in candidate_names:
            exact_match = lower_lookup.get(candidate_name.lower())
            if exact_match:
                selected_location_names.append(exact_match)
            selected_location_names.extend(normalized_lookup.get(normalize_society_name(candidate_name), []))

        if not selected_location_names:
            fuzzy_match = select_best_society_match(base_society_name, zameen_rates_by_location.keys())
            if fuzzy_match:
                selected_location_names.append(fuzzy_match)

        selected_location_names = list(dict.fromkeys(selected_location_names))
        if not selected_location_names:
            unmatched_societies.append(base_society_name)
            continue

        selected_rates = [safe_float(zameen_rates_by_location[name]) for name in selected_location_names if name in zameen_rates_by_location]
        selected_rates = [rate for rate in selected_rates if rate is not None]
        if not selected_rates:
            unmatched_societies.append(base_society_name)
            continue

        mapped_rates[base_society_name] = round(float(median(selected_rates)), 2)

    return mapped_rates, unmatched_societies


def normalize_market_rates_payload(payload):
    rates_block = payload.get("rates", payload) if isinstance(payload, dict) else payload
    payload_updated_at = payload.get("updated_at") if isinstance(payload, dict) else None
    normalized_rates = {}

    def add_rate_entry(society_name, rate_value, tier_value):
        if not society_name:
            return
        parsed_rate = safe_float(rate_value)
        if parsed_rate is None or parsed_rate <= 0:
            return
        normalized_rates[society_name] = {"avg_price_per_marla_lakhs": parsed_rate, "tier": tier_value}

    if isinstance(rates_block, dict):
        for society_name, value in rates_block.items():
            if isinstance(value, dict):
                rate_value = value.get("avg_price_per_marla_lakhs")
                if rate_value is None:
                    rate_value = value.get("rate_per_marla_lakhs", value.get("rate_lakhs", value.get("rate")))
                tier = value.get("tier")
            else:
                rate_value, tier = value, None
            add_rate_entry(society_name, rate_value, tier)

    if isinstance(rates_block, list):
        for item in rates_block:
            if not isinstance(item, dict):
                continue
            society_name = item.get("society") or item.get("name")
            rate_value = item.get("avg_price_per_marla_lakhs")
            if rate_value is None:
                rate_value = item.get("rate_per_marla_lakhs", item.get("rate_lakhs", item.get("rate")))
            add_rate_entry(society_name, rate_value, item.get("tier"))

    return normalized_rates, payload_updated_at


def merge_normalized_rates(merged_societies, normalized_rates, allow_new_societies=True):
    if not normalized_rates:
        return 0

    existing_name_map = {normalize_society_name(name): name for name in merged_societies.keys()}
    updated_count = 0
    for incoming_name, incoming_info in normalized_rates.items():
        incoming_key = normalize_society_name(incoming_name)
        canonical_name = existing_name_map.get(incoming_key, incoming_name)
        incoming_tier = incoming_info.get("tier")
        incoming_rate = safe_float(incoming_info.get("avg_price_per_marla_lakhs"))
        if incoming_rate is None or incoming_rate <= 0:
            continue

        if canonical_name not in merged_societies:
            if not allow_new_societies:
                continue
            merged_societies[canonical_name] = {
                "avg_price_per_marla_lakhs": incoming_rate,
                "tier": incoming_tier if incoming_tier else "affordable",
            }
            existing_name_map[normalize_society_name(canonical_name)] = canonical_name
            updated_count += 1
            continue

        merged_societies[canonical_name]["avg_price_per_marla_lakhs"] = incoming_rate
        if incoming_tier:
            merged_societies[canonical_name]["tier"] = incoming_tier
        updated_count += 1

    return updated_count


def load_live_society_data_core(base_societies, market_rates_url, enable_zameen_auto_rates=True, zameen_timeout_seconds=12):
    merged_societies = json.loads(json.dumps(base_societies))
    source = "Local baseline (society_data.json)"
    updated_at_iso = datetime.now(timezone.utc).isoformat()
    warning_message = None
    has_live_update = False

    if market_rates_url:
        try:
            response = requests.get(market_rates_url, timeout=10)
            response.raise_for_status()
            payload = response.json()

            normalized_rates, payload_updated_at = normalize_market_rates_payload(payload)
            updated_count = merge_normalized_rates(merged_societies, normalized_rates, allow_new_societies=True)
            if updated_count > 0:
                has_live_update = True
                source = f"Live 2026 market feed ({market_rates_url})"
                updated_at_iso = str(payload_updated_at) if payload_updated_at else datetime.now(timezone.utc).isoformat()
            else:
                warning_message = "Configured market feed returned no valid society rates."
        except Exception as exc:
            warning_message = str(exc)

    if not has_live_update and enable_zameen_auto_rates:
        try:
            mapped_rates, unmatched_societies, matched_type_ids = fetch_and_map_zameen_rates_by_priority(
                base_societies,
                timeout_seconds=zameen_timeout_seconds,
            )

            if mapped_rates:
                zameen_normalized_rates = {
                    society_name: {
                        "avg_price_per_marla_lakhs": rate_lakh,
                        "tier": base_societies.get(society_name, {}).get("tier"),
                    }
                    for society_name, rate_lakh in mapped_rates.items()
                }
                merge_normalized_rates(merged_societies, zameen_normalized_rates, allow_new_societies=False)

                mapped_count, total_count = len(mapped_rates), len(base_societies)
                plot_mapped_count = sum(1 for type_id in matched_type_ids.values() if type_id == ZAMEEN_TYPE_ID_PLOT)
                house_fallback_count = sum(1 for type_id in matched_type_ids.values() if type_id == ZAMEEN_TYPE_ID_HOUSE)
                if house_fallback_count > 0:
                    source = (
                        "Live Zameen Lahore indices 2026 "
                        f"(plots {plot_mapped_count}, house fallback {house_fallback_count}; mapped {mapped_count}/{total_count})"
                    )
                else:
                    source = f"Live Zameen Lahore indices 2026 (residential plots; mapped {mapped_count}/{total_count})"
                updated_at_iso = datetime.now(timezone.utc).isoformat()

                if not market_rates_url:
                    warning_message = None
                elif warning_message:
                    warning_message = f"Configured feed failed; Zameen fallback is active. Details: {warning_message}"

                if mapped_count < max(10, int(total_count * 0.50)):
                    warning_message = (
                        f"Low live mapping coverage from Zameen ({mapped_count}/{total_count}); "
                        "remaining societies are using baseline rates."
                    )
                elif unmatched_societies:
                    sample_unmatched = ", ".join(unmatched_societies[:5])
                    more_label = "" if len(unmatched_societies) <= 5 else ", ..."
                    warning_message = (
                        f"Official Zameen indices are unavailable for {len(unmatched_societies)} societies "
                        f"({sample_unmatched}{more_label}); these are using baseline rates."
                    )
            elif not warning_message:
                warning_message = "Zameen live indices returned no usable Lahore society rates."
        except Exception as exc:
            if warning_message:
                warning_message = f"{warning_message} | Zameen fallback: {exc}"
            else:
                warning_message = f"Zameen live indices unavailable. Using local baseline rates. Details: {exc}"

    return merged_societies, source, updated_at_iso, warning_message


def build_online_training_arrays(training_frame, encoder, rate_reference):
    required_columns = {"Society", "Marla", "Bedrooms", "Bathrooms", "Age (years)", "Condition (1-5)", "Price (PKR)"}
    if not required_columns.issubset(set(training_frame.columns)):
        return None, None

    filtered = training_frame[training_frame["Society"].isin(set(encoder.classes_))].copy()
    if filtered.empty:
        return None, None

    filtered["MarketRateLakh"] = filtered["Society"].map(lambda name: rate_reference.get(name, {}).get("avg_price_per_marla_lakhs"))
    filtered = filtered.dropna(subset=["MarketRateLakh"])
    if filtered.empty:
        return None, None

    marla_values = filtered["Marla"].to_numpy(dtype=float)
    bedroom_values = filtered["Bedrooms"].to_numpy(dtype=float)
    bathroom_values = filtered["Bathrooms"].to_numpy(dtype=float)
    age_values = filtered["Age (years)"].to_numpy(dtype=float)
    condition_values = filtered["Condition (1-5)"].to_numpy(dtype=float)

    estimated_layout = [estimate_layout_features(marla, bedrooms) for marla, bedrooms in zip(marla_values, bedroom_values)]
    estimated_garage = pd.Series([item[0] for item in estimated_layout], index=filtered.index, dtype=float)
    estimated_terrace = pd.Series([item[1] for item in estimated_layout], index=filtered.index, dtype=float)
    estimated_kitchens = pd.Series([item[2] for item in estimated_layout], index=filtered.index, dtype=float)
    estimated_drawing_rooms = pd.Series([item[3] for item in estimated_layout], index=filtered.index, dtype=float)

    def pick_feature_values(column_name, estimated_series):
        if column_name in filtered.columns:
            return pd.to_numeric(filtered[column_name], errors="coerce").fillna(estimated_series).to_numpy(dtype=float)
        return estimated_series.to_numpy(dtype=float)

    garage_values = pick_feature_values("Garage", estimated_garage)
    terrace_values = pick_feature_values("Terrace", estimated_terrace)
    kitchen_values = pick_feature_values("Kitchens", estimated_kitchens)
    drawing_room_values = pick_feature_values("Drawing Rooms", estimated_drawing_rooms)

    society_encoded = encoder.transform(filtered["Society"])
    X_online = np.column_stack([
        society_encoded, marla_values, bedroom_values, bathroom_values, garage_values, terrace_values,
        kitchen_values, drawing_room_values, age_values, condition_values, filtered["MarketRateLakh"].to_numpy(dtype=float),
    ])
    y_online_lakh = filtered["Price (PKR)"].to_numpy(dtype=float) / PKR_PER_LAKH
    return X_online, y_online_lakh


def is_online_model_prediction_scale_valid(model_online, encoder, rate_reference):
    known_societies = [society for society in encoder.classes_ if society in rate_reference]
    if not known_societies:
        return False

    sample_society = known_societies[0]
    sample_rate = safe_float(rate_reference[sample_society].get("avg_price_per_marla_lakhs"))
    if sample_rate is None or sample_rate <= 0:
        return False

    encoded_society = encoder.transform([sample_society])[0]
    sample_vector = np.array([[encoded_society, 5, 3, 2, 1, 1, 1, 1, 5, 3, sample_rate]], dtype=float)
    try:
        sample_prediction_lakh = float(model_online.predict(sample_vector)[0])
    except Exception:
        return False
    return np.isfinite(sample_prediction_lakh) and 5 <= sample_prediction_lakh <= 5000


def initialize_online_model(online_model_file, dataset_file, encoder, rate_reference, state_file=None):
    calibration_state = {}
    if state_file and os.path.exists(state_file):
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                calibration_state = json.load(f)
        except (json.JSONDecodeError, OSError):
            calibration_state = {}

    model_version_matches = calibration_state.get("calibration_version") == ONLINE_CALIBRATION_VERSION

    if os.path.exists(online_model_file) and model_version_matches:
        try:
            loaded_model = joblib.load(online_model_file)
            if isinstance(loaded_model, SGDRegressor) and getattr(loaded_model, "n_features_in_", LATEST_ONLINE_FEATURE_COUNT) == LATEST_ONLINE_FEATURE_COUNT and is_online_model_prediction_scale_valid(loaded_model, encoder, rate_reference):
                return loaded_model
        except Exception:
            pass

    if not os.path.exists(dataset_file):
        return None

    try:
        training_frame = pd.read_csv(dataset_file)
    except Exception:
        return None

    X_online, y_online = build_online_training_arrays(training_frame, encoder, rate_reference)
    if X_online is None or len(X_online) == 0:
        return None

    model_online = SGDRegressor(loss="huber", penalty="l2", alpha=0.0001, learning_rate="adaptive", eta0=1e-3, max_iter=3500, tol=1e-3, random_state=42)
    model_online.fit(X_online, y_online)
    if not is_online_model_prediction_scale_valid(model_online, encoder, rate_reference):
        return None

    joblib.dump(model_online, online_model_file)
    return model_online


def compute_market_signature(society_rates):
    compact_rates = {society: round(float(info.get("avg_price_per_marla_lakhs", 0)), 4) for society, info in sorted(society_rates.items())}
    payload = json.dumps(compact_rates, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_market_calibration_batch(society_rates, encoder):
    profiles = [
        (3, 2, 2, 0, 0, 1, 0, 4, 3),
        (5, 3, 3, 1, 1, 1, 1, 5, 3),
        (7, 4, 4, 1, 1, 1, 1, 6, 3),
        (10, 5, 4, 2, 1, 2, 1, 7, 4),
        (15, 5, 5, 2, 2, 2, 2, 8, 4),
        (20, 6, 6, 3, 2, 2, 2, 10, 4),
    ]
    known_societies = set(encoder.classes_)
    rows, targets = [], []

    for society_name, info in society_rates.items():
        if society_name not in known_societies:
            continue

        current_rate_lakh = safe_float(info.get("avg_price_per_marla_lakhs", 0))
        if current_rate_lakh is None or current_rate_lakh <= 0:
            continue

        encoded_society = encoder.transform([society_name])[0]
        for marla, bedrooms, bathrooms, garage, terrace, kitchens, drawing_rooms, age, condition in profiles:
            rows.append([encoded_society, marla, bedrooms, bathrooms, garage, terrace, kitchens, drawing_rooms, age, condition, current_rate_lakh])
            targets.append(calculate_market_anchored_price(current_rate_lakh, marla, bedrooms, bathrooms, garage, terrace, kitchens, drawing_rooms, age, condition) / PKR_PER_LAKH)

    if not rows:
        return None, None
    return np.array(rows, dtype=float), np.array(targets, dtype=float)


def apply_online_update(online_model, society_rates, encoder, state_file, online_model_file):
    if online_model is None:
        return False, None

    market_signature = compute_market_signature(society_rates)
    state = {}
    if os.path.exists(state_file):
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
        except (json.JSONDecodeError, OSError):
            state = {}

    calibration_changed = state.get("calibration_version") != ONLINE_CALIBRATION_VERSION

    if not calibration_changed and state.get("last_market_signature") == market_signature:
        return False, state.get("updated_at")

    X_batch, y_batch = build_market_calibration_batch(society_rates, encoder)
    if X_batch is None or len(X_batch) == 0:
        return False, state.get("updated_at")

    # Apply stronger correction after calibration logic updates.
    fit_passes = 4 if calibration_changed else 1
    for _ in range(fit_passes):
        online_model.partial_fit(X_batch, y_batch)
    joblib.dump(online_model, online_model_file)

    updated_at = datetime.now(timezone.utc).isoformat()
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "last_market_signature": market_signature,
                "updated_at": updated_at,
                "calibration_version": ONLINE_CALIBRATION_VERSION,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return True, updated_at
