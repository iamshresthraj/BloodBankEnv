"""
Real-time data fetcher for eRakt Kosh (https://eraktkosh.mohfw.gov.in)
Ministry of Health & Family Welfare, Government of India

Fetches live blood bank stock availability data for use in BloodBankEnv.
Falls back gracefully to synthetic data if the API is unreachable.
"""

import requests
import json
import re
import base64
import time
import random
from typing import Dict, List, Optional, Tuple
from .models import BloodType

BASE_URL = "https://eraktkosh.mohfw.gov.in/BLDAHIMS/bloodbank"
STOCK_ENDPOINT = f"{BASE_URL}/nearbyBB.cnt"

# All Indian states and UTs with their eRakt Kosh codes
STATE_CODES = {
    "Andaman and Nicobar Islands": 35,
    "Andhra Pradesh": 28,
    "Arunachal Pradesh": 12,
    "Assam": 18,
    "Bihar": 10,
    "Chandigarh": 4,
    "Chhattisgarh": 22,
    "Dadra and Nagar Haveli": 26,
    "Daman and Diu": 25,
    "Delhi": 97,
    "Goa": 30,
    "Gujarat": 24,
    "Haryana": 6,
    "Himachal Pradesh": 2,
    "Jammu and Kashmir": 1,
    "Jharkhand": 20,
    "Karnataka": 29,
    "Kerala": 32,
    "Ladakh": 37,
    "Lakshadweep": 31,
    "Madhya Pradesh": 23,
    "Maharashtra": 27,
    "Manipur": 14,
    "Meghalaya": 17,
    "Mizoram": 15,
    "Nagaland": 13,
    "Odisha": 21,
    "Puducherry": 34,
    "Punjab": 3,
    "Rajasthan": 8,
    "Sikkim": 11,
    "Tamil Nadu": 33,
    "Telangana": 36,
    "Tripura": 16,
    "Uttar Pradesh": 9,
    "Uttarakhand": 5,
    "West Bengal": 19,
}

# States with typically higher blood bank activity (faster, better data)
HIGH_ACTIVITY_STATES = [
    "Delhi", "Maharashtra", "Karnataka", "Tamil Nadu", "Telangana",
    "Uttar Pradesh", "Gujarat", "West Bengal", "Rajasthan", "Kerala",
    "Madhya Pradesh", "Punjab", "Haryana", "Andhra Pradesh", "Bihar",
]

# Mapping from eRakt Kosh blood type strings to our BloodType enum
_BT_MAP = {
    "O+": BloodType.O_POS, "O-": BloodType.O_NEG,
    "A+": BloodType.A_POS, "A-": BloodType.A_NEG,
    "B+": BloodType.B_POS, "B-": BloodType.B_NEG,
    "AB+": BloodType.AB_POS, "AB-": BloodType.AB_NEG,
}


def _generate_security_token(params: Dict[str, str]) -> str:
    """Generate the 'abfhttf' security token required by eRakt Kosh API."""
    param_list = [
        {"name": k, "value": str(v)}
        for k, v in params.items()
        if k != "hmode"
    ]
    json_str = json.dumps(param_list, separators=(',', ':'))
    b64 = base64.b64encode(json_str.encode('utf-8')).decode('utf-8')
    return "".join("\\u" + format(ord(c), '04x') for c in b64)


def _parse_availability(html_str: str) -> Dict[str, int]:
    """
    Parse HTML availability string into {blood_type: count} dict.
    e.g. "Available, O-Ve:1, AB+Ve:2, A+Ve:4" → {"O-": 1, "AB+": 2, "A+": 4}
    """
    stock = {}
    if not html_str or "Not Available" in html_str:
        return stock
    matches = re.findall(r'((?:AB|A|B|O)[+-])Ve[:\s]*(\d+)', html_str)
    for bt, count in matches:
        stock[bt] = int(count)
    return stock


def _parse_blood_bank_name(html_str: str) -> str:
    """Extract blood bank name from the HTML details field."""
    text = re.sub(r'<br\s*/?>', '\n', html_str)
    text = re.sub(r'<[^>]+>', '', text)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return lines[0] if lines else "Unknown"


def fetch_state_stock(state_name: str) -> Tuple[List[Dict], str]:
    """
    Fetch real-time blood stock data for a single state from eRakt Kosh.
    
    Returns:
        Tuple of (list of blood bank records, state name used)
        Each record: {
            "name": str,
            "category": str,
            "stock": {BloodType: int, ...},  # units per type
            "total_units": int,
        }
    """
    state_code = STATE_CODES.get(state_name)
    if state_code is None:
        return [], state_name

    params = {
        "stateCode": str(state_code),
        "districtCode": "-1",
        "bloodGroup": "all",
        "bloodComponent": "11",  # Whole Blood
        "lang": "0",
    }
    token = _generate_security_token(params)
    query = {
        "hmode": "GETNEARBYSTOCKDETAILS",
        **params,
        "abfhttf": token,
        "_": str(int(time.time() * 1000)),
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"{BASE_URL}/stockAvailability.cnt",
    }

    response = requests.get(STOCK_ENDPOINT, params=query, headers=headers, timeout=15)
    response.raise_for_status()
    data = response.json()

    results = []
    for row in data.get("data", []):
        if len(row) < 4:
            continue
        raw_stock = _parse_availability(row[3])
        # Convert string keys to BloodType enum keys
        stock = {}
        for bt_str, count in raw_stock.items():
            bt_enum = _BT_MAP.get(bt_str)
            if bt_enum:
                stock[bt_enum] = count

        total = sum(stock.values())
        if total > 0:  # Only include banks that have stock
            results.append({
                "name": _parse_blood_bank_name(row[1]),
                "category": row[2] if row[2] and row[2] != "null" else "Unknown",
                "stock": stock,
                "total_units": total,
            })

    return results, state_name


def fetch_live_inventory(
    preferred_state: Optional[str] = None,
) -> Tuple[Dict[BloodType, int], str, List[str]]:
    """
    Fetch real-time blood bank inventory from eRakt Kosh.
    
    Tries the preferred state first, then falls back to other high-activity
    states. Aggregates stock across all blood banks in the chosen state.
    
    Args:
        preferred_state: State name to try first (random if None).
    
    Returns:
        Tuple of:
        - Dict[BloodType, int]: Aggregated stock per blood type
        - str: Name of the state used as data source
        - List[str]: Names of blood banks that contributed data
    """
    # Build ordered list of states to try
    if preferred_state and preferred_state in STATE_CODES:
        states_to_try = [preferred_state]
    else:
        states_to_try = random.sample(HIGH_ACTIVITY_STATES, len(HIGH_ACTIVITY_STATES))

    last_error = None
    for state in states_to_try:
        try:
            banks, state_name = fetch_state_stock(state)
            if not banks:
                continue

            # Aggregate stock across all blood banks
            aggregated: Dict[BloodType, int] = {bt: 0 for bt in BloodType}
            bank_names = []
            for bank in banks:
                for bt, count in bank["stock"].items():
                    aggregated[bt] += count
                bank_names.append(bank["name"])

            total = sum(aggregated.values())
            if total > 0:
                return aggregated, state_name, bank_names

        except Exception as e:
            last_error = e
            continue

    # If all states failed, raise the last error so caller can fallback
    raise ConnectionError(
        f"Could not fetch live data from eRakt Kosh. Last error: {last_error}"
    )


def compute_live_distribution(stock: Dict[BloodType, int]) -> Dict[BloodType, float]:
    """
    Compute probability distribution from actual stock counts.
    This replaces the hardcoded type_dist with real-world data.
    """
    total = sum(stock.values())
    if total == 0:
        # Return Indian subcontinent defaults as fallback
        return {
            BloodType.O_POS: 0.37, BloodType.B_POS: 0.32,
            BloodType.A_POS: 0.22, BloodType.AB_POS: 0.08,
            BloodType.O_NEG: 0.005, BloodType.B_NEG: 0.002,
            BloodType.A_NEG: 0.002, BloodType.AB_NEG: 0.001,
        }
    return {bt: count / total for bt, count in stock.items()}
