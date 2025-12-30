import csv
import dataclasses
import hashlib
import json
from json import JSONDecodeError
import logging
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import requests

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

# ==========================
# Search Criteria (User Configuration)
# ==========================

# Enter the states you want to search. Can be "ALL" or a comma-separated list.
# Example: States_To_Search = "FL, NY"
States_To_Search = "WA"

# Enter the keywords for the businesses you want to scrape. Comma-separated.
# The system will intelligently map these to OpenStreetMap tags and Overture categories.
# Example: Bussinesses_To_Scrape = "technology repair, pc support"
Bussinesses_To_Scrape = "marketing agency"

# ==========================
# Advanced Configuration
# ==========================

# Minimum Quality Score to keep a record (0-100).
# Higher means we prioritize records with Name, Address, Phone, Website, Email.
# 0 = Keep everything. 50 = Reasonable quality. 80 = High quality (must have most fields).
CONFIG_MIN_QUALITY_SCORE = 10

# Set to True to only keep records that have a website.
CONFIG_REQUIRE_WEBSITE = False
CONFIG_REQUIRE_PHONE = False
CONFIG_REQUIRE_ADDRESS = False

# Data Sources to Run (Set to True/False)
CONFIG_RUN_OSM = True       # OpenStreetMap (Overpass API)
CONFIG_RUN_OVERTURE = False  # Overture Maps (DuckDB/S3) - Requires DuckDB

# Output Options
CONFIG_MERGE_RESULTS = True
CONFIG_SPLIT_BY_STATE = True
CONFIG_OUT_DIR = os.path.join(os.path.dirname(__file__), "runs_smart_scrape")
CONFIG_CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")

# Run Controls
CONFIG_KEEP_ONLY_NAMED = True
CONFIG_TIMEOUT_S = 180
CONFIG_MAX_ATTEMPTS = 3
CONFIG_OVERPASS_ENDPOINTS: Optional[List[str]] = None  # None uses DEFAULT_OVERPASS_ENDPOINTS

# ==========================
# Smart Mapping Logic
# ==========================

# Pre-defined mappings for common business types to OSM tags.
# This helps get accurate results for known categories.
KEYWORD_TO_OSM_TAGS: Dict[str, List[Tuple[str, str]]] = {
    "technology repair": [
        ("shop", "computer|electronics|mobile_phone|robot"),
        ("craft", "electronics_repair"),
        ("office", "it"),
    ],
    "pc support": [
        ("shop", "computer"),
        ("office", "it|consulting"),
    ],
    "computer repair": [
        ("shop", "computer"),
        ("craft", "electronics_repair"),
    ],
    "appliance stores": [
        ("shop", "electronics|appliance|doityourself|department_store"),
    ],
    "appliance store": [
        ("shop", "electronics|appliance|doityourself|department_store"),
    ],
    "hardware store": [
        ("shop", "hardware|doityourself|trade"),
    ],
    "plumber": [
        ("craft", "plumber"),
        ("trade", "plumbing"),
    ],
    "electrician": [
        ("craft", "electrician"),
        ("trade", "electrical"),
    ],
    "marketing agency": [
        ("office", "advertising|marketing|public_relations"),
    ],
    "marketing agencies": [
        ("office", "advertising|marketing|public_relations"),
    ],
    "advertising agency": [
        ("office", "advertising|marketing|public_relations"),
    ],
    "advertising agencies": [
        ("office", "advertising|marketing|public_relations"),
    ],
    "restaurant": [
        ("amenity", "restaurant"),
    ],
    "fast food": [
        ("amenity", "fast_food"),
    ],
    "cafe": [
        ("amenity", "cafe"),
    ],
    "pharmacy": [
        ("shop", "chemist|pharmacy"),
        ("amenity", "pharmacy"),
    ],
    "supermarket": [
        ("shop", "supermarket"),
    ],
    "convenience store": [
        ("shop", "convenience"),
    ],
    "gas station": [
        ("amenity", "fuel"),
    ],
    "bank": [
        ("amenity", "bank"),
    ],
    "hotel": [
        ("tourism", "hotel|motel"),
    ],
    # Education / Vocational
    "vocational": [
        ("amenity", "college|school"),
        ("training", "yes"),
    ],
    "trade school": [
        ("amenity", "college|school"),
    ],
    "technical school": [
        ("amenity", "college|school"),
    ],
    "nursing": [
        ("amenity", "college|school|university"),
        ("healthcare", "nurse"),
        ("name", "nursing"),
    ],
    "welding": [
        ("amenity", "college|school"),
        ("craft", "welder"),
        ("name", "welding"),
    ],
    "cosmetology": [
        ("amenity", "college|school"),
        ("shop", "beauty"),
        ("name", "cosmetology|beauty school|hair school"),
    ],
    "education": [
        ("amenity", "school|college|university|kindergarten"),
        ("office", "educational_institution"),
    ],
    "school": [
        ("amenity", "school|college"),
    ],
    "college": [
        ("amenity", "college"),
    ],
    "university": [
        ("amenity", "university"),
    ],
}

def _generate_osm_selectors(keywords_str: str) -> List[Tuple[str, str]]:
    """
    Generates OSM selectors based on user keywords.
    If a keyword matches a known category, it uses specific tags.
    Otherwise, it generates broad regex searches across common keys.
    """
    selectors: List[Tuple[str, str]] = []
    keywords = [k.strip().lower() for k in keywords_str.split(",") if k.strip()]
    
    for kw in keywords:
        found_mapping = False
        
        # 1. Check exact match in our mapping
        if kw in KEYWORD_TO_OSM_TAGS:
            selectors.extend(KEYWORD_TO_OSM_TAGS[kw])
            found_mapping = True
        
        # 2. Check partial match (if not exact)
        # If a known key (e.g. "welding") is inside the user's keyword (e.g. "welding school")
        if not found_mapping:
            for map_key, map_tags in KEYWORD_TO_OSM_TAGS.items():
                # Only match if the map_key is significant (len > 3) to avoid matching "car" in "care"
                if len(map_key) > 3 and map_key in kw:
                    selectors.extend(map_tags)
                    found_mapping = True
        
        # 3. Fallback / Supplemental: Search for the keyword in common keys
        # We ALWAYS add this for specific names (e.g. "Smith Welding Academy")
        # even if we found a mapping, to ensure we catch name-based matches.
        
        # Escape the keyword for regex safety but keep spaces as literal spaces (nicer for readability in logs)
        safe_kw = re.escape(kw).replace("\\ ", " ")

        # If keyword is plural (very simple heuristic), also search the singular form
        if kw.endswith("ies"):
            singular_kw = kw[:-3] + "y"
        elif kw.endswith("s") and len(kw) > 3:
            singular_kw = kw[:-1]
        else:
            singular_kw = ""

        safe_singular = re.escape(singular_kw).replace("\\ ", " ") if singular_kw else ""
        
        # Search in name (very broad but captures things like "Joe's Technology Repair")
        selectors.append(("name", safe_kw))
        
        # Search in shop, office, craft, amenity, industrial
        selectors.append(("shop", safe_kw))
        selectors.append(("office", safe_kw))
        selectors.append(("craft", safe_kw))
        selectors.append(("amenity", safe_kw))
        selectors.append(("industrial", safe_kw))

        # Add singular token fallbacks if applicable
        if safe_singular:
            selectors.append(("name", safe_singular))
            selectors.append(("shop", safe_singular))
            selectors.append(("office", safe_singular))
            selectors.append(("craft", safe_singular))
            selectors.append(("amenity", safe_singular))
            selectors.append(("industrial", safe_singular))

        # 4. Strip common suffixes for broader name matching
        # e.g. "marketing agency" -> "marketing"
        suffixes = [" agency", " firm", " company", " inc", " llc", " services", " store", " shop", " group", " solutions"]
        stripped_kw = kw
        for suffix in suffixes:
            if stripped_kw.endswith(suffix):
                stripped_kw = stripped_kw[:-len(suffix)].strip()
        
        if stripped_kw and stripped_kw != kw and len(stripped_kw) > 3:
             safe_stripped = re.escape(stripped_kw).replace("\\ ", " ")
             selectors.append(("name", safe_stripped))
             # Also add to other tags just in case
             selectors.append(("office", safe_stripped))
             selectors.append(("shop", safe_stripped))
             selectors.append(("amenity", safe_stripped))
        
    # Deduplicate selectors
    return list(set(selectors))

def _generate_overture_categories(keywords_str: str) -> List[str]:
    """
    Generates Overture category substrings based on user keywords.
    """
    return [k.strip().lower() for k in keywords_str.split(",") if k.strip()]

# ==========================
# Configuration & Constants
# ==========================

STATE_ABBR_TO_NAME: Dict[str, str] = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
    "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
    "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}

CSV_FIELDS: List[str] = [
    "record_id", "osm_type", "osm_id", "state", "state_abbr", "category_shop",
    "name", "brand", "operator", "website", "phone", "email", "address",
    "addr_city", "addr_state", "addr_postcode", "lat", "lon", "opening_hours",
    "source", "collected_at", "excluded", "excluded_reason", "quality_score", "quality_flags",
]

# For this vertical, avoid excluding big brands by default (franchises are desired)
CONFIG_EXCLUDED_TERMS: List[str] = []

DEFAULT_OVERPASS_ENDPOINTS: List[str] = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

OVERPASS_REQUEST_HEADERS: Dict[str, str] = {
    "User-Agent": "CommercialServiceScraper/1.0 (contact: user@example.com) based on Python requests",
    "Accept": "application/json,text/plain;q=0.9,*/*;q=0.8",
}

OVERTURE_S3_PATH = "s3://overturemaps-us-west-2/release/2025-12-17.0/theme=places/type=place/*"

# ==========================
# Helper Functions
# ==========================

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _writable_snapshot_path(path: str) -> str:
    p = _clean_str(path)
    if not p:
        return p
    out_dir = os.path.dirname(p) or "."
    os.makedirs(out_dir, exist_ok=True)
    try:
        with open(p, "a", encoding="utf-8", newline=""):
            pass
        return p
    except PermissionError:
        base, ext = os.path.splitext(os.path.basename(p))
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        alt = os.path.join(out_dir, f"{base}.{ts}{ext or '.csv'}")
        logging.warning("Output file locked, using %s instead of %s", alt, p)
        return alt

def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    return str(value).strip()

def _normalize_text(value: str) -> str:
    value = value.lower()
    value = value.replace("&", " and ")
    value = re.sub(r"\s+", " ", value).strip()
    return value

def _normalize_phone(value: str) -> str:
    v = _clean_str(value)
    if not v:
        return ""
    has_plus = v.strip().startswith("+")
    digits = re.sub(r"\D+", "", v)
    if not digits:
        return ""
    if has_plus:
        return f"+{digits}"
    if len(digits) == 10:
        return f"+1{digits}"
    return digits

def _normalize_website(value: str) -> str:
    v = _clean_str(value)
    if not v:
        return ""
    if v.startswith("http://") or v.startswith("https://"):
        return v
    return f"https://{v}"

def _compile_exclusion_patterns(excluded_terms: Sequence[str]) -> List[re.Pattern[str]]:
    patterns: List[re.Pattern[str]] = []
    for term in excluded_terms:
        t = _normalize_text(term)
        if not t:
            continue
        patterns.append(re.compile(re.escape(t), flags=re.IGNORECASE))
    return patterns

def _match_exclusion(
    *,
    name: str,
    brand: str,
    operator: str,
    excluded_patterns: Sequence[re.Pattern[str]],
) -> Optional[str]:
    haystack = " ".join([name or "", brand or "", operator or ""]).strip()
    if not haystack:
        return None
    normalized = _normalize_text(haystack)
    for pat in excluded_patterns:
        if pat.search(normalized):
            return pat.pattern
    return None

def _compute_quality(*, name: str, brand: str, operator: str, address: str, phone: str, website: str, email: str) -> Tuple[int, str]:
    flags: List[str] = []
    score = 0
    if name or brand or operator:
        score += 20
    else:
        flags.append("missing_identity")
    if address:
        score += 30
    else:
        flags.append("missing_address")
    if phone:
        score += 20
    else:
        flags.append("missing_phone")
    if website:
        score += 15
    else:
        flags.append("missing_website")
    if email:
        score += 15
    else:
        flags.append("missing_email")
    return score, ";".join(flags)

def _parse_states_csv(states_csv: str) -> Set[str]:
    st_raw = (states_csv or "").strip()
    if not st_raw:
        return set()
    if st_raw.lower() in {"all", "*"}:
        return set(STATE_ABBR_TO_NAME.keys())
    return {s.strip().upper() for s in st_raw.split(",") if s.strip()}

@dataclasses.dataclass(frozen=True)
class OSMElementRef:
    element_type: str
    element_id: int

    @property
    def key(self) -> str:
        return f"{self.element_type}/{self.element_id}"

def _build_osm_address(tags: Dict[str, Any]) -> str:
    full = _clean_str(tags.get("addr:full"))
    if full:
        return full
    housenumber = _clean_str(tags.get("addr:housenumber"))
    street = _clean_str(tags.get("addr:street"))
    city = _clean_str(tags.get("addr:city"))
    state = _clean_str(tags.get("addr:state"))
    postcode = _clean_str(tags.get("addr:postcode"))
    first_line = " ".join([p for p in [housenumber, street] if p]).strip()
    second_parts = [p for p in [city, state, postcode] if p]
    second_line = ", ".join(second_parts).strip() if second_parts else ""
    if first_line and second_line:
        return f"{first_line}, {second_line}"
    return first_line or second_line

def _osm_primary_category(tags: Dict[str, Any]) -> str:
    for key in ("shop", "amenity", "office", "industrial", "craft"):
        v = _clean_str(tags.get(key))
        if v:
            return f"{key}:{v}"
    return ""

def _post_overpass(
    *,
    query: str,
    endpoints: Sequence[str],
    timeout_s: int,
    max_attempts: int,
    sleep_base_s: float,
    cache_path: Optional[str],
) -> Dict[str, Any]:
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    last_error: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        endpoint = endpoints[(attempt - 1) % len(endpoints)]
        attempt_started = time.time()
        logging.info("Overpass attempt %s/%s -> %s (timeout=%ss)", attempt, max_attempts, endpoint, timeout_s)
        try:
            resp = requests.post(
                endpoint,
                data={"data": query},
                headers=OVERPASS_REQUEST_HEADERS,
                timeout=timeout_s,
            )
            resp.raise_for_status()
            logging.info(
                "Overpass attempt %s/%s succeeded in %.1fs", attempt, max_attempts, time.time() - attempt_started
            )
            try:
                payload = resp.json()
            except JSONDecodeError as exc:
                preview = (resp.text or "").strip().replace("\n", " ")[:220]
                raise RuntimeError(f"Non-JSON response (status={resp.status_code}) preview={preview!r}") from exc
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f)
            return payload
        except Exception as exc:
            last_error = exc
            status_code: Optional[int] = None
            if isinstance(exc, requests.HTTPError) and getattr(exc, "response", None) is not None:
                try:
                    status_code = int(exc.response.status_code)
                except Exception:
                    status_code = None

            sleep_s = sleep_base_s * (2 ** (attempt - 1)) + random.uniform(0, 1.25)
            if status_code in {429, 502, 503, 504}:
                sleep_s = max(sleep_s, 12.0 + random.uniform(0.0, 4.0))

            logging.warning(
                "Overpass error %s (attempt %s/%s, %.1fs elapsed): %s",
                endpoint,
                attempt,
                max_attempts,
                time.time() - attempt_started,
                repr(exc),
            )
            time.sleep(sleep_s)

    raise RuntimeError("Overpass failed") from last_error

# ==========================
# OSM Discovery Logic
# ==========================

def run_osm_discovery(
    *,
    out_csv: str,
    cache_dir: str,
    states: Sequence[str],
    selectors: Sequence[Tuple[str, str]],
    timeout_s: int,
    max_attempts: int,
    excluded_terms: Sequence[str],
    keep_only_named: bool,
    endpoints: Optional[Sequence[str]],
) -> None:
    out_csv = _writable_snapshot_path(out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    excluded_patterns = _compile_exclusion_patterns(excluded_terms)
    endpoints_final = list(endpoints or DEFAULT_OVERPASS_ENDPOINTS)
    random.shuffle(endpoints_final)  # Shuffle to spread load across mirrors

    seen: Set[str] = set()

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for state_abbr in states:
            state_name = STATE_ABBR_TO_NAME.get(state_abbr, state_abbr)
            logging.info("Querying OSM for %s (selectors=%s)", state_abbr, len(selectors))

            iso = f"US-{state_abbr}"

            selector_lines: List[str] = []
            for k, regex in selectors:
                # Use case-insensitive regex matching
                selector_lines.append(f"  nwr[\"{k}\"~\"{regex}\",i](area.a);")

            query = (
                f"[out:json][timeout:{timeout_s}][maxsize:2073741824];\n"
                f"rel[\"ISO3166-2\"=\"{iso}\"][admin_level=4];\n"
                f"map_to_area -> .a;\n"
                f"(\n"
                + "\n".join(selector_lines)
                + "\n);\n"
                f"out center tags;"
            )

            # Create a hash of the query to ensure unique cache per search criteria
            query_hash = hashlib.md5(query.encode("utf-8")).hexdigest()[:8]
            cache_path = os.path.join(cache_dir, f"osm_{state_abbr}_{query_hash}.json")
            try:
                query_started = time.time()
                payload = _post_overpass(
                    query=query,
                    endpoints=endpoints_final,
                    timeout_s=timeout_s + 30,
                    max_attempts=max_attempts,
                    sleep_base_s=1.5,
                    cache_path=cache_path,
                )
                logging.info("OSM %s: query finished in %.1fs", state_abbr, time.time() - query_started)
            except Exception as e:
                logging.error("Failed OSM query for %s: %s", state_abbr, e)
                continue

            elements = payload.get("elements") or []
            total_elements = len(elements)
            progress_every = max(100, total_elements // 10 or 1)
            collected_at_now = _utc_now_iso()
            count = 0

            logging.info("OSM %s: %s candidates returned", state_abbr, total_elements)

            for idx, el in enumerate(elements, start=1):
                el_type = el.get("type")
                el_id = el.get("id")
                if not el_type or not el_id:
                    continue

                ref = OSMElementRef(el_type, el_id)
                if ref.key in seen:
                    continue
                seen.add(ref.key)

                tags = el.get("tags") or {}
                name = _clean_str(tags.get("name"))
                brand = _clean_str(tags.get("brand"))
                operator = _clean_str(tags.get("operator"))

                if keep_only_named and not (name or brand or operator):
                    continue

                lat = str(el.get("lat") or el.get("center", {}).get("lat", ""))
                lon = str(el.get("lon") or el.get("center", {}).get("lon", ""))
                if not lat or not lon:
                    continue

                website = _normalize_website(_clean_str(tags.get("website") or tags.get("contact:website")))
                phone = _normalize_phone(_clean_str(tags.get("phone") or tags.get("contact:phone")))
                email = _clean_str(tags.get("email") or tags.get("contact:email"))
                address = _build_osm_address(tags)

                excluded_reason = _match_exclusion(name=name, brand=brand, operator=operator, excluded_patterns=excluded_patterns)
                quality_score, quality_flags = _compute_quality(
                    name=name, brand=brand, operator=operator, address=address, phone=phone, website=website, email=email
                )
                
                # Filter by quality score
                if quality_score < CONFIG_MIN_QUALITY_SCORE:
                    continue

                # Filter by website requirement
                if CONFIG_REQUIRE_WEBSITE and not website:
                    continue
                
                # Filter by phone requirement
                if CONFIG_REQUIRE_PHONE and not phone:
                    continue

                # Filter by address requirement
                if CONFIG_REQUIRE_ADDRESS and not address:
                    continue

                row = {
                    "record_id": ref.key,
                    "osm_type": el_type,
                    "osm_id": str(el_id),
                    "state": state_name,
                    "state_abbr": state_abbr,
                    "category_shop": _osm_primary_category(tags),
                    "name": name,
                    "brand": brand,
                    "operator": operator,
                    "website": website,
                    "phone": phone,
                    "email": email,
                    "address": address,
                    "addr_city": _clean_str(tags.get("addr:city")),
                    "addr_state": _clean_str(tags.get("addr:state")) or state_abbr,
                    "addr_postcode": _clean_str(tags.get("addr:postcode")),
                    "lat": lat,
                    "lon": lon,
                    "opening_hours": _clean_str(tags.get("opening_hours")),
                    "source": "osm_overpass",
                    "collected_at": collected_at_now,
                    "excluded": "1" if excluded_reason else "0",
                    "excluded_reason": excluded_reason or "",
                    "quality_score": str(quality_score),
                    "quality_flags": quality_flags,
                }
                writer.writerow(row)
                count += 1

                if idx % progress_every == 0 or idx == total_elements:
                    logging.info(
                        "OSM %s: scanned %s/%s, kept %s (>= quality %s)",
                        state_abbr,
                        idx,
                        total_elements,
                        count,
                        CONFIG_MIN_QUALITY_SCORE,
                    )

            logging.info("OSM %s: %s rows (Quality >= %s)", state_abbr, count, CONFIG_MIN_QUALITY_SCORE)
            if count == 0 and CONFIG_REQUIRE_WEBSITE:
                logging.warning("  -> 0 rows found. Note: CONFIG_REQUIRE_WEBSITE is True. Many businesses in OSM lack website tags.")

# ==========================
# Overture Maps Logic
# ==========================

def run_overture_discovery(
    *,
    out_csv: str,
    states: Sequence[str],
    excluded_terms: Sequence[str],
    category_substrings: Sequence[str],
) -> None:
    if not HAS_DUCKDB:
        logging.error("DuckDB not installed. Cannot run Overture discovery.")
        return

    out_csv = _writable_snapshot_path(out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    excluded_patterns = _compile_exclusion_patterns(excluded_terms)

    logging.info("Initializing DuckDB for Overture Maps query...")
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL spatial; LOAD spatial;")

    # Configure for anonymous S3 access to Overture bucket
    con.execute("SET s3_region='us-west-2';")
    con.execute("SET s3_access_key_id='';")
    con.execute("SET s3_secret_access_key='';")

    state_filter_sql = ""
    if states and "ALL" not in [s.upper() for s in states]:
        state_list = ", ".join([f"'{s}'" for s in states])
        state_filter_sql = f"AND addresses[1].region IN ({state_list})"

    # Substring matching for categories.primary
    cat_predicates = []
    for sub in category_substrings:
        s = _clean_str(sub).lower()
        if not s:
            continue
        cat_predicates.append(f"lower(categories.primary) LIKE '%{s.replace("'", "''")}%'")

    category_filter_sql = ""
    if cat_predicates:
        category_filter_sql = "AND (" + " OR ".join(cat_predicates) + ")"

    query = f"""
        SELECT 
            id,
            names.primary as name,
            categories.primary as category,
            addresses[1].freeform as address,
            addresses[1].locality as city,
            addresses[1].region as state,
            addresses[1].postcode as postcode,
            websites[1] as website,
            phones[1] as phone,
            ST_Y(geometry) as lat,
            ST_X(geometry) as lon,
            confidence,
            sources
        FROM read_parquet('{OVERTURE_S3_PATH}')
        WHERE addresses[1].country = 'US'
        {state_filter_sql}
        {category_filter_sql}
    """

    logging.info("Running Overture query...")
    results = con.execute(query).fetchall()
    cols = [c[0] for c in con.description]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        count = 0
        for r in results:
            row_data = dict(zip(cols, r))
            name = _clean_str(row_data.get("name"))
            address = _clean_str(row_data.get("address"))
            website = _normalize_website(_clean_str(row_data.get("website")))
            phone = _normalize_phone(_clean_str(row_data.get("phone")))
            category = _clean_str(row_data.get("category"))

            excluded_reason = _match_exclusion(name=name, brand="", operator="", excluded_patterns=excluded_patterns)
            q_score, q_flags = _compute_quality(
                name=name,
                brand="",
                operator="",
                address=address,
                phone=phone,
                website=website,
                email="",
            )
            
            if q_score < CONFIG_MIN_QUALITY_SCORE:
                continue

            if CONFIG_REQUIRE_WEBSITE and not website:
                continue

            if CONFIG_REQUIRE_PHONE and not phone:
                continue

            if CONFIG_REQUIRE_ADDRESS and not address:
                continue

            record_id = f"overture/{_clean_str(row_data.get('id'))}"
            out_row = {
                "record_id": record_id,
                "osm_type": "",
                "osm_id": "",
                "state": STATE_ABBR_TO_NAME.get(_clean_str(row_data.get("state")), _clean_str(row_data.get("state"))),
                "state_abbr": _clean_str(row_data.get("state")),
                "category_shop": f"overture:{category}" if category else "overture",
                "name": name,
                "brand": "",
                "operator": "",
                "website": website,
                "phone": phone,
                "email": "",
                "address": address,
                "addr_city": _clean_str(row_data.get("city")),
                "addr_state": _clean_str(row_data.get("state")),
                "addr_postcode": _clean_str(row_data.get("postcode")),
                "lat": _clean_str(row_data.get("lat")),
                "lon": _clean_str(row_data.get("lon")),
                "opening_hours": "",
                "source": "overture",
                "collected_at": _utc_now_iso(),
                "excluded": "1" if excluded_reason else "0",
                "excluded_reason": excluded_reason or "",
                "quality_score": str(q_score),
                "quality_flags": q_flags,
            }
            writer.writerow(out_row)
            count += 1

    logging.info("Overture: wrote %s rows (Quality >= %s)", count, CONFIG_MIN_QUALITY_SCORE)

# ==========================
# Merge
# ==========================

def merge_csvs(input_paths: List[str], out_csv: str) -> None:
    out_csv = _writable_snapshot_path(out_csv)
    seen = set()
    rows: List[Dict[str, str]] = []
    for p in input_paths:
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid = row.get("record_id")
                if rid and rid in seen:
                    continue
                if rid:
                    seen.add(rid)
                rows.append(row)

    if not rows:
        logging.info("No rows to merge")
        return

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    logging.info("Merged %s rows into %s", len(rows), out_csv)

def create_clean_filtered_csv(input_csv: str, output_csv: str) -> None:
    """
    Creates a filtered version of the CSV containing only rows that have
    values for category_shop, name, website, phone, and address.
    """
    if not os.path.exists(input_csv):
        logging.warning("Input file for cleaning not found: %s", input_csv)
        return

    logging.info("Creating clean filtered CSV: %s", output_csv)
    
    required_fields = ["category_shop", "name", "website", "phone", "address"]
    count = 0
    
    try:
        with open(input_csv, "r", encoding="utf-8", newline="") as f_in, \
             open(output_csv, "w", encoding="utf-8", newline="") as f_out:
            
            reader = csv.DictReader(f_in)
            if not reader.fieldnames:
                logging.warning("Input CSV has no header")
                return
                
            writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
            writer.writeheader()
            
            for row in reader:
                # Check if all required fields have data
                if all(row.get(field, "").strip() for field in required_fields):
                    writer.writerow(row)
                    count += 1
                    
        logging.info("Cleaned CSV created with %s rows.", count)
        
    except Exception as e:
        logging.error("Failed to create clean CSV: %s", e)

# ==========================
# Main
# ==========================

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    states = _parse_states_csv(States_To_Search)
    if not states:
        logging.error("No states configured. Set States_To_Search to 'ALL' or e.g. 'DE,RI'.")
        return

    excluded_terms = [t.strip() for t in (CONFIG_EXCLUDED_TERMS or []) if t and t.strip()]
    
    # Generate selectors dynamically
    osm_selectors = _generate_osm_selectors(Bussinesses_To_Scrape)
    overture_categories = _generate_overture_categories(Bussinesses_To_Scrape)
    
    logging.info(f"Searching for businesses: {Bussinesses_To_Scrape}")
    logging.info(f"Generated OSM Selectors: {osm_selectors}")

    generated_files: List[str] = []

    if CONFIG_RUN_OSM:
        logging.info("Starting OSM Discovery...")
        if CONFIG_SPLIT_BY_STATE:
            for state_abbr in sorted(list(states)):
                osm_out = os.path.join(CONFIG_OUT_DIR, f"smart_osm_{state_abbr}.csv")
                run_osm_discovery(
                    out_csv=osm_out,
                    cache_dir=os.path.join(CONFIG_CACHE_DIR, "osm"),
                    states=[state_abbr],
                    selectors=osm_selectors,
                    timeout_s=CONFIG_TIMEOUT_S,
                    max_attempts=CONFIG_MAX_ATTEMPTS,
                    excluded_terms=excluded_terms,
                    keep_only_named=CONFIG_KEEP_ONLY_NAMED,
                    endpoints=CONFIG_OVERPASS_ENDPOINTS,
                )
                generated_files.append(osm_out)
        else:
            osm_out = os.path.join(CONFIG_OUT_DIR, "smart_osm.csv")
            run_osm_discovery(
                out_csv=osm_out,
                cache_dir=os.path.join(CONFIG_CACHE_DIR, "osm"),
                states=sorted(list(states)),
                selectors=osm_selectors,
                timeout_s=CONFIG_TIMEOUT_S,
                max_attempts=CONFIG_MAX_ATTEMPTS,
                excluded_terms=excluded_terms,
                keep_only_named=CONFIG_KEEP_ONLY_NAMED,
                endpoints=CONFIG_OVERPASS_ENDPOINTS,
            )
            generated_files.append(osm_out)

    if CONFIG_RUN_OVERTURE:
        logging.info("Starting Overture Discovery...")
        if CONFIG_SPLIT_BY_STATE:
            for state_abbr in sorted(list(states)):
                overture_out = os.path.join(CONFIG_OUT_DIR, f"smart_overture_{state_abbr}.csv")
                run_overture_discovery(
                    out_csv=overture_out,
                    states=[state_abbr],
                    excluded_terms=excluded_terms,
                    category_substrings=overture_categories,
                )
                generated_files.append(overture_out)
        else:
            overture_out = os.path.join(CONFIG_OUT_DIR, "smart_overture.csv")
            run_overture_discovery(
                out_csv=overture_out,
                states=sorted(list(states)),
                excluded_terms=excluded_terms,
                category_substrings=overture_categories,
            )
            generated_files.append(overture_out)

    if CONFIG_MERGE_RESULTS and generated_files:
        merge_out = os.path.join(CONFIG_OUT_DIR, "smart_merged.csv")
        merge_csvs(generated_files, merge_out)

        # Create clean version
        clean_out = os.path.join(CONFIG_OUT_DIR, "clean_scraped_data.csv")
        create_clean_filtered_csv(merge_out, clean_out)

    logging.info("All tasks completed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("\n\nProcess interrupted by user (Ctrl+C). Exiting gracefully...")
        sys.exit(0)
