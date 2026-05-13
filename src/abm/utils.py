"""
ماژول توابع کمکی برای ABM تخصیص تاکسی آنلاین.

این ماژول شامل توابع پایه است که در سراسر سیستم استفاده می‌شوند:
- haversine_km: محاسبه فاصله جغرافیایی روی سطح کره
- traffic_factor: محاسبه ضریب ترافیک ساعت‌محور (طبق فصل ۳ بخش ۳-۵-۲)
- weather_factor: محاسبه ضرایب تأثیر آب‌و‌هوا بر ترافیک/تقاضا/عرضه
- zone_lookup: تعیین نزدیک‌ترین ناحیه به یک مختصات
- load_targets / load_zones / load_config: خواندن فایل‌های پیکربندی
- map_weather: نگاشت ۵ وضعیت هوای داده واقعی به ۴ دسته فصل ۳
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ============================================================================
# ثابت‌ها
# ============================================================================

EARTH_RADIUS_KM: float = 6371.0088

# نگاشت ۵ کلاس آب‌وهوای داده‌ی واقعی به ۴ کلاس فصل ۳
# Dusty در فصل ۳ تعریف نشده → به cloudy نگاشت می‌شود
# Light_Rain در فصل ۳ صریح نیست → به rainy نگاشت می‌شود
WEATHER_MAP_RAW_TO_CHAPTER3: Dict[str, str] = {
    "Clear": "clear",
    "Cloudy": "cloudy",
    "Dusty": "cloudy",
    "Light_Rain": "rainy",
    "Heavy_Rain": "heavy_rain",
}

WEATHER_CATEGORIES: Tuple[str, ...] = ("clear", "cloudy", "rainy", "heavy_rain")


# ============================================================================
# فاصله جغرافیایی
# ============================================================================

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    محاسبه فاصله Haversine بین دو نقطه روی سطح کره (کیلومتر).

    ورودی:
        lat1, lon1: عرض و طول جغرافیایی نقطه اول (درجه)
        lat2, lon2: عرض و طول جغرافیایی نقطه دوم (درجه)

    خروجی:
        فاصله بزرگ‌دایره بین دو نقطه به کیلومتر.

    نکته: این تابع برای محاسبات پیوسته در ABM طراحی شده. برای
    ماتریس فاصله از نسخه برداری haversine_matrix_km استفاده شود.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


def haversine_matrix_km(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    محاسبه برداری ماتریس فاصله Haversine بین مجموعه‌ای از نقاط.

    ورودی:
        lats: آرایه (n,) عرض جغرافیایی نقاط (درجه)
        lons: آرایه (n,) طول جغرافیایی نقاط (درجه)

    خروجی:
        ماتریس فاصله (n, n) به کیلومتر؛ قطر اصلی صفر است.
    """
    lats_rad = np.radians(lats)
    lons_rad = np.radians(lons)
    dphi = lats_rad[:, None] - lats_rad[None, :]
    dlam = lons_rad[:, None] - lons_rad[None, :]
    a = (np.sin(dphi / 2.0) ** 2
         + np.cos(lats_rad[:, None]) * np.cos(lats_rad[None, :]) * np.sin(dlam / 2.0) ** 2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


# ============================================================================
# مدل ترافیک
# ============================================================================

def traffic_factor(hour: int) -> float:
    """
    ضریب ترافیک ساعت‌محور طبق فصل ۳ بخش ۳-۵-۲.

    ورودی:
        hour: ساعت روز در بازه [0, 23]

    خروجی:
        ضریب ضربی روی زمان سفر:
          - ۱.۴ در ساعات اوج (۷-۹ صبح و ۱۷-۲۰ عصر)
          - ۰.۸ در ساعات خلوت (۲۲-۶ بامداد)
          - ۱.۰ در سایر ساعات

    این ضریب روی *زمان* اعمال می‌شود: travel_time = (dist/speed) × factor.
    """
    if hour < 0 or hour > 23:
        raise ValueError(f"hour must be in [0,23], got {hour}")

    if 7 <= hour <= 9 or 17 <= hour <= 20:
        return 1.4
    if hour >= 22 or hour <= 6:
        return 0.8
    return 1.0


def weather_factors(weather: str) -> Dict[str, float]:
    """
    ضرایب تأثیر آب‌و‌هوا بر ترافیک، تقاضا و عرضه راننده.

    ورودی:
        weather: یکی از {clear, cloudy, rainy, heavy_rain}

    خروجی:
        دیکشنری با کلیدهای:
          - traffic: ضرب در زمان سفر
          - demand: ضرب در نرخ ورود درخواست
          - supply: ضرب در تعداد راننده فعال

    منبع: فصل ۳ بخش ۳-۵-۲ برای heavy_rain؛ مقادیر rainy درون‌یابی شده.
    """
    table: Dict[str, Dict[str, float]] = {
        "clear":      {"traffic": 1.00, "demand": 1.00, "supply": 1.00},
        "cloudy":     {"traffic": 1.00, "demand": 1.00, "supply": 1.00},
        "rainy":      {"traffic": 1.10, "demand": 1.05, "supply": 0.97},
        "heavy_rain": {"traffic": 1.43, "demand": 1.20, "supply": 0.90},
    }
    if weather not in table:
        logger.warning("unknown weather '%s', falling back to clear", weather)
        return table["clear"]
    return table[weather]


def map_weather(raw_label: str) -> str:
    """نگاشت برچسب آب‌وهوای داده‌ی واقعی به یکی از ۴ دسته فصل ۳."""
    return WEATHER_MAP_RAW_TO_CHAPTER3.get(raw_label, "clear")


# ============================================================================
# Surge
# ============================================================================

def normalize_surge(surge_multiplier: float,
                    surge_min: float = 1.0,
                    surge_max: float = 2.5) -> float:
    """
    نرمال‌سازی ضریب سرج به بازه [0, 1].

    ورودی:
        surge_multiplier: ضریب سرج خام (>=۱.۰)
        surge_min, surge_max: بازه منطقی صنعت تاکسی (پیش‌فرض ۱.۰ تا ۲.۵)

    خروجی:
        عدد در بازه [0, 1].

    استفاده در فرمول f_surge = 0.5 + 0.5 × normalize_surge(surge) (رابطه ۳-۵-ب).
    """
    rng = surge_max - surge_min
    if rng <= 0:
        return 0.0
    return float(np.clip((surge_multiplier - surge_min) / rng, 0.0, 1.0))


# ============================================================================
# Lookups
# ============================================================================

def zone_lookup(lat: float,
                lon: float,
                zone_lats: np.ndarray,
                zone_lons: np.ndarray) -> int:
    """
    یافتن نزدیک‌ترین مرکز ناحیه به یک مختصات (با Haversine).

    ورودی:
        lat, lon: مختصات نقطه
        zone_lats, zone_lons: آرایه‌های مراکز نواحی

    خروجی:
        شناسه ناحیه (عدد صحیح ۰ تا n_zones-۱)
    """
    n = zone_lats.shape[0]
    dists = np.empty(n, dtype=np.float64)
    for i in range(n):
        dists[i] = haversine_km(lat, lon, float(zone_lats[i]), float(zone_lons[i]))
    return int(np.argmin(dists))


# ============================================================================
# Loaders
# ============================================================================

def load_targets(path: str | Path) -> Dict[str, Any]:
    """خواندن فایل اهداف کالیبراسیون (targets.json)."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("loaded targets from %s (n_records=%s)",
                p, data.get("metadata", {}).get("n_records", "?"))
    return data


def load_zones(path: str | Path) -> Dict[str, Any]:
    """خواندن فایل مراکز نواحی (zones.json)."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    n = data.get("n_zones", len(data.get("zones", [])))
    logger.info("loaded %d zones from %s", n, p)
    return data


def load_config(path: str | Path) -> Dict[str, Any]:
    """خواندن فایل پیکربندی YAML."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info("loaded config from %s", p)
    return cfg


def zones_to_arrays(zones_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    تبدیل ساختار JSON نواحی به آرایه‌های numpy.

    خروجی: (lats, lons, demand_share) — همه float64 با طول n_zones
    """
    zones: List[Dict[str, Any]] = zones_data["zones"]
    lats = np.array([z["center_lat"] for z in zones], dtype=np.float64)
    lons = np.array([z["center_lon"] for z in zones], dtype=np.float64)
    share = np.array([z["demand_share"] for z in zones], dtype=np.float64)
    return lats, lons, share


# ============================================================================
# Logging setup
# ============================================================================

def setup_logging(level: int = logging.INFO) -> None:
    """پیکربندی یکنواخت logging برای کل سیستم."""
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ============================================================================
# اجرای مستقل برای آزمون سریع
# ============================================================================

if __name__ == "__main__":
    setup_logging()
    # تست haversine: تهران تا کرج (~۴۰ کیلومتر)
    d = haversine_km(35.6892, 51.3890, 35.8400, 50.9391)
    logger.info("haversine(Tehran, Karaj) = %.2f km (expected ~40)", d)

    # تست ترافیک
    for h in [3, 8, 12, 18, 23]:
        logger.info("traffic_factor(hour=%d) = %.2f", h, traffic_factor(h))

    # تست هوا
    for w in WEATHER_CATEGORIES:
        logger.info("weather_factors(%s) = %s", w, weather_factors(w))

    # تست surge
    for s in [1.0, 1.5, 2.0, 2.5, 3.0]:
        logger.info("normalize_surge(%.1f) = %.3f", s, normalize_surge(s))
