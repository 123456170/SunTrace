import streamlit as st
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2
import io
import os
import math
import time
import tempfile
import datetime
import json
from PIL import Image


st.set_page_config(
    page_title="SunTrace",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Global reproducibility and scene constants
# -----------------------------------------------------------------------------
SEED = 20261016
IMG_W, IMG_H = 640, 480
HORIZON_Y = 245

# Ground marker square in meters: x = east, y = north, stick base at (0, 0).
SRC_SQUARE = np.float32(
    [
        [-0.5, -0.5],
        [0.5, -0.5],
        [0.5, 0.5],
        [-0.5, 0.5],
    ]
)

# Matching pixel quadrilateral for the synthetic camera view.
DST_SQUARE = np.float32(
    [
        [165, 415],
        [475, 415],
        [535, 245],
        [105, 245],
    ]
)

H_GROUND = cv2.getPerspectiveTransform(SRC_SQUARE, DST_SQUARE)


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def as_pydatetime(dt):
    """Convert pandas Timestamp to Python datetime when needed."""
    if isinstance(dt, pd.Timestamp):
        return dt.to_pydatetime()
    return dt


def dt_hours_utc(dt):
    """Fractional UTC hours from a naive UTC datetime/Timestamp."""
    dt = as_pydatetime(dt)
    return (
        dt.hour
        + dt.minute / 60.0
        + dt.second / 3600.0
        + dt.microsecond / 3_600_000_000.0
    )


def angle_diff_deg(a, b):
    """Smallest absolute difference between two angles in degrees."""
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometers."""
    if not all(np.isfinite([lat1, lon1, lat2, lon2])):
        return float("nan")
    r = 6371.0088
    p1 = math.radians(float(lat1))
    p2 = math.radians(float(lat2))
    dp = p2 - p1
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.asin(math.sqrt(a))


def percentile_interval(arr, level):
    """Return central percentile interval, e.g. level=0.68 or 0.95."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [float("nan"), float("nan")]
    alpha = (1.0 - float(level)) / 2.0
    lo = float(np.percentile(arr, alpha * 100.0))
    hi = float(np.percentile(arr, (1.0 - alpha) * 100.0))
    return [lo, hi]


# -----------------------------------------------------------------------------
# Solar geometry: Spencer equations from the project specification
# -----------------------------------------------------------------------------
def day_of_year(dt):
    dt = as_pydatetime(dt)
    return dt.timetuple().tm_yday


def g_angle(dt):
    """Day angle g in radians."""
    doy = day_of_year(dt)
    hu = dt_hours_utc(dt)
    return 2.0 * math.pi / 365.0 * (doy - 1 + (hu - 12.0) / 24.0)


def eot_minutes(g):
    """Equation of time in minutes."""
    return 229.18 * (
        0.000075
        + 0.001868 * math.cos(g)
        - 0.032077 * math.sin(g)
        - 0.014615 * math.cos(2 * g)
        - 0.040849 * math.sin(2 * g)
    )


def declination_rad(g):
    """Solar declination in radians."""
    return (
        0.006918
        - 0.399912 * math.cos(g)
        + 0.070257 * math.sin(g)
        - 0.006758 * math.cos(2 * g)
        + 0.000907 * math.sin(2 * g)
        - 0.002697 * math.cos(3 * g)
        + 0.00148 * math.sin(3 * g)
    )


def solar_alt_az(dt, lat_deg, lon_deg):
    """
    Return solar altitude degrees, solar azimuth degrees from north clockwise,
    equation of time minutes, and declination radians.
    """
    g = g_angle(dt)
    eot = eot_minutes(g)
    dec = declination_rad(g)
    hu = dt_hours_utc(dt)

    # Apparent solar time in hours.
    ast = hu + float(lon_deg) / 15.0 + eot / 60.0
    H = math.radians((ast - 12.0) * 15.0)

    lat = math.radians(float(lat_deg))

    sin_alt = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(H)
    sin_alt = max(-1.0, min(1.0, sin_alt))
    alt = math.asin(sin_alt)

    # Azimuth from north, clockwise.
    az = math.atan2(
        -math.sin(H),
        math.tan(dec) * math.cos(lat) - math.sin(lat) * math.cos(H),
    )
    az_deg = (math.degrees(az) + 360.0) % 360.0

    return math.degrees(alt), az_deg, eot, dec


# -----------------------------------------------------------------------------
# Synthetic data generation
# -----------------------------------------------------------------------------
def generate_track(
    lat_deg,
    lon_deg,
    hemisphere,
    date,
    tz_offset,
    start_local,
    duration_h,
    step_min,
    stick_h,
    noise_m,
    cloud_frac,
    seed,
):
    """
    Generate a realistic synthetic shadow track.

    Coordinates:
      tip_x_m = east in meters relative to stick base
      tip_y_m = north in meters relative to stick base
    """
    start_utc = datetime.datetime.combine(date, datetime.time()) + datetime.timedelta(
        hours=float(start_local) - float(tz_offset)
    )
    n = int(round(float(duration_h) * 60.0 / float(step_min))) + 1
    rng = np.random.default_rng(int(seed))
    rows = []

    for i in range(n):
        dt = start_utc + datetime.timedelta(minutes=i * float(step_min))
        alt, sun_az, _, _ = solar_alt_az(dt, lat_deg, lon_deg)

        q = 1.0
        if alt < 1.0:
            L = float("nan")
            x = float("nan")
            y = float("nan")
            sh_az = float("nan")
            q = 0.0
        else:
            tan_alt = math.tan(math.radians(max(alt, 0.15)))
            L = float(stick_h) / max(tan_alt, 1e-4)
            if L > 200.0:
                L = 200.0

            sh_az = (sun_az + 180.0) % 360.0
            rad = math.radians(sh_az)

            x = L * math.sin(rad) + rng.normal(0.0, float(noise_m))
            y = L * math.cos(rad) + rng.normal(0.0, float(noise_m))

            if rng.random() < float(cloud_frac):
                q = 0.0
                x = float("nan")
                y = float("nan")
                L = float("nan")
                sh_az = float("nan")
            elif rng.random() < 0.08:
                q = 0.45

        rows.append(
            {
                "t_utc": dt,
                "tip_x_m": x,
                "tip_y_m": y,
                "shadow_length_m": L,
                "azimuth_deg": sh_az,
                "quality": q,
            }
        )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Track cleaning and solar-noon estimation
# -----------------------------------------------------------------------------
def preprocess_track(df):
    """Reject bad frames and apply a simple MAD-based outlier filter."""
    d = df.copy()
    d = d[d.quality >= 0.35]
    d = d.dropna(subset=["tip_x_m", "tip_y_m", "shadow_length_m", "azimuth_deg"])

    if len(d) == 0:
        return d

    vals = d.shadow_length_m.values.astype(float)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)) * 1.4826)
    if mad < 1e-3:
        mad = float(np.std(vals) * 0.1 + 1e-3)

    out = np.abs(vals - med) > 4.0 * mad
    d.loc[out, "quality"] = d.loc[out, "quality"] * 0.2
    d = d[d.quality >= 0.35].sort_values("t_utc").reset_index(drop=True)

    if len(d) > 0:
        d["t_min"] = (d.t_utc - d.t_utc.iloc[0]).dt.total_seconds() / 60.0

    return d


def robust_quadratic(x, y):
    """
    Huber-weighted quadratic fit.

    Returns vertex time, vertex value, residual sigma, and curvature.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    if len(x) < 3:
        return None

    x0 = float(np.mean(x))
    xs = x - x0

    try:
        c = np.polyfit(xs, y, 2)
    except Exception:
        return None

    for _ in range(12):
        r = y - np.polyval(c, xs)
        s = 1.4826 * float(np.median(np.abs(r - np.median(r))))
        if s < 1e-6:
            s = float(np.std(r)) + 1e-6
        k = 1.345 * s
        ar = np.abs(r)
        w = np.ones_like(r)
        mask = ar > k
        w[mask] = k / ar[mask]

        X = np.vstack([xs**2, xs, np.ones_like(xs)]).T
        sw = np.sqrt(w)
        try:
            c = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        except Exception:
            break

    a, b, c0 = c

    if a <= 1e-8:
        idx = int(np.argmin(y))
        t_rel = float(xs[idx] + x0)
        y_val = float(y[idx])
        sigma = float(np.std(y - np.polyval(c, xs)))
    else:
        t_rel = float(-b / (2.0 * a) + x0)
        y_val = float(np.polyval(c, t_rel - x0))
        sigma = float(np.std(y - np.polyval(c, xs)))

    return {
        "t_min": t_rel,
        "value": y_val,
        "sigma": sigma,
        "curvature": float(a),
    }


def estimate_solar_noon(d, hemisphere):
    """
    Estimate solar noon from:
      1. robust quadratic fit of shadow length near its minimum
      2. independent azimuth-symmetry crossing
    """
    start = d.t_utc.iloc[0]
    warnings = []

    min_i = d.shadow_length_m.idxmin()
    t_min = float(d.loc[min_i, "t_min"])
    win = d[(d.t_min >= t_min - 45.0) & (d.t_min <= t_min + 45.0)]

    fit = robust_quadratic(win.t_min.values, win.shadow_length_m.values)

    if fit is not None and np.isfinite(fit["value"]) and fit["value"] > 0:
        noon_rel = float(fit["t_min"])
        L_noon = float(fit["value"])
        sigma_L = float(fit["sigma"])
        sigma_t = 1.0 + 10.0 * sigma_L / max(1.0, L_noon)
        sigma_t = min(max(sigma_t, 0.5), 10.0)
    else:
        noon_rel = t_min
        L_noon = float(np.interp(noon_rel, d.t_min.values, d.shadow_length_m.values))
        sigma_L = float(np.std(d.shadow_length_m.values) * 0.2 + 0.01)
        sigma_t = 5.0
        warnings.append("Huber quadratic fit failed; used minimum-length frame.")

    # Independent azimuth symmetry estimate.
    target = 0.0 if hemisphere == "N" else 180.0
    az = d.azimuth_deg.values.astype(float)
    az_center = ((az - target + 180.0) % 360.0) - 180.0

    idx0 = int(np.argmin(np.abs(az_center)))
    left = np.where((np.arange(len(d)) < idx0) & (az_center < 0.0))[0]
    right = np.where((np.arange(len(d)) > idx0) & (az_center > 0.0))[0]

    if len(left) > 0 and len(right) > 0:
        i1 = int(left[-1])
        i2 = int(right[0])
        x1 = float(d.t_min.iloc[i1])
        y1 = float(az_center[i1])
        x2 = float(d.t_min.iloc[i2])
        y2 = float(az_center[i2])
        if abs(y2 - y1) > 1e-6:
            noon_rel_az = x1 + (0.0 - y1) * (x2 - x1) / (y2 - y1)
        else:
            noon_rel_az = float(d.t_min.iloc[idx0])
    else:
        noon_rel_az = float(d.t_min.iloc[idx0])

    noon_L = start + pd.Timedelta(minutes=noon_rel)
    noon_az = start + pd.Timedelta(minutes=noon_rel_az)
    agreement_min = abs((noon_L - noon_az).total_seconds()) / 60.0

    if agreement_min > 2.0:
        warnings.append(
            f"Length-fit noon and azimuth-symmetry noon disagree by {agreement_min:.1f} min."
        )

    az_noon_center = float(np.interp(noon_rel, d.t_min.values, az_center))

    return {
        "noon_utc": noon_L,
        "noon_rel": noon_rel,
        "noon_az_utc": noon_az,
        "L_noon": L_noon,
        "sigma_L": sigma_L,
        "sigma_t_min": sigma_t,
        "az_noon_center": az_noon_center,
        "agreement_min": agreement_min,
        "warnings": warnings,
    }


def locate_coordinates(d, stick_h, hemisphere, noon_info):
    """Compute latitude, longitude, noon altitude, and warnings."""
    noon_utc = noon_info["noon_utc"]
    L_noon = max(float(noon_info["L_noon"]), 1e-3)
    alt_noon = math.degrees(math.atan(float(stick_h) / L_noon))

    g = g_angle(noon_utc)
    eot = eot_minutes(g)
    dec = declination_rad(g)

    hours = dt_hours_utc(noon_utc)
    lon_est = 15.0 * (12.0 - hours - eot / 60.0)
    lon_est = (lon_est + 180.0) % 360.0 - 180.0

    target = 0.0 if hemisphere == "N" else 180.0
    az_noon = (target + float(noon_info["az_noon_center"])) % 360.0

    z = 90.0 - alt_noon
    dec_deg = math.degrees(dec)

    # If the noon shadow points north, the Sun is south: lat = dec + z.
    # If the noon shadow points south, the Sun is north: lat = dec - z.
    if min(az_noon, 360.0 - az_noon) < 90.0:
        lat_est = dec_deg + z
    else:
        lat_est = dec_deg - z

    warnings = []
    if alt_noon > 80.0:
        warnings.append("Noon altitude is above 80 degrees; shadow is very short.")
    if hemisphere == "N" and lat_est < -5.0:
        warnings.append("Northern-hemisphere input produced a southern latitude estimate.")
    if hemisphere == "S" and lat_est > 5.0:
        warnings.append("Southern-hemisphere input produced a northern latitude estimate.")

    return {
        "latitude_deg": float(lat_est),
        "longitude_deg": float(lon_est),
        "altitude_noon_deg": float(alt_noon),
        "declination_deg": float(dec_deg),
        "eot_minutes": float(eot),
        "L_noon_m": float(L_noon),
        "azimuth_noon_deg": float(az_noon),
        "warnings": warnings,
    }


def monte_carlo_intervals(
    d,
    stick_h,
    hemisphere,
    noon_info,
    loc,
    homography_sigma,
    n_draws,
    seed,
):
    """Monte Carlo uncertainty over tip/noise, height, noon time, and homography."""
    rng = np.random.default_rng(int(seed) + 777)
    start = d.t_utc.iloc[0]

    base_rel = float(noon_info["noon_rel"])
    base_L = float(loc["L_noon_m"])
    base_azc = float(noon_info["az_noon_center"])

    sigma_L = math.sqrt(
        max(float(noon_info["sigma_L"]), 0.005) ** 2 + float(homography_sigma) ** 2
    )
    sigma_t = max(0.5, float(noon_info["sigma_t_min"]))

    target = 0.0 if hemisphere == "N" else 180.0
    az = d.azimuth_deg.values.astype(float)
    azc = ((az - target + 180.0) % 360.0) - 180.0

    if len(azc) > 3:
        sigma_az = max(1.0, float(np.std(np.diff(azc)) / math.sqrt(2.0)))
    else:
        sigma_az = 2.0

    lats = []
    lons = []
    rels = []

    for _ in range(int(n_draws)):
        L = base_L + rng.normal(0.0, sigma_L)
        h = float(stick_h) + rng.normal(0.0, 0.002)
        if L <= 0.01 or h <= 0.01:
            continue

        rel = base_rel + rng.normal(0.0, sigma_t)
        noon = start + pd.Timedelta(minutes=rel)
        azc_s = base_azc + rng.normal(0.0, sigma_az)

        alt = math.degrees(math.atan(h / L))
        g = g_angle(noon)
        eot = eot_minutes(g)
        dec = declination_rad(g)

        hours = dt_hours_utc(noon)
        lon = 15.0 * (12.0 - hours - eot / 60.0)
        lon = (lon + 180.0) % 360.0 - 180.0

        az_noon = (target + azc_s) % 360.0
        z = 90.0 - alt
        dec_deg = math.degrees(dec)

        if min(az_noon, 360.0 - az_noon) < 90.0:
            lat = dec_deg + z
        else:
            lat = dec_deg - z

        if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
            lats.append(lat)
            lons.append(lon)
            rels.append(rel)

    if len(lats) < 10:
        lat_arr = np.array([loc["latitude_deg"]], dtype=float)
        lon_arr = np.array([loc["longitude_deg"]], dtype=float)
        rel_arr = np.array([base_rel], dtype=float)
    else:
        lat_arr = np.array(lats, dtype=float)
        lon_arr = np.array(lons, dtype=float)
        rel_arr = np.array(rels, dtype=float)

    return {
        "lat_samples": lat_arr,
        "lon_samples": lon_arr,
        "noon_samples": rel_arr,
        "lat_median": float(np.median(lat_arr)),
        "lon_median": float(np.median(lon_arr)),
        "lat_68": percentile_interval(lat_arr, 0.68),
        "lon_68": percentile_interval(lon_arr, 0.68),
        "lat_95": percentile_interval(lat_arr, 0.95),
        "lon_95": percentile_interval(lon_arr, 0.95),
        "noon_68_min": percentile_interval(rel_arr, 0.68),
        "noon_95_min": percentile_interval(rel_arr, 0.95),
        "sigma_L_m": float(sigma_L),
        "sigma_t_min": float(sigma_t),
        "sigma_az_deg": float(sigma_az),
        "n_effective": int(lat_arr.size),
    }


def analyze_pipeline(raw_df, stick_h, hemisphere, homography_sigma, mc_draws, seed):
    """Full analysis pipeline from track.csv-style DataFrame to results.json."""
    d = preprocess_track(raw_df)

    if len(d) < 5:
        return (
            {
                "error": "Not enough usable frames. Check cloud cover, timestamps, and quality flags.",
                "n_frames_used": int(len(d)),
                "n_frames_raw": int(len(raw_df)),
            },
            d,
            None,
            None,
            None,
        )

    noon = estimate_solar_noon(d, hemisphere)
    loc = locate_coordinates(d, stick_h, hemisphere, noon)
    mc = monte_carlo_intervals(
        d,
        stick_h,
        hemisphere,
        noon,
        loc,
        homography_sigma,
        mc_draws,
        seed,
    )

    removed = 1.0 - float(len(d)) / float(max(1, len(raw_df)))
    warnings = list(noon["warnings"]) + list(loc["warnings"])

    if removed > 0.30:
        warnings.append(
            f"{removed * 100.0:.0f}% of frames were rejected; this exceeds the 30% guidance."
        )

    results = {
        "solar_noon_utc": noon["noon_utc"].isoformat() + "Z",
        "latitude_deg": float(loc["latitude_deg"]),
        "longitude_deg": float(loc["longitude_deg"]),
        "north_azimuth_in_marker_frame_deg": 0.0,
        "intervals": {
            "68_percent": {
                "latitude_deg": mc["lat_68"],
                "longitude_deg": mc["lon_68"],
                "solar_noon_minutes_relative": mc["noon_68_min"],
            },
            "95_percent": {
                "latitude_deg": mc["lat_95"],
                "longitude_deg": mc["lon_95"],
                "solar_noon_minutes_relative": mc["noon_95_min"],
            },
        },
        "n_frames_used": int(len(d)),
        "n_frames_raw": int(len(raw_df)),
        "warnings": warnings,
        "diagnostics": {
            "cloud_rejected_fraction": float(removed),
            "noon_agreement_min": float(noon["agreement_min"]),
            "sigma_L_m": float(noon["sigma_L"]),
            "sigma_t_min": float(noon["sigma_t_min"]),
            "altitude_noon_deg": float(loc["altitude_noon_deg"]),
            "declination_deg": float(loc["declination_deg"]),
            "eot_minutes": float(loc["eot_minutes"]),
            "azimuth_noon_deg": float(loc["azimuth_noon_deg"]),
        },
    }

    return results, d, noon, loc, mc


# -----------------------------------------------------------------------------
# Synthetic image rendering and video generation
# -----------------------------------------------------------------------------
def project_ground(x, y):
    """Project ground meters to pixel coordinates using the marker homography."""
    pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
    uv = cv2.perspectiveTransform(pt, H_GROUND)[0][0]
    return float(uv[0]), float(uv[1])


def create_background():
    """Create a reusable synthetic ground/sky background."""
    img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

    # Sky gradient.
    for yy in range(HORIZON_Y):
        frac = yy / max(1, HORIZON_Y - 1)
        b = int(220 - 80 * frac)
        g = int(170 - 40 * frac)
        r = int(110 + 30 * frac)
        cv2.line(img, (0, yy), (IMG_W, yy), (b, g, r), 1)

    # Ground gradient.
    for yy in range(HORIZON_Y, IMG_H):
        frac = (yy - HORIZON_Y) / max(1, IMG_H - HORIZON_Y - 1)
        b = int(70 + 30 * frac)
        g = int(95 + 25 * frac)
        r = int(65 + 20 * frac)
        cv2.line(img, (0, yy), (IMG_W, yy), (b, g, r), 1)

    # Meter grid.
    for gv in np.arange(-1.5, 1.51, 0.25):
        p1 = project_ground(gv, -1.5)
        p2 = project_ground(gv, 1.5)
        cv2.line(
            img,
            (int(p1[0]), int(p1[1])),
            (int(p2[0]), int(p2[1])),
            (120, 120, 120),
            1,
        )

        p1 = project_ground(-1.5, gv)
        p2 = project_ground(1.5, gv)
        cv2.line(
            img,
            (int(p1[0]), int(p1[1])),
            (int(p2[0]), int(p2[1])),
            (120, 120, 120),
            1,
        )

    # Marked 1 m square.
    pts = np.array([project_ground(x, y) for x, y in SRC_SQUARE], dtype=np.int32)
    cv2.polylines(img, [pts], True, (0, 220, 0), 2)
    cv2.putText(
        img,
        "1 m marked square",
        (int(pts[0][0]), int(pts[0][1]) - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (230, 230, 230),
        1,
    )

    cv2.putText(
        img,
        "SunTrace synthetic scene",
        (10, IMG_H - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (240, 240, 240),
        1,
    )

    return img


BACKGROUND = create_background()


def draw_shadow(img, x_m, y_m, quality):
    """Draw a tapered shadow polygon from stick base to tip."""
    if quality < 0.25:
        return

    if not (np.isfinite(x_m) and np.isfinite(y_m)):
        return

    Lg = math.hypot(float(x_m), float(y_m))
    if not math.isfinite(Lg) or Lg < 1e-3:
        return

    # Keep drawing stable for absurd uploaded values.
    if Lg > 50.0:
        scale = 50.0 / Lg
        x_m = float(x_m) * scale
        y_m = float(y_m) * scale
        Lg = 50.0

    ux = float(x_m) / Lg
    uy = float(y_m) / Lg
    nx = -uy
    ny = ux

    wb = 0.045
    wt = 0.014

    pts_ground = [
        (-nx * wb / 2.0, -ny * wb / 2.0),
        (nx * wb / 2.0, ny * wb / 2.0),
        (float(x_m) + nx * wt / 2.0, float(y_m) + ny * wt / 2.0),
        (float(x_m) - nx * wt / 2.0, float(y_m) - ny * wt / 2.0),
    ]

    pts = np.array([project_ground(px, py) for px, py in pts_ground], dtype=np.int32)

    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], (18, 18, 18))
    cv2.addWeighted(overlay, 0.58, img, 0.42, 0.0, img)


def apply_clouds(img, quality, seed_int):
    """Add simple synthetic cloud haze for low-quality frames."""
    q = float(np.clip(quality, 0.0, 1.0))
    if q >= 0.80:
        return img

    overlay = img.copy()
    rng = np.random.default_rng(int(seed_int) & 0xFFFFFFFF)
    n = int((0.85 - q) * 12.0) + 1

    for _ in range(n):
        cx = int(rng.integers(0, IMG_W))
        cy = int(rng.integers(20, HORIZON_Y + 30))
        ax = int(rng.integers(25, 95))
        ay = int(rng.integers(8, 35))
        cv2.ellipse(overlay, (cx, cy), (ax, ay), 0, 0, 360, (255, 255, 255), -1)

    alpha = 0.25 + 0.45 * (1.0 - q)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0, img)
    return img


def draw_frame(row, stick_h):
    """Render one synthetic camera frame from a track row."""
    img = BACKGROUND.copy()

    try:
        q = float(row["quality"]) if pd.notna(row["quality"]) else 0.0
    except Exception:
        q = 0.0

    try:
        x = float(row["tip_x_m"]) if pd.notna(row["tip_x_m"]) else float("nan")
        y = float(row["tip_y_m"]) if pd.notna(row["tip_y_m"]) else float("nan")
    except Exception:
        x = float("nan")
        y = float("nan")

    draw_shadow(img, x, y, q)

    # Stick.
    base_px = project_ground(0.0, 0.0)
    top_px = (base_px[0], base_px[1] - int(float(stick_h) * 120.0))
    cv2.line(
        img,
        (int(base_px[0]), int(base_px[1])),
        (int(top_px[0]), int(top_px[1])),
        (55, 55, 55),
        4,
    )
    cv2.circle(img, (int(base_px[0]), int(base_px[1])), 5, (0, 0, 255), -1)

    # Clouds.
    try:
        seed_int = int(pd.Timestamp(row["t_utc"]).value)
    except Exception:
        seed_int = 0
    img = apply_clouds(img, q, seed_int)

    # Debug overlay.
    try:
        ts = pd.Timestamp(row["t_utc"]).strftime("%H:%M:%S")
    except Exception:
        ts = "--:--:--"

    cv2.putText(
        img,
        f"{ts} UTC  quality={q:.2f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    if q >= 0.35 and np.isfinite(x) and np.isfinite(y):
        tip_px = project_ground(x, y)
        cv2.circle(img, (int(tip_px[0]), int(tip_px[1])), 4, (255, 0, 255), -1)

    return img


def render_video_from_track(df, stick_h, fps=12, max_frames=90):
    """
    Generate a live video file from the synthetic track.

    Tries browser-friendly WebM/H.264 first, then falls back to MP4/GIF.
    """
    if df.empty:
        return b"", "video/mp4", "none", ".mp4"

    n_frames = min(int(max_frames), len(df))
    idx = np.linspace(0, len(df) - 1, n_frames).astype(int)
    subset = df.iloc[idx]

    specs = [
        ("VP80", ".webm", "video/webm"),
        ("avc1", ".mp4", "video/mp4"),
        ("H264", ".mp4", "video/mp4"),
        ("mp4v", ".mp4", "video/mp4"),
    ]

    for fourcc, ext, mime in specs:
        try:
            with tempfile.TemporaryDirectory() as td:
                path = os.path.join(td, f"suntrace{ext}")
                writer = cv2.VideoWriter(
                    path,
                    cv2.VideoWriter_fourcc(*fourcc),
                    float(fps),
                    (IMG_W, IMG_H),
                )
                if not writer.isOpened():
                    continue

                for _, row in subset.iterrows():
                    img = draw_frame(row, stick_h)
                    writer.write(img)

                writer.release()

                if os.path.exists(path) and os.path.getsize(path) > 1000:
                    with open(path, "rb") as f:
                        return f.read(), mime, fourcc, ext
        except Exception:
            continue

    # GIF fallback.
    try:
        pil_frames = []
        for j, (_, row) in enumerate(subset.iterrows()):
            if j % 2 == 0:
                img = draw_frame(row, stick_h)
                pil_frames.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

        if pil_frames:
            buf = io.BytesIO()
            pil_frames[0].save(
                buf,
                format="GIF",
                save_all=True,
                append_images=pil_frames[1:],
                duration=int(1000 / max(1, fps)),
                loop=0,
            )
            return buf.getvalue(), "image/gif", "GIF", ".gif"
    except Exception:
        pass

    return b"", "video/mp4", "failed", ".mp4"


@st.cache_data(show_spinner="Building SunTrace live demo...")
def build_demo(
    lat,
    lon,
    hem,
    date_iso,
    tz,
    start_local,
    duration,
    step_min,
    stick_h,
    noise_m,
    cloud_frac,
    homography_sigma,
    mc_draws,
    seed,
    fps,
    max_frames,
):
    """Cached end-to-end demo builder."""
    date = datetime.date.fromisoformat(date_iso)

    raw = generate_track(
        lat_deg=lat,
        lon_deg=lon,
        hemisphere=hem,
        date=date,
        tz_offset=tz,
        start_local=start_local,
        duration_h=duration,
        step_min=step_min,
        stick_h=stick_h,
        noise_m=noise_m,
        cloud_frac=cloud_frac,
        seed=seed,
    )

    results, clean, noon, loc, mc = analyze_pipeline(
        raw_df=raw,
        stick_h=stick_h,
        hemisphere=hem,
        homography_sigma=homography_sigma,
        mc_draws=mc_draws,
        seed=seed,
    )

    video_bytes, video_mime, video_codec, video_ext = render_video_from_track(
        raw,
        stick_h,
        fps=fps,
        max_frames=max_frames,
    )

    return {
        "raw": raw,
        "clean": clean,
        "results": results,
        "noon": noon,
        "loc": loc,
        "mc": mc,
        "video_bytes": video_bytes,
        "video_mime": video_mime,
        "video_codec": video_codec,
        "video_ext": video_ext,
    }


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
def make_track_plot(d, noon_info, hemisphere):
    """Plot shadow length and azimuth with solar-noon marker."""
    if d is None or d.empty or noon_info is None:
        return None

    fig, ax = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)

    ax[0].plot(d.t_utc, d.shadow_length_m, marker="o", ms=3, lw=1, label="shadow length")
    ax[0].axvline(noon_info["noon_utc"], color="red", ls="--", label="solar noon")
    ax[0].set_ylabel("L (m)")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    target = 0.0 if hemisphere == "N" else 180.0
    ax[1].plot(d.t_utc, d.azimuth_deg, marker="s", ms=3, lw=1, label="shadow azimuth")
    ax[1].axhline(target, color="green", ls=":", label="noon shadow direction")
    ax[1].axvline(noon_info["noon_utc"], color="red", ls="--")
    ax[1].set_ylabel("azimuth (deg)")
    ax[1].set_xlabel("UTC time")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def make_mc_plot(mc, truth_lat, truth_lon):
    """Plot Monte Carlo posterior-like samples and truth."""
    if mc is None:
        return None

    lat_samples = np.asarray(mc.get("lat_samples", []), dtype=float)
    lon_samples = np.asarray(mc.get("lon_samples", []), dtype=float)

    if lat_samples.size < 5 or lon_samples.size < 5:
        return None

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))

    ax[0].hist(lat_samples, bins=30, color="steelblue", alpha=0.8)
    ax[0].axvline(truth_lat, color="red", lw=2, label="truth")
    ax[0].axvline(mc["lat_median"], color="black", ls="--", label="median")
    ax[0].set_title("Latitude Monte Carlo")
    ax[0].set_xlabel("deg N")
    ax[0].legend()

    ax[1].hist(lon_samples, bins=30, color="seagreen", alpha=0.8)
    ax[1].axvline(truth_lon, color="red", lw=2, label="truth")
    ax[1].axvline(mc["lon_median"], color="black", ls="--", label="median")
    ax[1].set_title("Longitude Monte Carlo")
    ax[1].set_xlabel("deg E")
    ax[1].legend()

    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Synthetic validation
# -----------------------------------------------------------------------------
def validation_sessions():
    """Eight synthetic sessions across hemispheres, seasons, and conditions."""
    return [
        ("Boston summer", 42.3601, -71.0589, "N", datetime.date(2026, 6, 21), -4.0, 0.010, 0.05),
        ("Boston autumn", 42.3601, -71.0589, "N", datetime.date(2026, 10, 16), -4.0, 0.010, 0.10),
        ("Santiago winter", -33.4489, -70.6693, "S", datetime.date(2026, 7, 15), -4.0, 0.015, 0.08),
        ("Sydney summer", -33.8688, 151.2093, "S", datetime.date(2026, 12, 21), 11.0, 0.015, 0.12),
        ("Singapore equatorial", 1.3521, 103.8198, "N", datetime.date(2026, 3, 20), 8.0, 0.020, 0.20),
        ("Nairobi highland", -1.2921, 36.8219, "S", datetime.date(2026, 9, 22), 3.0, 0.020, 0.15),
        ("Reykjavik short shadow", 64.1466, -21.9426, "N", datetime.date(2026, 6, 21), 0.0, 0.010, 0.05),
        ("Cape Town cloudy", -33.9249, 18.4241, "S", datetime.date(2026, 2, 15), 2.0, 0.030, 0.35),
    ]


@st.cache_data(show_spinner="Running 8-session synthetic validation...")
def run_validation(seed):
    """Run synthetic validation and return error table plus failure thumbnails."""
    rows = []
    thumbs = {}

    for i, (
        name,
        lat,
        lon,
        hem,
        date,
        tz,
        noise,
        cloud,
    ) in enumerate(validation_sessions()):
        raw = generate_track(
            lat_deg=lat,
            lon_deg=lon,
            hemisphere=hem,
            date=date,
            tz_offset=tz,
            start_local=10.75,
            duration_h=3.5,
            step_min=1.0,
            stick_h=1.0,
            noise_m=noise,
            cloud_frac=cloud,
            seed=int(seed) + i * 17,
        )

        res, clean, noon, loc, mc = analyze_pipeline(
            raw_df=raw,
            stick_h=1.0,
            hemisphere=hem,
            homography_sigma=0.02,
            mc_draws=200,
            seed=int(seed) + i * 17,
        )

        if isinstance(res, dict) and "error" not in res:
            est_lat = float(res["latitude_deg"])
            est_lon = float(res["longitude_deg"])
            noon_alt = float(loc["altitude_noon_deg"])
            rejected = float(res["diagnostics"]["cloud_rejected_fraction"])
        else:
            est_lat = float("nan")
            est_lon = float("nan")
            noon_alt = float("nan")
            rejected = float("nan")

        lat_err = abs(est_lat - lat) if np.isfinite(est_lat) else float("nan")
        lon_err = angle_diff_deg(est_lon, lon) if np.isfinite(est_lon) else float("nan")
        km_err = haversine_km(lat, lon, est_lat, est_lon)

        condition = (
            f"cloud={cloud:.0%}, noise={noise * 100:.1f} cm, "
            f"noon_alt={noon_alt:.1f} deg, rejected={rejected:.0%}"
        )

        rows.append(
            {
                "session": name,
                "truth_lat": lat,
                "truth_lon": lon,
                "est_lat": est_lat,
                "est_lon": est_lon,
                "lat_error_deg": lat_err,
                "lon_error_deg": lon_err,
                "km_error": km_err,
                "conditions": condition,
            }
        )

        # Failure-gallery thumbnail near estimated noon.
        if clean is not None and not clean.empty and noon is not None:
            try:
                target = noon["noon_utc"]
                idx = (clean.t_utc - target).abs().idxmin()
                row = clean.loc[idx]
                img = draw_frame(row, 1.0)
                ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
                if ok:
                    thumbs[name] = enc.tobytes()
            except Exception:
                pass

    return pd.DataFrame(rows), thumbs


# -----------------------------------------------------------------------------
# Self-tests
# -----------------------------------------------------------------------------
def self_test_astro():
    """Loose published-value checks for Spencer EoT and declination."""
    checks = []

    cases = [
        (datetime.date(2026, 3, 20), 0.0, -7.5, 0.6, 3.0),
        (datetime.date(2026, 6, 21), 23.44, -1.8, 0.4, 3.0),
        (datetime.date(2026, 9, 22), 0.0, 7.5, 0.6, 3.0),
        (datetime.date(2026, 11, 3), -15.0, 16.4, 0.8, 3.0),
        (datetime.date(2026, 12, 21), -23.44, 1.9, 0.4, 3.0),
    ]

    for d, exp_dec, exp_eot, tol_dec, tol_eot in cases:
        dt = datetime.datetime(d.year, d.month, d.day, 12, 0, 0)
        g = g_angle(dt)
        dec = math.degrees(declination_rad(g))
        eot = eot_minutes(g)

        checks.append(
            {
                "test": f"Declination {d.isoformat()}",
                "passed": bool(abs(dec - exp_dec) <= tol_dec),
                "value": f"{dec:.3f} deg",
                "expected": f"{exp_dec:.3f} +/- {tol_dec:.3f} deg",
            }
        )
        checks.append(
            {
                "test": f"EoT {d.isoformat()}",
                "passed": bool(abs(eot - exp_eot) <= tol_eot),
                "value": f"{eot:.3f} min",
                "expected": f"{exp_eot:.3f} +/- {tol_eot:.3f} min",
            }
        )

    return checks


def self_test_homography():
    """Check that the marker homography round-trips the known square."""
    checks = []
    Hinv = cv2.getPerspectiveTransform(DST_SQUARE, SRC_SQUARE)

    for i, (dst_pt, src_pt) in enumerate(zip(DST_SQUARE, SRC_SQUARE)):
        p = np.array([[[float(dst_pt[0]), float(dst_pt[1])]]], dtype=np.float32)
        back = cv2.perspectiveTransform(p, Hinv)[0][0]
        err = math.hypot(float(back[0] - src_pt[0]), float(back[1] - src_pt[1]))
        checks.append(
            {
                "test": f"Homography corner {i}",
                "passed": bool(err < 1e-3),
                "value": f"error={err:.6f} m",
                "expected": "< 0.001 m",
            }
        )

    return checks


def self_test_noon_fit():
    """Robust quadratic should recover the vertex of a noisy parabola."""
    rng = np.random.default_rng(SEED + 31)
    x = np.linspace(0.0, 60.0, 61)
    y = 0.01 * (x - 30.0) ** 2 + 5.0 + rng.normal(0.0, 0.02, size=x.shape)
    fit = robust_quadratic(x, y)

    if fit is None:
        return [
            {
                "test": "Solar-noon quadratic fit",
                "passed": False,
                "value": "fit failed",
                "expected": "vertex near 30 min",
            }
        ]

    return [
        {
            "test": "Solar-noon quadratic fit",
            "passed": bool(abs(fit["t_min"] - 30.0) < 1.0),
            "value": f"vertex={fit['t_min']:.3f} min",
            "expected": "30 +/- 1 min",
        }
    ]


def self_test_roundtrip():
    """Synthetic round trip: known location/date -> track -> recovered location."""
    raw = generate_track(
        lat_deg=40.0,
        lon_deg=-75.0,
        hemisphere="N",
        date=datetime.date(2026, 4, 15),
        tz_offset=-5.0,
        start_local=10.75,
        duration_h=3.0,
        step_min=1.0,
        stick_h=1.0,
        noise_m=0.005,
        cloud_frac=0.0,
        seed=SEED + 101,
    )

    res, _, _, _, _ = analyze_pipeline(
        raw_df=raw,
        stick_h=1.0,
        hemisphere="N",
        homography_sigma=0.01,
        mc_draws=50,
        seed=SEED + 101,
    )

    if "error" in res:
        return [
            {
                "test": "Synthetic round trip",
                "passed": False,
                "value": res["error"],
                "expected": "lat within 0.5 deg, lon within 0.5 deg",
            }
        ]

    lat_err = abs(res["latitude_deg"] - 40.0)
    lon_err = angle_diff_deg(res["longitude_deg"], -75.0)

    return [
        {
            "test": "Synthetic round trip latitude",
            "passed": bool(lat_err < 0.5),
            "value": f"error={lat_err:.4f} deg",
            "expected": "< 0.5 deg",
        },
        {
            "test": "Synthetic round trip longitude",
            "passed": bool(lon_err < 0.5),
            "value": f"error={lon_err:.4f} deg",
            "expected": "< 0.5 deg",
        },
    ]


@st.cache_data(show_spinner="Running self-tests...")
def run_self_tests(seed):
    """Run all built-in tests."""
    checks = []
    checks.extend(self_test_astro())
    checks.extend(self_test_homography())
    checks.extend(self_test_noon_fit())
    checks.extend(self_test_roundtrip())
    return pd.DataFrame(checks)


# -----------------------------------------------------------------------------
# Uploaded real-track loader
# -----------------------------------------------------------------------------
def load_uploaded_track(file):
    """Load and validate a user-provided track.csv."""
    df = pd.read_csv(file)
    required = [
        "t_utc",
        "tip_x_m",
        "tip_y_m",
        "shadow_length_m",
        "azimuth_deg",
        "quality",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["t_utc"] = pd.to_datetime(df["t_utc"], utc=True, errors="coerce").dt.tz_localize(None)
    for c in required[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["t_utc"])
    return df


# -----------------------------------------------------------------------------
# Sidebar configuration
# -----------------------------------------------------------------------------
DEFAULTS = {
    "lat": 42.3601,
    "lon": -71.0589,
    "hem": "N",
    "date": datetime.date(2026, 10, 16),
    "tz": -4.0,
    "start_local": 10.5,
    "duration": 3.0,
    "step": 1.0,
    "stick": 1.0,
    "noise": 0.010,
    "cloud": 0.08,
    "homography_sigma": 0.020,
    "mc": 1000,
    "seed": SEED,
    "fps": 12,
    "max_frames": 90,
}

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

with st.sidebar:
    st.title("SunTrace")
    st.caption(
        "Estimate latitude, longitude, and true north from a vertical-stick shadow track. "
        "This demo uses synthetic data by default and can also process an uploaded track.csv."
    )

    st.header("Location and time")
    st.session_state.lat = st.number_input(
        "True latitude, deg N positive",
        value=float(st.session_state.lat),
        step=0.01,
        key="lat_widget",
    )
    st.session_state.lon = st.number_input(
        "True longitude, deg E positive",
        value=float(st.session_state.lon),
        step=0.01,
        key="lon_widget",
    )
    st.session_state.hem = st.radio(
        "Hemisphere",
        ["N", "S"],
        index=0 if st.session_state.hem == "N" else 1,
        horizontal=True,
        key="hem_widget",
    )
    st.session_state.date = st.date_input(
        "Date",
        value=st.session_state.date,
        key="date_widget",
    )
    st.session_state.tz = st.number_input(
        "Timezone offset, hours",
        value=float(st.session_state.tz),
        step=0.5,
        key="tz_widget",
    )
    st.session_state.start_local = st.slider(
        "Start local time, hours",
        min_value=6.0,
        max_value=18.0,
        value=float(st.session_state.start_local),
        step=0.25,
        key="start_widget",
    )
    st.session_state.duration = st.slider(
        "Duration, hours",
        min_value=1.0,
        max_value=6.0,
        value=float(st.session_state.duration),
        step=0.5,
        key="duration_widget",
    )
    st.session_state.step = st.select_slider(
        "Frame interval, minutes",
        options=[0.5, 1.0, 2.0, 5.0, 10.0],
        value=float(st.session_state.step),
        key="step_widget",
    )

    st.header("Stick, noise, uncertainty")
    st.session_state.stick = st.number_input(
        "Stick height, m",
        value=float(st.session_state.stick),
        min_value=0.05,
        step=0.01,
        key="stick_widget",
    )
    st.session_state.noise = st.slider(
        "Tip noise, m",
        min_value=0.0,
        max_value=0.10,
        value=float(st.session_state.noise),
        step=0.005,
        key="noise_widget",
    )
    st.session_state.cloud = st.slider(
        "Cloudy frame fraction",
        min_value=0.0,
        max_value=0.60,
        value=float(st.session_state.cloud),
        step=0.01,
        key="cloud_widget",
    )
    st.session_state.homography_sigma = st.slider(
        "Homography sigma, m",
        min_value=0.0,
        max_value=0.10,
        value=float(st.session_state.homography_sigma),
        step=0.005,
        key="homo_widget",
    )
    st.session_state.mc = st.selectbox(
        "Monte Carlo draws",
        options=[100, 500, 1000, 2000],
        index=[100, 500, 1000, 2000].index(int(st.session_state.mc)),
        key="mc_widget",
    )
    st.session_state.seed = st.number_input(
        "Random seed",
        value=int(st.session_state.seed),
        step=1,
        key="seed_widget",
    )

    st.header("Live video settings")
    st.session_state.fps = st.select_slider(
        "Playback FPS",
        options=[8, 10, 12, 15, 20],
        value=int(st.session_state.fps),
        key="fps_widget",
    )
    st.session_state.max_frames = st.select_slider(
        "Max video frames",
        options=[45, 60, 90, 120],
        value=int(st.session_state.max_frames),
        key="maxframes_widget",
    )

    st.divider()
    st.markdown(
        "**Privacy:** use only permitted photos, avoid faces, process locally. "
        "This project makes no navigation, medical, safety, or legal claims."
    )


# -----------------------------------------------------------------------------
# Build demo automatically on load
# -----------------------------------------------------------------------------
param_key = (
    float(st.session_state.lat),
    float(st.session_state.lon),
    str(st.session_state.hem),
    st.session_state.date.isoformat(),
    float(st.session_state.tz),
    float(st.session_state.start_local),
    float(st.session_state.duration),
    float(st.session_state.step),
    float(st.session_state.stick),
    float(st.session_state.noise),
    float(st.session_state.cloud),
    float(st.session_state.homography_sigma),
    int(st.session_state.mc),
    int(st.session_state.seed),
    int(st.session_state.fps),
    int(st.session_state.max_frames),
)

try:
    demo = build_demo(*param_key)
except Exception as exc:
    st.error(f"Demo build failed: {exc}")
    st.stop()


# -----------------------------------------------------------------------------
# Main UI tabs
# -----------------------------------------------------------------------------
tabs = st.tabs(
    [
        "Live demo",
        "Validation",
        "Capture protocol",
        "Novelty audit",
        "Known limits",
        "Self-tests",
        "Upload track.csv",
    ]
)


# --- Live demo tab -----------------------------------------------------------
with tabs[0]:
    st.header("Live SunTrace demo")
    st.markdown(
        "The app automatically loads a realistic synthetic shadow sequence, renders a video, "
        "estimates solar noon, computes latitude/longitude, and reports Monte Carlo intervals."
    )

    res = demo["results"]
    clean = demo["clean"]
    noon = demo["noon"]
    loc = demo["loc"]
    mc = demo["mc"]
    raw = demo["raw"]

    col_video, col_metrics = st.columns([2.0, 1.0])

    with col_video:
        if demo["video_bytes"]:
            if demo["video_mime"].startswith("video"):
                st.video(demo["video_bytes"], format=demo["video_mime"])
            else:
                st.image(demo["video_bytes"], use_container_width=True)

            st.download_button(
                "Download live demo video",
                data=demo["video_bytes"],
                file_name=f"suntrace_demo{demo['video_ext']}",
                mime=demo["video_mime"],
            )
            st.caption(f"Encoder used: {demo['video_codec']}")
        else:
            st.warning("Video encoder unavailable in this environment. Use the scrubber below.")

    with col_metrics:
        if "error" in res:
            st.error(res["error"])
        else:
            st.metric("Estimated latitude", f"{res['latitude_deg']:.4f} deg")
            st.metric("Estimated longitude", f"{res['longitude_deg']:.4f} deg")
            st.metric(
                "Solar noon UTC",
                res["solar_noon_utc"][11:19] if len(res["solar_noon_utc"]) > 19 else res["solar_noon_utc"],
            )
            st.metric("Frames used", f"{res['n_frames_used']} / {res['n_frames_raw']}")
            st.metric(
                "Noon altitude",
                f"{res['diagnostics']['altitude_noon_deg']:.2f} deg",
            )

            st.subheader("68% intervals")
            st.write("Latitude:", res["intervals"]["68_percent"]["latitude_deg"])
            st.write("Longitude:", res["intervals"]["68_percent"]["longitude_deg"])

            st.subheader("95% intervals")
            st.write("Latitude:", res["intervals"]["95_percent"]["latitude_deg"])
            st.write("Longitude:", res["intervals"]["95_percent"]["longitude_deg"])

    if res.get("warnings"):
        st.subheader("Warnings")
        for w in res["warnings"]:
            st.warning(w)

    st.subheader("Shadow track plots")
    fig1 = make_track_plot(clean, noon, st.session_state.hem)
    if fig1 is not None:
        st.pyplot(fig1)
        plt.close(fig1)
    else:
        st.info("Not enough clean frames to plot.")

    st.subheader("Monte Carlo uncertainty")
    fig2 = make_mc_plot(mc, st.session_state.lat, st.session_state.lon)
    if fig2 is not None:
        st.pyplot(fig2)
        plt.close(fig2)
    else:
        st.info("Monte Carlo samples unavailable.")

    st.subheader("Interactive frame scrubber")
    if len(raw) > 0:
        idx = st.slider(
            "Frame index",
            min_value=0,
            max_value=len(raw) - 1,
            value=int((len(raw) - 1) * 0.5),
            key="scrubber_idx",
        )
        row = raw.iloc[idx]
        img = draw_frame(row, float(st.session_state.stick))
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

        if st.checkbox("Auto-play preview for a few seconds"):
            placeholder = st.empty()
            step = max(1, len(raw) // 60)
            for i in range(0, len(raw), step):
                frame = draw_frame(raw.iloc[i], float(st.session_state.stick))
                placeholder.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                )
                time.sleep(0.08)
    else:
        st.warning("No frames generated.")

    st.subheader("results.json")
    st.download_button(
        "Download results.json",
        data=json.dumps(res, indent=2, default=str).encode("utf-8"),
        file_name="results.json",
        mime="application/json",
    )
    st.json(res)


# --- Validation tab ----------------------------------------------------------
with tabs[1]:
    st.header("Synthetic validation against known truth")
    st.markdown(
        "This is a simulated validation, not real GPS field validation. It checks whether the "
        "mathematical pipeline recovers known synthetic truth under different hemispheres, "
        "seasons, cloud fractions, and noise levels."
    )

    if st.button("Run 8-session synthetic validation"):
        st.session_state.val_df, st.session_state.val_thumbs = run_validation(
            int(st.session_state.seed)
        )

    if "val_df" in st.session_state:
        df = st.session_state.val_df
        thumbs = st.session_state.get("val_thumbs", {})

        st.dataframe(df, use_container_width=True)

        valid = df.dropna(subset=["lat_error_deg", "lon_error_deg", "km_error"])
        if not valid.empty:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean lat error", f"{valid.lat_error_deg.mean():.4f} deg")
            c2.metric("Mean lon error", f"{valid.lon_error_deg.mean():.4f} deg")
            c3.metric("Mean km error", f"{valid.km_error.mean():.3f} km")
            c4.metric("Worst km error", f"{valid.km_error.max():.3f} km")

            st.markdown("**Spread and worst cases**")
            st.write(
                {
                    "lat_error_mean_deg": float(valid.lat_error_deg.mean()),
                    "lat_error_std_deg": float(valid.lat_error_deg.std()),
                    "lat_error_worst_deg": float(valid.lat_error_deg.max()),
                    "lon_error_mean_deg": float(valid.lon_error_deg.mean()),
                    "lon_error_std_deg": float(valid.lon_error_deg.std()),
                    "lon_error_worst_deg": float(valid.lon_error_deg.max()),
                    "km_error_mean": float(valid.km_error.mean()),
                    "km_error_std": float(valid.km_error.std()),
                    "km_error_worst": float(valid.km_error.max()),
                }
            )

            st.subheader("Failure gallery")
            worst = valid.sort_values("km_error", ascending=False).head(3)
            cols = st.columns(len(worst))
            for col, (_, r) in zip(cols, worst.iterrows()):
                with col:
                    st.markdown(f"**{r['session']}**")
                    st.write(f"km error: {r['km_error']:.3f}")
                    st.write(f"lat error: {r['lat_error_deg']:.4f} deg")
                    st.write(f"lon error: {r['lon_error_deg']:.4f} deg")
                    st.write(r["conditions"])
                    if r["session"] in thumbs:
                        st.image(thumbs[r["session"]], use_container_width=True)
        else:
            st.warning("No valid synthetic sessions completed.")
    else:
        st.info("Press the button to run the synthetic validation.")


# --- Capture protocol tab ----------------------------------------------------
with tabs[2]:
    st.header("Capture protocol")
    st.markdown(
        """
### Field checklist

1. Use a straight vertical stick.
2. Check plumb with a plumb line or phone level.
3. Measure stick height to 1 mm and record `stick_height_m`.
4. Mark a 1 m by 1 m square on level ground around the stick base.
5. Put the phone/camera on a tripod.
6. Record date, timezone offset, and whether daylight saving time is active.
7. Sync device clock to network time before capture.
8. Take one photo every 60 seconds from 90 minutes before to 90 minutes after expected solar noon.
9. Photograph only objects and people you have permission to photograph.
10. Avoid faces. Process locally unless you deliberately choose a public demo.

### Minimum input contract

- folder of photos or a video
- `stick_height_m`
- date
- timezone offset
- hemisphere: N or S
- four marked ground points forming a 1 m square

### Output contract

- `outputs/track.csv`
- `outputs/results.json`
- `outputs/plots.png`
- `outputs/debug/`
"""
    )


# --- Novelty audit tab -------------------------------------------------------
with tabs[3]:
    st.header("Novelty audit")
    st.markdown(
        """
### Assumption

Because this app cannot browse the live web inside this single file, the novelty audit below is a
starting audit based on the project specification and common open-source knowledge. Before publishing,
run these searches and paste results into `NOVELTY.md`:

- `latitude longitude from shadow video computer vision`
- `gnomon shadow geolocation python`
- `shadow trajectory geolocation camera calibration`

Search on: Google Scholar, arXiv, GitHub, PyPI, Hugging Face Spaces, YouTube, and Google Patents.

### Closest prior work categories

1. Classical gnomon latitude methods.
   - Do: estimate latitude from shadow length at solar noon.
   - Do not: provide full video tracking, uncertainty intervals, longitude from solar noon, and validation.

2. Shadow-based geolocation research.
   - Do: use shadow directions or trajectories for camera/geographic inference.
   - Often do not: provide a free, reproducible, beginner-runnable Streamlit measurement pipeline with
     Monte Carlo intervals and explicit failure reporting.

3. Open-source sundial/solar-position tools.
   - Do: compute solar position accurately.
   - Do not: close the loop from observed shadow track to lat/lon/north with uncertainty and GPS comparison.

### Differentiation statement

Existing work often estimates latitude or solar geometry from a gnomon shadow. It usually does not provide
an end-to-end free, reproducible, uncertainty-aware pipeline that estimates latitude, longitude, and true-north
orientation from a time-lapse shadow track under beginner constraints, validated against known truth.

SunTrace does that under the constraints of free open-source tools, local processing, fixed seeds, one config file,
unit tests, Monte Carlo intervals, and honest failure reporting.

### Patent note

This is a learning project. Check patents before commercializing. This is not legal advice.
"""
    )


# --- Known limits tab --------------------------------------------------------
with tabs[4]:
    st.header("Known limits")
    st.markdown(
        """
- This is not navigation-grade accuracy.
- The live demo is synthetic. Real field validation requires GPS ground truth collected by the user.
- Very short noon shadows, altitude above about 80 degrees, degrade latitude estimation.
- Heavy cloud can remove too many frames; the app warns when more than 30% are rejected.
- Stick tilt, uneven ground, incorrect timezone/DST, and unsynced clocks create systematic errors.
- The built-in marker homography is demonstrated synthetically. Real photo homography needs either
  manually clicked square corners or printed ArUco markers; this single-file app focuses on the measurable
  track pipeline and demo.
- Faces should be avoided. Process locally. Do not upload private imagery unless you choose a public demo.
- No medical, safety, or legal claims are made.
"""
    )


# --- Self-tests tab ----------------------------------------------------------
with tabs[5]:
    st.header("Self-tests")
    st.markdown(
        "These tests check the solar formulas, homography round-trip, robust solar-noon fit, "
        "and a synthetic round-trip recovery."
    )

    if st.button("Run self-tests"):
        st.session_state.selftest_df = run_self_tests(int(st.session_state.seed))

    if "selftest_df" in st.session_state:
        df = st.session_state.selftest_df
        st.dataframe(df, use_container_width=True)
        passed = int(df["passed"].sum())
        total = int(len(df))
        st.metric("Passed", f"{passed} / {total}")
        if passed != total:
            st.warning("Some tests failed. Inspect the table above.")
        else:
            st.success("All built-in tests passed.")
    else:
        st.info("Press the button to run the tests.")


# --- Upload tab --------------------------------------------------------------
with tabs[6]:
    st.header("Upload your own track.csv")
    st.markdown(
        """
Expected columns:

- `t_utc`
- `tip_x_m`
- `tip_y_m`
- `shadow_length_m`
- `azimuth_deg`
- `quality`

`tip_x_m` is east in meters relative to the stick base.  
`tip_y_m` is north in meters relative to the stick base.  
`quality` should be between 0 and 1.
"""
    )

    uploaded = st.file_uploader("Upload track.csv", type=["csv"])

    c1, c2, c3 = st.columns(3)
    with c1:
        upload_stick = st.number_input(
            "Uploaded stick height, m",
            min_value=0.05,
            value=1.0,
            step=0.01,
            key="upload_stick_widget",
        )
    with c2:
        upload_hem = st.radio(
            "Uploaded hemisphere",
            ["N", "S"],
            horizontal=True,
            key="upload_hem_widget",
        )
    with c3:
        upload_seed = st.number_input(
            "Uploaded seed",
            value=int(st.session_state.seed),
            step=1,
            key="upload_seed_widget",
        )

    if uploaded is not None:
        try:
            df_upload = load_uploaded_track(uploaded)
            st.write(f"Loaded {len(df_upload)} rows.")

            res_u, clean_u, noon_u, loc_u, mc_u = analyze_pipeline(
                raw_df=df_upload,
                stick_h=float(upload_stick),
                hemisphere=upload_hem,
                homography_sigma=float(st.session_state.homography_sigma),
                mc_draws=int(st.session_state.mc),
                seed=int(upload_seed),
            )

            if "error" in res_u:
                st.error(res_u["error"])
            else:
                st.success("Analysis complete.")
                st.json(res_u)

                fig_u = make_track_plot(clean_u, noon_u, upload_hem)
                if fig_u is not None:
                    st.pyplot(fig_u)
                    plt.close(fig_u)

                fig_mc = make_mc_plot(mc_u, res_u["latitude_deg"], res_u["longitude_deg"])
                if fig_mc is not None:
                    st.pyplot(fig_mc)
                    plt.close(fig_mc)

                st.download_button(
                    "Download uploaded results.json",
                    data=json.dumps(res_u, indent=2, default=str).encode("utf-8"),
                    file_name="results_uploaded.json",
                    mime="application/json",
                    key="download_upload_results",
                )
        except Exception as exc:
            st.error(f"Could not process uploaded file: {exc}")
    else:
        st.info("No file uploaded yet. The Live demo tab already contains a working automatic demo.")