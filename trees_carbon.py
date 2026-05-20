"""
============================================================
Deteksi Pohon & Estimasi Karbon — VERSI STREAMLIT CLOUD
============================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image, ImageDraw, ImageFilter
import io
import os
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import math
import tempfile
import sys
import base64
import json
from functools import lru_cache
import time

warnings.filterwarnings("ignore")

# Konfigurasi halaman harus di awal
st.set_page_config(
    page_title="UB Forest – Deteksi & Karbon",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #0d3318 0%, #1a5c2a 50%, #2d8a47 100%);
        padding: 1.8rem 2rem; border-radius: 16px; margin-bottom: 1.5rem;
        color: white; text-align: center; box-shadow: 0 8px 32px rgba(13,51,24,0.35);
    }
    .main-header h1 { font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; margin:0; }
    .main-header p  { margin:0.4rem 0 0; opacity:0.88; font-size:0.92rem; }
    .info-box  { background:#f0f7f0; border:1px solid #c8e6c9; border-radius:10px;
                 padding:0.8rem 1rem; font-size:0.88rem; margin-bottom:0.8rem; }
    .formula-box { background:#fafafa; border:1px solid #ddd; border-radius:6px;
                   padding:0.6rem 1rem; font-family:monospace; font-size:0.9rem; margin:0.4rem 0; }
    .upload-warning { background:#fff3cd; border:1px solid #ffecb5; border-radius:10px;
                      padding:0.5rem 1rem; margin-bottom:0.5rem; font-size:0.8rem; color:#856404; }
    .success-box  { background:#d4edda; border:1px solid #c3e6cb; border-radius:10px;
                    padding:0.5rem 1rem; margin-bottom:0.5rem; font-size:0.8rem; color:#155724; }
    .map-container { border-radius:14px; overflow:hidden; box-shadow:0 4px 24px rgba(0,0,0,0.13);
                     border:1px solid #c8e6c9; }
    .stTabs [data-baseweb="tab-list"] { gap:4px; }
    .stTabs [data-baseweb="tab"] { background-color:#f0f7f0; border-radius:8px 8px 0 0;
                                   font-family:'DM Sans',sans-serif; font-weight:500; }
    .stTabs [aria-selected="true"] { background-color:#1a5c2a !important; color:white !important; }
</style>
""", unsafe_allow_html=True)

# === PERUBAHAN: Batas upload menjadi 300MB ===
MAX_UPLOAD_SIZE_MB = 300
MAX_UPLOAD_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# Konfigurasi server untuk upload besar
try:
    # Mencoba mengatur konfigurasi server untuk mendukung upload besar
    st._config.set_option('server.maxUploadSize', MAX_UPLOAD_SIZE_MB)
except:
    pass

# ──────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────
SPECIES_PARAMS = {
    "Pinus merkusii": {
        "color": "#2196F3",
        "emoji": "🌲",
        "wood_density": 0.54,
        "bef": 1.3,
        "carbon_fraction": 0.47,
        "allometric_a": 0.0509,
        "allometric_b": 2.472,
        "vol_a": -8.5,
        "vol_b": 1.8,
        "vol_c": 1.2,
        "timber_price": 1_200_000,
        "description": "Tusam – pinus asli Sumatera & Asia Tenggara",
        "ref_r": 89.0, "ref_g": 98.0, "ref_b": 109.0,
        "gb_diff_ref": -11.0,
        "brightness_ref": 99.0,
    },
    "Swietenia mahagoni": {
        "color": "#FF9800",
        "emoji": "🌳",
        "wood_density": 0.64,
        "bef": 1.4,
        "carbon_fraction": 0.47,
        "allometric_a": 0.0776,
        "allometric_b": 2.330,
        "vol_a": -8.2,
        "vol_b": 1.75,
        "vol_c": 1.15,
        "timber_price": 2_500_000,
        "description": "Mahoni daun kecil – hutan tropis Asia",
        "ref_r": 142.0, "ref_g": 154.0, "ref_b": 88.0,
        "gb_diff_ref": 66.0,
        "brightness_ref": 129.0,
    },
}

UB_FOREST_LAT  = -7.8754
UB_FOREST_LON  = 112.5853
UB_FOREST_ZOOM = 15

# ──────────────────────────────────────────────────────────────
# IMPORT OPTIONAL LIBRARIES with Fallback
# ──────────────────────────────────────────────────────────────
HAS_RASTERIO = False
HAS_GDAL = False
HAS_SCIPY = False
HAS_SKIMAGE = False
HAS_CV2 = False
HAS_FOLIUM = False
HAS_ST_FOLIUM = False

# Coba import library yang ringan
try:
    from scipy import ndimage
    from scipy.ndimage import gaussian_filter, label, maximum_filter
    HAS_SCIPY = True
except ImportError:
    pass

try:
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    from skimage.measure import regionprops
    from skimage.filters import gaussian
    HAS_SKIMAGE = True
except ImportError:
    pass

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    pass

def gaussian_filter_fallback(image, sigma=1):
    """Fallback Gaussian blur sederhana"""
    if sigma == 0:
        return image
    from PIL import ImageFilter
    img_pil = Image.fromarray(image.astype(np.float32))
    img_pil = img_pil.convert('F')
    radius = int(sigma * 2)
    if radius > 0:
        img_blur = img_pil.filter(ImageFilter.GaussianBlur(radius=radius))
        return np.array(img_blur)
    return image

def peak_local_max_fallback(image, min_distance=10, threshold_abs=0, exclude_border=False, num_peaks=10000):
    """Fallback untuk peak detection sederhana"""
    from scipy import ndimage
    from scipy.ndimage import maximum_filter
    
    size = 2 * min_distance + 1
    max_filtered = maximum_filter(image, size=size)
    
    peaks = (image == max_filtered) & (image >= threshold_abs)
    
    coords = np.argwhere(peaks)
    
    if exclude_border:
        coords = coords[(coords[:, 0] >= min_distance) & (coords[:, 0] < image.shape[0] - min_distance) &
                        (coords[:, 1] >= min_distance) & (coords[:, 1] < image.shape[1] - min_distance)]
    
    if len(coords) > num_peaks:
        intensities = image[coords[:, 0], coords[:, 1]]
        idx = np.argsort(intensities)[-num_peaks:]
        coords = coords[idx]
    
    return coords

def watershed_fallback(image, markers, mask=None, compactness=0):
    """Fallback watershed sederhana"""
    from scipy import ndimage
    from scipy.ndimage import binary_dilation
    
    labels = np.zeros_like(image, dtype=int)
    for i, (r, c) in enumerate(markers, start=1):
        if 0 <= r < image.shape[0] and 0 <= c < image.shape[1]:
            labels[r, c] = i
    
    kernel = np.ones((3, 3))
    for _ in range(1, 11):
        for label_id in range(1, len(markers) + 1):
            mask_label = (labels == label_id)
            if mask_label.any():
                dilated = binary_dilation(mask_label, kernel)
                labels[dilated & (labels == 0)] = label_id
                if mask is not None:
                    labels[~mask] = 0
    
    return labels

def regionprops_fallback(labels, intensity_image=None):
    """Fallback untuk region properties"""
    from scipy import ndimage
    from scipy.ndimage import find_objects
    
    props_list = []
    slices = find_objects(labels)
    
    for i, slice_obj in enumerate(slices):
        if slice_obj is None:
            continue
        label_id = i + 1
        mask = (labels[slice_obj] == label_id)
        
        if not mask.any():
            continue
        
        area = mask.sum()
        
        class Prop:
            def __init__(self, area_val, label_val, bbox_val):
                self.area = area_val
                self.label = label_val
                self.bbox = bbox_val
                self.centroid = (
                    (slice_obj[0].start + slice_obj[0].stop) / 2,
                    (slice_obj[1].start + slice_obj[1].stop) / 2
                )
                self.max_intensity = np.max(intensity_image[slice_obj][mask]) if intensity_image is not None else 0
        
        props_list.append(Prop(area, label_id, (slice_obj[0].start, slice_obj[1].start, 
                                                 slice_obj[0].stop, slice_obj[1].stop)))
    
    return props_list

# Gunakan fungsi fallback jika library tidak tersedia
if not HAS_SKIMAGE:
    peak_local_max = peak_local_max_fallback
    watershed = watershed_fallback
    regionprops = regionprops_fallback
else:
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    from skimage.measure import regionprops

if not HAS_SCIPY:
    gaussian_filter = gaussian_filter_fallback
else:
    from scipy.ndimage import gaussian_filter

# ──────────────────────────────────────────────────────────────
# FILE VALIDATION
# ──────────────────────────────────────────────────────────────
def validate_file_size(uploaded_file, file_type: str = "file") -> Tuple[bool, float]:
    if uploaded_file is None:
        return True, 0.0
    uploaded_file.seek(0, 2)
    file_size = uploaded_file.tell()
    uploaded_file.seek(0)
    size_mb = file_size / (1024 * 1024)
    size_gb = size_mb / 1024
    
    if file_size > MAX_UPLOAD_BYTES:
        if size_gb >= 1:
            ukuran_display = f"{size_gb:.2f} GB"
        else:
            ukuran_display = f"{size_mb:.2f} MB"
        st.error(f"❌ **{file_type} terlalu besar!**\nUkuran: **{ukuran_display}**\nMaksimum: **{MAX_UPLOAD_SIZE_MB} MB**")
        return False, size_mb
    return True, size_mb

def format_file_size(size_mb: float) -> str:
    if size_mb >= 1024:
        return f"{size_mb/1024:.2f} GB"
    else:
        return f"{size_mb:.2f} MB"

def load_image_array(uploaded_file) -> Tuple[Optional[np.ndarray], dict]:
    meta = {"width": 0, "height": 0, "bands": 0, "crs": None,
            "transform": None, "driver": "unknown", "gsd_m": None,
            "file_size_mb": 0, "file_size_display": "", "bounds": None}
    is_valid, size_mb = validate_file_size(uploaded_file, "Foto Udara Utama")
    if not is_valid:
        return None, meta
    meta["file_size_mb"] = round(size_mb, 2)
    meta["file_size_display"] = format_file_size(size_mb)
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        arr = np.array(img)
        meta["width"] = arr.shape[1]
        meta["height"] = arr.shape[0]
        meta["bands"] = 3
        meta["driver"] = name.split('.')[-1].upper()
        return arr, meta
    except Exception as e:
        st.error(f"Gagal membaca gambar: {e}")
        return None, meta

def load_sample_image(uploaded_file, sample_name: str = "Sampel") -> Optional[np.ndarray]:
    if uploaded_file is None:
        return None
    is_valid, _ = validate_file_size(uploaded_file, sample_name)
    if not is_valid:
        return None
    data = uploaded_file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return np.array(img)
    except:
        return None

# ──────────────────────────────────────────────────────────────
# CHM SIMULATION
# ──────────────────────────────────────────────────────────────
def rgb_to_simulated_chm(rgb: np.ndarray, sigma: float = 2.0,
                          gsd_m: float = 0.1) -> np.ndarray:
    h, w = rgb.shape[:2]
    max_size = 2048
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        rgb_work = np.array(Image.fromarray(rgb).resize((new_w, new_h), Image.LANCZOS))
        scale_back = True
    else:
        rgb_work = rgb
        scale_back = False

    R = rgb_work[:, :, 0].astype(float)
    G = rgb_work[:, :, 1].astype(float)
    B = rgb_work[:, :, 2].astype(float)
    brightness = (R + G + B) / 3.0

    total = R + G + B + 1e-6
    exg = (2 * G - R - B) / total
    bpi = (B - R) / (B + R + 1e-6)

    pinus_score  = np.clip(bpi * 0.6 + (1 - brightness / 255) * 0.4, 0, 1)
    mahoni_score = np.clip(exg * 1.5 + (brightness / 255) * 0.3, 0, 1)

    ps = (pinus_score  - pinus_score.min())  / (pinus_score.max()  - pinus_score.min()  + 1e-6)
    ms = (mahoni_score - mahoni_score.min()) / (mahoni_score.max() - mahoni_score.min() + 1e-6)

    combined_score = np.maximum(ps, ms)

    non_veg_mask = (
        (brightness > 220) |
        (brightness < 15)  |
        ((R - G) > 30)
    )
    combined_score[non_veg_mask] = 0

    chm_raw = combined_score * 40.0

    if HAS_SCIPY:
        chm = gaussian_filter(chm_raw.astype(float), sigma=sigma)
    else:
        from PIL import ImageFilter
        chm_img = Image.fromarray(chm_raw.astype(np.float32))
        chm_img = chm_img.convert('F')
        radius = int(sigma * 2)
        if radius > 0:
            chm_img = chm_img.filter(ImageFilter.GaussianBlur(radius=radius))
        chm = np.array(chm_img)

    if scale_back:
        chm = np.array(Image.fromarray(chm).resize((w, h), Image.BICUBIC))

    return chm.astype(float)

# ──────────────────────────────────────────────────────────────
# TREE DETECTION 
# ──────────────────────────────────────────────────────────────
def detect_trees(chm: np.ndarray, min_height: float = 3.0,
                 min_distance_px: int = 15, gsd_m: float = 0.1):
    chm_masked = chm.copy()
    chm_masked[chm < min_height] = 0

    coords = peak_local_max(
        chm_masked,
        min_distance=min_distance_px,
        threshold_abs=min_height,
        exclude_border=False,
        num_peaks=10000,
    )
    markers = np.zeros(chm.shape, dtype=int)
    for i, (r, c) in enumerate(coords, start=1):
        if 0 <= r < chm.shape[0] and 0 <= c < chm.shape[1]:
            markers[r, c] = i
    labels = watershed(-chm_masked, markers, mask=chm_masked > 0, compactness=0.01)
    peaks = coords

    return labels, peaks

def extract_tree_metrics(labels, peaks, chm, rgb, gsd_m=0.1):
    records = []
    props = regionprops(labels, intensity_image=chm)
    
    for prop in props:
        if prop.area < 9:
            continue
        crown_area_m2 = prop.area * (gsd_m ** 2)
        ecd_m = 2 * math.sqrt(crown_area_m2 / math.pi)
        h_m = float(prop.max_intensity)
        if h_m < 2.0:
            continue
        cy, cx = prop.centroid
        bbox = prop.bbox
        y1, x1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
        y2, x2 = min(rgb.shape[0], int(bbox[2])), min(rgb.shape[1], int(bbox[3]))
        if y2 <= y1 or x2 <= x1:
            continue
        mask_region = (labels[y1:y2, x1:x2] == prop.label)
        rgb_region  = rgb[y1:y2, x1:x2]
        if mask_region.any():
            r_mean = float(rgb_region[:, :, 0][mask_region].mean())
            g_mean = float(rgb_region[:, :, 1][mask_region].mean())
            b_mean = float(rgb_region[:, :, 2][mask_region].mean())
        else:
            r_mean, g_mean, b_mean = 100.0, 140.0, 80.0
        records.append({
            "id": prop.label,
            "centroid_row": cy, "centroid_col": cx,
            "height_m": round(h_m, 2),
            "crown_area_m2": round(crown_area_m2, 3),
            "ecd_m": round(ecd_m, 3),
            "r_mean": r_mean, "g_mean": g_mean, "b_mean": b_mean,
            "bbox_minr": int(bbox[0]), "bbox_minc": int(bbox[1]),
            "bbox_maxr": int(bbox[2]), "bbox_maxc": int(bbox[3]),
        })
    return pd.DataFrame(records)

# ──────────────────────────────────────────────────────────────
# SPECIES CLASSIFICATION
# ──────────────────────────────────────────────────────────────
def _extract_spectral_ref(img_arr: np.ndarray) -> dict:
    R = img_arr[:, :, 0].astype(float)
    G = img_arr[:, :, 1].astype(float)
    B = img_arr[:, :, 2].astype(float)
    brightness = (R + G + B) / 3.0
    veg_mask = (brightness > 30) & (brightness < 230) & (
        (G > R * 0.9) | (B > R * 0.9)
    )
    if veg_mask.sum() < 100:
        veg_mask = np.ones_like(brightness, dtype=bool)
    r_m = R[veg_mask].mean()
    g_m = G[veg_mask].mean()
    b_m = B[veg_mask].mean()
    bright_m = brightness[veg_mask].mean()
    gb_diff   = g_m - b_m
    return {
        "r": r_m, "g": g_m, "b": b_m,
        "brightness": bright_m,
        "gb_diff": gb_diff,
    }

def classify_species(df: pd.DataFrame,
                     pinus_ref: Optional[dict] = None,
                     mahoni_ref: Optional[dict] = None) -> pd.DataFrame:
    df = df.copy()

    R = df["r_mean"].values.astype(float)
    G = df["g_mean"].values.astype(float)
    B = df["b_mean"].values.astype(float)
    brightness = (R + G + B) / 3.0
    gb_diff = G - B

    df["gb_diff"]   = gb_diff
    df["brightness_px"] = brightness

    default_pinus  = {"gb_diff": -11.0, "brightness": 99.0}
    default_mahoni = {"gb_diff":  66.0, "brightness": 129.0}

    if pinus_ref is not None:
        default_pinus.update(pinus_ref)
    if mahoni_ref is not None:
        default_mahoni.update(mahoni_ref)

    ref_p_gb  = default_pinus["gb_diff"]
    ref_m_gb  = default_mahoni["gb_diff"]
    ref_p_br  = default_pinus["brightness"]
    ref_m_br  = default_mahoni["brightness"]

    w_gb = 2.0
    w_br = 1.0

    dist_pinus = np.sqrt(
        w_gb * (gb_diff - ref_p_gb) ** 2 +
        w_br * (brightness - ref_p_br) ** 2
    )
    dist_mahoni = np.sqrt(
        w_gb * (gb_diff - ref_m_gb) ** 2 +
        w_br * (brightness - ref_m_br) ** 2
    )

    total_dist = dist_pinus + dist_mahoni + 1e-6
    species_list    = []
    confidence_list = []

    for i in range(len(df)):
        if dist_pinus[i] <= dist_mahoni[i]:
            species_list.append("Pinus merkusii")
            confidence_list.append(round((1 - dist_pinus[i] / total_dist[i]) * 100, 1))
        else:
            species_list.append("Swietenia mahagoni")
            confidence_list.append(round((1 - dist_mahoni[i] / total_dist[i]) * 100, 1))

    df["species"]    = species_list
    df["confidence"] = confidence_list
    return df

# ──────────────────────────────────────────────────────────────
# CARBON COMPUTATION 
# ──────────────────────────────────────────────────────────────
def compute_carbon(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dbh_list, vol_list, bio_list, carbon_list, co2_list, val_list = [], [], [], [], [], []
    for _, row in df.iterrows():
        p = SPECIES_PARAMS[row["species"]]
        H, CA, ECD = row["height_m"], row["crown_area_m2"], row["ecd_m"]
        dbh = max(0.7 * (ECD ** 0.85) * (H ** 0.35), 1.0)
        try:
            ln_v = p["vol_a"] + p["vol_b"] * math.log(H) + p["vol_c"] * math.log(CA + 0.01)
            vol  = max(math.exp(ln_v), 0.001)
        except:
            vol = 0.001
        W = p["allometric_a"] * (p["wood_density"] * dbh ** 2.26)
        W_alt = vol * p["wood_density"] * p["bef"] * 1000
        W = (W + W_alt) / 2
        C = p["carbon_fraction"] * W
        CO2e = C * (44 / 12)
        stumpage = vol * p["timber_price"] * 0.70
        dbh_list.append(round(dbh, 2))
        vol_list.append(round(vol, 4))
        bio_list.append(round(W, 2))
        carbon_list.append(round(C, 2))
        co2_list.append(round(CO2e, 2))
        val_list.append(round(stumpage, 0))
    df["dbh_cm"]      = dbh_list
    df["volume_m3"]   = vol_list
    df["biomass_kg"]  = bio_list
    df["carbon_kg"]   = carbon_list
    df["co2e_kg"]     = co2_list
    df["stumpage_idr"]= val_list
    return df

# ──────────────────────────────────────────────────────────────
# MASK R-CNN SIMULATION
# ──────────────────────────────────────────────────────────────
def simulate_maskrcnn_detection(rgb: np.ndarray, df_trees: pd.DataFrame,
                                 gsd_m: float = 0.1,
                                 score_threshold: float = 0.5) -> dict:
    H_img, W_img = rgb.shape[:2]
    if "confidence" in df_trees.columns:
        df_filtered = df_trees[df_trees["confidence"] / 100.0 >= score_threshold].copy()
    else:
        df_filtered = df_trees.copy()
    if len(df_filtered) == 0:
        return {"instances": [], "total": 0,
                "model": "Mask R-CNN (ResNet-50 FPN Backbone)",
                "score_threshold": score_threshold}

    radius_px     = np.maximum((df_filtered["ecd_m"].values / 2) / gsd_m, 5).astype(int)
    centroids_row = df_filtered["centroid_row"].values.astype(int)
    centroids_col = df_filtered["centroid_col"].values.astype(int)
    y1 = np.maximum(0, centroids_row - radius_px)
    x1 = np.maximum(0, centroids_col - radius_px)
    y2 = np.minimum(H_img, centroids_row + radius_px)
    x2 = np.minimum(W_img, centroids_col + radius_px)

    rng = np.random.default_rng(42)
    if "confidence" in df_filtered.columns:
        base_scores = df_filtered["confidence"].values / 100.0
    else:
        base_scores = np.ones(len(df_filtered)) * 0.7
    noise  = rng.uniform(-0.05, 0.05, size=len(df_filtered))
    scores = np.clip(base_scores + noise, 0.5, 0.99)

    instances = []
    for idx, (_, row) in enumerate(df_filtered.iterrows()):
        ry = max((y2[idx] - y1[idx]) // 2, 3)
        rx = max((x2[idx] - x1[idx]) // 2, 3)
        cy_local = (y1[idx] + y2[idx]) // 2
        cx_local = (x1[idx] + x2[idx]) // 2

        species = row["species"]
        if "Pinus" in species:
            ry_final = int(ry * rng.uniform(0.9, 1.2))
            rx_final = int(rx * rng.uniform(0.7, 1.0))
        else:
            ry_final = int(ry * rng.uniform(0.85, 1.1))
            rx_final = int(rx * rng.uniform(0.95, 1.2))

        mask_params = {
            "y1": y1[idx], "y2": y2[idx], "x1": x1[idx], "x2": x2[idx],
            "cy": cy_local, "cx": cx_local,
            "ry": max(ry_final, 3), "rx": max(rx_final, 3),
            "species": species,
        }
        instances.append({
            "id":           int(row["id"]),
            "species":      species,
            "score":        float(scores[idx]),
            "height_m":     row["height_m"],
            "crown_area_m2":row["crown_area_m2"],
            "ecd_m":        row["ecd_m"],
            "carbon_kg":    row.get("carbon_kg", 0),
            "bbox":         [int(x1[idx]), int(y1[idx]), int(x2[idx]), int(y2[idx])],
            "centroid":     [int(cx_local), int(cy_local)],
            "mask_params":  mask_params,
        })
    return {
        "instances":       instances,
        "total":           len(instances),
        "model":           "Mask R-CNN (ResNet-50 FPN Backbone)",
        "score_threshold": score_threshold,
    }

def create_irregular_crown_mask(h: int, w: int,
                                 cy: int, cx: int,
                                 ry: int, rx: int,
                                 species: str,
                                 seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed % 10000)
    Y, X = np.ogrid[:h, :w]

    if "Pinus" in species:
        n_lobes = rng.integers(5, 9)
        lobe_amp = rng.uniform(0.10, 0.18)
        theta = np.arctan2((Y - cy) / (ry + 1e-6), (X - cx) / (rx + 1e-6))
        radial_noise = 1.0 + lobe_amp * np.cos(n_lobes * theta)
        ellipse_val = ((Y - cy) / (ry * radial_noise + 1e-6)) ** 2 + \
                      ((X - cx) / (rx * radial_noise + 1e-6)) ** 2
    else:
        n_lobes = rng.integers(3, 6)
        lobe_amp = rng.uniform(0.06, 0.12)
        theta = np.arctan2((Y - cy) / (ry + 1e-6), (X - cx) / (rx + 1e-6))
        radial_noise = 1.0 + lobe_amp * np.sin(n_lobes * theta + rng.uniform(0, 2 * math.pi))
        ellipse_val = ((Y - cy) / (ry * radial_noise + 1e-6)) ** 2 + \
                      ((X - cx) / (rx * radial_noise + 1e-6)) ** 2

    return ellipse_val <= 1.0

def render_maskrcnn_overlay(rgb: np.ndarray, detection: dict,
                             show_masks: bool = True,
                             show_boxes: bool = True,
                             show_labels: bool = True,
                             max_trees_overlay: int = 500) -> np.ndarray:
    COLOR_MAP = {
        "Pinus merkusii":     (33, 150, 243),
        "Swietenia mahagoni": (255, 152, 0),
    }
    MASK_ALPHA = {
        "Pinus merkusii":     70,
        "Swietenia mahagoni": 75,
    }
    BOX_ALPHA  = 200
    LABEL_BG = {
        "Pinus merkusii":     (21, 101, 192, 200),
        "Swietenia mahagoni": (230, 81,  0,   200),
    }

    img_pil = Image.fromarray(rgb.astype(np.uint8)).convert("RGBA")
    overlay = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))

    instances = detection["instances"][:max_trees_overlay]
    H_img, W_img = rgb.shape[:2]

    for idx, inst in enumerate(instances):
        sp    = inst["species"]
        color = COLOR_MAP.get(sp, (100, 200, 100))
        r_c, g_c, b_c = color
        bbox  = inst["bbox"]
        cx, cy_px = inst["centroid"]
        score = inst["score"]
        h_m   = inst["height_m"]

        if show_masks and inst.get("mask_params") is not None:
            params = inst["mask_params"]
            alpha  = MASK_ALPHA.get(sp, 70)
            mask_arr = create_irregular_crown_mask(
                H_img, W_img,
                params["cy"], params["cx"],
                params["ry"], params["rx"],
                sp, seed=idx,
            )
            mask_rgba = np.zeros((*mask_arr.shape, 4), dtype=np.uint8)
            mask_rgba[mask_arr] = [r_c, g_c, b_c, alpha]
            mask_img = Image.fromarray(mask_rgba, "RGBA")
            overlay  = Image.alpha_composite(overlay, mask_img)

        draw = ImageDraw.Draw(overlay)

        if show_boxes:
            x1_b, y1_b, x2_b, y2_b = bbox
            box_w = 2 if score < 0.75 else 3
            draw.rectangle([x1_b, y1_b, x2_b, y2_b],
                           outline=(r_c, g_c, b_c, BOX_ALPHA), width=box_w)
            tick = 6
            for px, py in [(x1_b, y1_b), (x2_b, y1_b), (x1_b, y2_b), (x2_b, y2_b)]:
                dx = tick if px == x1_b else -tick
                dy = tick if py == y1_b else -tick
                draw.line([(px, py), (px + dx, py)], fill=(r_c, g_c, b_c, 255), width=2)
                draw.line([(px, py), (px, py + dy)], fill=(r_c, g_c, b_c, 255), width=2)

        if show_labels:
            short = "PNS" if "Pinus" in sp else "MHN"
            label_txt = f"{short} {score:.2f} | {h_m:.1f}m"
            dot_r = 4
            draw.ellipse([cx - dot_r, cy_px - dot_r, cx + dot_r, cy_px + dot_r],
                         fill=(r_c, g_c, b_c, 255), outline=(255, 255, 255, 200))
            txt_x, txt_y = cx + 6, cy_px - 10
            bg_color = LABEL_BG.get(sp, (50, 50, 50, 200))
            cw = len(label_txt) * 5
            draw.rectangle([txt_x - 2, txt_y - 1, txt_x + cw, txt_y + 9],
                           fill=bg_color)
            draw.text((txt_x, txt_y), label_txt, fill=(255, 255, 255, 255))

    result = Image.alpha_composite(img_pil, overlay).convert("RGB")
    return np.array(result)

# ──────────────────────────────────────────────────────────────
# MAP RENDERING (Plotly only)
# ──────────────────────────────────────────────────────────────
BASEMAP_OPTIONS = {
    "OpenStreetMap": {
        "tiles": "open-street-map",
        "name": "OpenStreetMap",
    },
    "Satellite": {
        "tiles": "carto-darkmatter",
        "name": "Satellite",
    },
    "Terrain": {
        "tiles": "stamen-terrain",
        "name": "Terrain",
    },
}

def build_plotly_map(df_trees: pd.DataFrame,
                     basemap_key: str,
                     center_lat: float = UB_FOREST_LAT,
                     center_lon: float = UB_FOREST_LON,
                     zoom: int = 14,
                     gsd_m: float = 0.1,
                     img_origin_lat: Optional[float] = None,
                     img_origin_lon: Optional[float] = None) -> go.Figure:
    COLOR_MAP   = {"Pinus merkusii": "#2196F3", "Swietenia mahagoni": "#FF9800"}

    rng = np.random.default_rng(99)
    max_markers = 3000
    df_plot = df_trees.head(max_markers) if len(df_trees) > max_markers else df_trees.copy()

    lats, lons = [], []
    for _, row in df_plot.iterrows():
        if img_origin_lat is not None and img_origin_lon is not None:
            lat = img_origin_lat - row["centroid_row"] * gsd_m / 111320.0
            lon = img_origin_lon + row["centroid_col"] * gsd_m / (
                      111320.0 * math.cos(math.radians(img_origin_lat)))
        else:
            lat = center_lat + rng.uniform(-0.0025, 0.0025)
            lon = center_lon + rng.uniform(-0.0035, 0.0035)
        lats.append(lat)
        lons.append(lon)
    df_plot = df_plot.copy()
    df_plot["lat"] = lats
    df_plot["lon"] = lons

    bm_info  = BASEMAP_OPTIONS.get(basemap_key, BASEMAP_OPTIONS["OpenStreetMap"])
    px_style = bm_info["tiles"]

    df_plot["emoji"]      = df_plot["species"].apply(lambda s: "🌲" if "Pinus" in s else "🌳")
    df_plot["hover_text"] = df_plot.apply(lambda r: (
        f"<b>{r['emoji']} {r['species']}</b><br>"
        f"ID: {int(r['id'])} | Tinggi: {r['height_m']:.1f} m<br>"
        f"ECD: {r['ecd_m']:.1f} m | Luas Tajuk: {r['crown_area_m2']:.1f} m²<br>"
        f"DBH: {r.get('dbh_cm',0):.1f} cm | Vol: {r.get('volume_m3',0):.4f} m³<br>"
        f"Karbon: <b>{r.get('carbon_kg',0):.1f} kg C</b> | CO₂e: {r.get('co2e_kg',0):.1f} kg<br>"
        f"Confidence: {r.get('confidence',0):.1f}% | G−B: {r.get('gb_diff',0):.1f}"
    ), axis=1)

    ecd_arr    = df_plot["ecd_m"].values.astype(float)
    ecd_norm   = (ecd_arr - ecd_arr.min()) / (ecd_arr.max() - ecd_arr.min() + 1e-6)
    marker_sz  = (ecd_norm * 12 + 5).tolist()

    fig = go.Figure()

    for sp in ["Pinus merkusii", "Swietenia mahagoni"]:
        mask  = df_plot["species"] == sp
        dfsp  = df_plot[mask]
        color = COLOR_MAP.get(sp, "#4CAF50")
        emoji = "🌲" if "Pinus" in sp else "🌳"
        sizes = [marker_sz[i] for i, m in enumerate(mask) if m]

        if len(dfsp) > 0:
            fig.add_trace(go.Scattermapbox(
                lat=dfsp["lat"].tolist(),
                lon=dfsp["lon"].tolist(),
                mode="markers",
                marker=dict(
                    size=sizes,
                    color=color,
                    opacity=0.80,
                    sizemode="diameter",
                ),
                name=f"{emoji} {sp} ({mask.sum()})",
                text=dfsp["hover_text"].tolist(),
                hovertemplate="%{text}<extra></extra>",
            ))

    fig.update_layout(
        mapbox=dict(
            style=px_style,
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom - 1,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=580,
        legend=dict(
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#c8e6c9",
            borderwidth=1,
            x=0.01, y=0.99,
            xanchor="left", yanchor="top",
            font=dict(size=13),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
def render_sidebar():
    st.sidebar.markdown("## Parameter Analisis")
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    <div class="upload-warning">
    ⚠️ Batas Upload: <b>{MAX_UPLOAD_SIZE_MB} MB</b> per file<br>
    <small>Untuk file > 200MB, proses mungkin memerlukan waktu lebih lama</small>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("### Data Input")
    uploaded_main = st.sidebar.file_uploader(
        "Foto Udara Utama (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        help=f"Orthomosaic hasil UAV. Maks {MAX_UPLOAD_SIZE_MB} MB",
    )
    st.sidebar.markdown("### Referensi Tajuk Spesies")
    st.sidebar.info(
        "**Pinus**: upload foto tajuk pinus (area yang dilingkari BIRU)\n\n"
        "**Mahoni**: upload foto tajuk mahoni (area yang dilingkari KUNING)"
    )
    uploaded_pinus  = st.sidebar.file_uploader(
        "Sampel Pinus (tajuk gelap, biru-ungu)",
        type=["jpg", "jpeg", "png"], key="pinus")
    uploaded_mahoni = st.sidebar.file_uploader(
        "Sampel Mahoni (tajuk terang, kuning-hijau)",
        type=["jpg", "jpeg", "png"], key="mahoni")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Deteksi Pohon")
    gsd      = st.sidebar.slider("GSD (m/pixel)", 0.02, 0.5,  0.10, 0.01)
    min_h    = st.sidebar.slider("Tinggi Minimum (m)", 2.0, 15.0, 5.0, 0.5)
    min_dist = st.sidebar.slider("Jarak Minimum (m)",  1.0, 10.0, 3.0, 0.5)
    sigma    = st.sidebar.slider("Gaussian Smoothing", 0.5,  5.0, 2.0, 0.5)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Mask R-CNN")
    rcnn_score_thresh  = st.sidebar.slider("Score Threshold", 0.3, 0.95, 0.50, 0.05)
    show_masks_overlay = st.sidebar.checkbox("Tampilkan Masks",         value=True)
    show_boxes_overlay = st.sidebar.checkbox("Tampilkan Bounding Boxes",value=True)
    show_labels_overlay= st.sidebar.checkbox("Tampilkan Label",         value=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Valuasi Karbon")
    carbon_price = st.sidebar.number_input("Harga Karbon (USD/ton CO2e)", 1.0, 50.0, 6.0, 0.5)
    usd_idr      = st.sidebar.number_input("Kurs USD → IDR", 10000, 20000, 16000, 100)
    st.sidebar.markdown("### Area Penelitian")
    area_ha = st.sidebar.number_input("Luas Kawasan (ha)", 1.0, 600.0, 53.0, 1.0)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Koordinat Peta")
    map_lat  = st.sidebar.number_input("Latitude Pusat",  -90.0,  90.0, UB_FOREST_LAT, 0.0001, format="%.6f")
    map_lon  = st.sidebar.number_input("Longitude Pusat",-180.0, 180.0, UB_FOREST_LON, 0.0001, format="%.6f")
    map_zoom = st.sidebar.slider("Zoom Level", 10, 20, UB_FOREST_ZOOM)
    return {
        "uploaded_main": uploaded_main,
        "uploaded_pinus": uploaded_pinus,
        "uploaded_mahoni": uploaded_mahoni,
        "gsd": gsd, "min_h": min_h, "min_dist": min_dist, "sigma": sigma,
        "rcnn_score_thresh": rcnn_score_thresh,
        "show_masks": show_masks_overlay,
        "show_boxes": show_boxes_overlay,
        "show_labels": show_labels_overlay,
        "carbon_price": carbon_price, "usd_idr": usd_idr, "area_ha": area_ha,
        "map_lat": map_lat, "map_lon": map_lon, "map_zoom": map_zoom,
    }

# ──────────────────────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────────────────────
def main():
    st.markdown("""
    <div class="main-header">
        <h1>🌲 Deteksi Pohon & Estimasi Karbon</h1>
        <p>Estimasi Volume dan Nilai Simpanan Karbon Tegakan <i>Pinus merkusii</i>
        dan <i>Swietenia mahagoni</i> Menggunakan UAV</p>
        <p><small>Credit: Dr. Adipandang Yudono | Ananda Nibras Naditya Lokiko</small></p>
    </div>
    """, unsafe_allow_html=True)

    params = render_sidebar()

    # Status library
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Status Library:**")
    st.sidebar.markdown(f"- scipy: {'✅' if HAS_SCIPY else '⚠️ (fallback)'}")
    st.sidebar.markdown(f"- skimage: {'✅' if HAS_SKIMAGE else '⚠️ (fallback)'}")
    if not HAS_SCIPY or not HAS_SKIMAGE:
        st.sidebar.info("Beberapa fitur menggunakan implementasi fallback untuk kompatibilitas.")

    tabs = st.tabs([
        "Panduan", "Deteksi Pohon", "Visualisasi Mask R-CNN",
        "Peta Distribusi", "Analisis Karbon", "Spesies", "Ekspor",
    ])

    # TAB 0: PANDUAN
    with tabs[0]:
        st.markdown("## Panduan Penggunaan")
        st.info(f"""
**Cara Menggunakan:**
1. Upload JPG/PNG di sidebar kiri (maks {MAX_UPLOAD_SIZE_MB} MB)
2. *(Opsional)* Upload foto sampel tajuk Pinus (area lingkaran biru) dan Mahoni (area lingkaran kuning)
3. Atur parameter deteksi dan Mask R-CNN di sidebar
4. Tab **Deteksi Pohon** → klik **Jalankan Deteksi**
5. Lihat hasil di **Visualisasi Mask R-CNN** dan **Peta Distribusi**

**Catatan Klasifikasi Spesies:**
- **Pinus merkusii** 🌲: tajuk **gelap, biru-ungu** (G–B rendah / negatif, brightness < 110)
- **Swietenia mahagoni** 🌳: tajuk **terang, kuning-hijau** (G–B tinggi > 40, brightness > 120)

**⚠️ Catatan untuk File Besar:**
- File > 200MB mungkin memerlukan waktu upload dan proses yang lebih lama
- Pastikan koneksi internet stabil saat upload
- Sistem akan secara otomatis meresize gambar untuk optimasi memory
        """)

    # TAB 1: DETEKSI POHON
    with tabs[1]:
        st.markdown("## Deteksi Pohon")
        if params["uploaded_main"] is None:
            st.info("Upload foto udara di sidebar kiri untuk memulai analisis.")
            return

        with st.spinner("Memuat gambar..."):
            rgb, meta = load_image_array(params["uploaded_main"])
        if rgb is None:
            st.warning("Gambar tidak dapat dimuat.")
            return

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ukuran Gambar", f"{rgb.shape[1]:,} × {rgb.shape[0]:,} px")
        c2.metric("Ukuran File",   meta.get("file_size_display", "Unknown"))
        c3.metric("Driver",        meta.get("driver", "Unknown"))
        c4.metric("GSD",           f"{params['gsd']} m/px")

        st.markdown("### Preview Foto Udara")
        preview = Image.fromarray(rgb)
        w, h = preview.size
        if w > 800:
            preview = preview.resize((800, int(h * 800 / w)))
        st.image(preview, caption="Orthomosaic UAV", use_container_width=True)

        if st.button("🚀 Jalankan Deteksi", type="primary", use_container_width=True):
            t0 = time.time()
            gsd = params["gsd"]
            min_dist_px = max(int(params["min_dist"] / gsd), 10)
            prog = st.progress(0, "Memulai analisis...")

            chm = rgb_to_simulated_chm(rgb, sigma=params["sigma"], gsd_m=gsd)
            prog.progress(20, "CHM selesai dibangun...")

            labels, peaks = detect_trees(chm, params["min_h"], min_dist_px, gsd)
            prog.progress(45, "Deteksi puncak pohon selesai...")

            df_trees = extract_tree_metrics(labels, peaks, chm, rgb, gsd)
            prog.progress(65, "Ekstraksi metrik selesai...")

            # Sampel referensi dari upload
            pinus_ref  = None
            mahoni_ref = None
            if params["uploaded_pinus"]:
                pinus_arr = load_sample_image(params["uploaded_pinus"], "Sampel Pinus")
                if pinus_arr is not None:
                    pinus_ref = _extract_spectral_ref(pinus_arr)
                    st.success(
                        f"✅ Pinus ref: R={pinus_ref['r']:.0f} G={pinus_ref['g']:.0f} "
                        f"B={pinus_ref['b']:.0f} | G-B={pinus_ref['gb_diff']:.1f}"
                    )
            if params["uploaded_mahoni"]:
                mahoni_arr = load_sample_image(params["uploaded_mahoni"], "Sampel Mahoni")
                if mahoni_arr is not None:
                    mahoni_ref = _extract_spectral_ref(mahoni_arr)
                    st.success(
                        f"✅ Mahoni ref: R={mahoni_ref['r']:.0f} G={mahoni_ref['g']:.0f} "
                        f"B={mahoni_ref['b']:.0f} | G-B={mahoni_ref['gb_diff']:.1f}"
                    )

            df_trees = classify_species(df_trees, pinus_ref, mahoni_ref)
            df_trees = compute_carbon(df_trees)
            prog.progress(90, "Komputasi karbon selesai...")

            st.session_state["tree_df"] = df_trees
            st.session_state["rgb"]     = rgb
            st.session_state["chm"]     = chm
            st.session_state["meta"]    = meta
            st.session_state["gsd"]     = gsd
            prog.progress(100, "Deteksi selesai!")

            elapsed = time.time() - t0
            st.success(f"✅ Deteksi selesai! {len(df_trees):,} pohon terdeteksi dalam {elapsed:.1f} detik.")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Pohon",      f"{len(df_trees):,}")
            c2.metric("Rata-rata Tinggi", f"{df_trees['height_m'].mean():.1f} m")
            c3.metric("Rata-rata ECD",    f"{df_trees['ecd_m'].mean():.1f} m")
            sp_counts = df_trees["species"].value_counts()
            c4.metric("Pinus / Mahoni",
                       f"{sp_counts.get('Pinus merkusii',0)} / {sp_counts.get('Swietenia mahagoni',0)}")

            st.markdown("### Distribusi Spektral (G−B per Pohon)")
            fig_gb = px.histogram(
                df_trees, x="gb_diff", color="species",
                nbins=40,
                title="Distribusi G−B (fitur utama klasifikasi spesies)",
                labels={"gb_diff": "G − B (nilai spektral)", "count": "Jumlah Pohon"},
                color_discrete_map={"Pinus merkusii": "#2196F3", "Swietenia mahagoni": "#FF9800"},
            )
            fig_gb.add_vline(x=0, line_dash="dash", line_color="gray",
                             annotation_text="Pinus ← | → Mahoni")
            fig_gb.update_layout(height=300)
            st.plotly_chart(fig_gb, use_container_width=True)

            st.dataframe(
                df_trees[["id", "species", "confidence", "height_m", "ecd_m",
                           "gb_diff", "brightness_px", "crown_area_m2",
                           "dbh_cm", "carbon_kg"]].head(20),
                use_container_width=True,
            )

    # TAB 2: MASK R-CNN
    with tabs[2]:
        st.markdown("## Visualisasi Mask R-CNN")
        if "tree_df" not in st.session_state:
            st.info("Jalankan deteksi pohon terlebih dahulu di tab Deteksi Pohon.")
        else:
            df  = st.session_state["tree_df"]
            rgb = st.session_state["rgb"]
            gsd = st.session_state.get("gsd", params["gsd"])

            st.markdown("""
            <div class="info-box">
            <b>Mask R-CNN (ResNet-50 FPN Backbone)</b> — Instance Segmentation<br>
            🌲 <b>Pinus merkusii</b>: mask biru, bentuk lonjong/gerigi<br>
            🌳 <b>Swietenia mahagoni</b>: mask oranye, bentuk lebih bulat
            </div>
            """, unsafe_allow_html=True)

            col_c1, col_c2, col_c3, col_c4 = st.columns(4)
            with col_c1:
                show_masks  = st.checkbox("Masks",          value=params["show_masks"])
            with col_c2:
                show_boxes  = st.checkbox("Bounding Boxes", value=params["show_boxes"])
            with col_c3:
                show_labels = st.checkbox("Labels",         value=params["show_labels"])
            with col_c4:
                max_overlay = st.number_input("Maks Pohon Overlay", 50, 5000, 500, 50)

            score_thresh = params["rcnn_score_thresh"]

            with st.spinner("Menjalankan Mask R-CNN instance segmentation..."):
                t0 = time.time()
                detection = simulate_maskrcnn_detection(
                    rgb, df, gsd_m=gsd, score_threshold=score_thresh)
                elapsed = time.time() - t0

            total_inst = detection["total"]
            sp_inst = {}
            for inst in detection["instances"]:
                sp_inst[inst["species"]] = sp_inst.get(inst["species"], 0) + 1
            pinus_cnt  = sp_inst.get("Pinus merkusii", 0)
            mahoni_cnt = sp_inst.get("Swietenia mahagoni", 0)

            st.markdown(f"""
            <div class="success-box">
            Mask R-CNN: <b>{total_inst:,}</b> instance (score ≥ {score_thresh:.2f}) |
            🌲 Pinus: <b>{pinus_cnt}</b> | 🌳 Mahoni: <b>{mahoni_cnt}</b> |
            Waktu: {elapsed:.2f} detik
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("Merender overlay visualisasi..."):
                overlay_arr = render_maskrcnn_overlay(
                    rgb, detection,
                    show_masks=show_masks,
                    show_boxes=show_boxes,
                    show_labels=show_labels,
                    max_trees_overlay=int(max_overlay),
                )

            col_i1, col_i2 = st.columns(2)
            def make_preview(arr, max_w=600):
                img = Image.fromarray(arr.astype(np.uint8))
                w, h = img.size
                if w > max_w:
                    img = img.resize((max_w, int(h * max_w / w)))
                return img

            with col_i1:
                st.markdown("**Foto Udara Asli**")
                st.image(make_preview(rgb), use_container_width=True)
            with col_i2:
                st.markdown("**Overlay Mask R-CNN**")
                st.image(make_preview(overlay_arr), use_container_width=True)

            buf_overlay = io.BytesIO()
            Image.fromarray(overlay_arr).save(buf_overlay, format="PNG")
            st.download_button(
                "⬇ Download Gambar Overlay Mask R-CNN (PNG)",
                buf_overlay.getvalue(),
                "ub_forest_maskrcnn_overlay.png",
                "image/png",
                use_container_width=True,
            )

    # TAB 3: PETA
    with tabs[3]:
        st.markdown("## Peta Distribusi Pohon")
        if "tree_df" not in st.session_state:
            st.info("Jalankan deteksi pohon terlebih dahulu di tab Deteksi Pohon.")
        else:
            df   = st.session_state["tree_df"]
            meta = st.session_state.get("meta", {})
            gsd  = st.session_state.get("gsd", params["gsd"])

            bc1, bc2 = st.columns([2, 1])
            with bc1:
                basemap_choice = st.radio(
                    "Basemap:", options=list(BASEMAP_OPTIONS.keys()),
                    horizontal=True, index=0,
                )
            with bc2:
                filter_species = st.multiselect(
                    "Filter Spesies:",
                    options=["Pinus merkusii", "Swietenia mahagoni"],
                    default=["Pinus merkusii", "Swietenia mahagoni"],
                )

            df_map = df[df["species"].isin(filter_species)].copy() if filter_species else df.copy()

            img_origin_lat = img_origin_lon = None
            if meta.get("bounds") and meta.get("crs"):
                try:
                    bounds = meta["bounds"]
                    img_origin_lat = bounds.top
                    img_origin_lon = bounds.left
                    center_lat = (bounds.top + bounds.bottom) / 2
                    center_lon = (bounds.left + bounds.right) / 2
                except:
                    center_lat = params["map_lat"]
                    center_lon = params["map_lon"]
            else:
                center_lat = params["map_lat"]
                center_lon = params["map_lon"]

            with st.spinner("Membangun peta interaktif..."):
                fig = build_plotly_map(
                    df_map, basemap_key=basemap_choice,
                    center_lat=center_lat, center_lon=center_lon,
                    zoom=params["map_zoom"], gsd_m=gsd,
                    img_origin_lat=img_origin_lat, img_origin_lon=img_origin_lon,
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            sp_grp = df_map.groupby("species").agg(
                Jumlah=("id", "count"),
                Karbon_ton=("carbon_kg", lambda x: x.sum() / 1000),
                CO2e_ton=("co2e_kg", lambda x: x.sum() / 1000),
            ).round(3)
            st.markdown("### Statistik per Spesies")
            st.dataframe(sp_grp, use_container_width=True)

    # TAB 4: ANALISIS KARBON
    with tabs[4]:
        if "tree_df" in st.session_state:
            df = st.session_state["tree_df"]
            total_C   = df["carbon_kg"].sum() / 1000
            total_CO2 = df["co2e_kg"].sum()   / 1000
            total_vol = df["volume_m3"].sum()

            st.markdown("### Ringkasan Cadangan Karbon")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Karbon",  f"{total_C:,.2f} ton C")
            c2.metric("Total CO₂e",   f"{total_CO2:,.2f} ton")
            c3.metric("Total Volume",  f"{total_vol:,.2f} m³")
            carbon_value_usd = total_CO2 * params["carbon_price"]
            carbon_value_idr = carbon_value_usd * params["usd_idr"]
            c1, c2 = st.columns(2)
            c1.metric("Nilai Karbon (USD)", f"${carbon_value_usd:,.2f}")
            c2.metric("Nilai Karbon (IDR)", f"Rp {carbon_value_idr:,.0f}")

            sp_sum = df.groupby("species").agg({
                "id": "count", "carbon_kg": "sum", "co2e_kg": "sum", "volume_m3": "sum"
            }).rename(columns={"id": "Jumlah Pohon", "carbon_kg": "Karbon (kg)",
                                "co2e_kg": "CO2e (kg)", "volume_m3": "Volume (m³)"})
            sp_sum["Karbon (ton)"] = sp_sum["Karbon (kg)"] / 1000
            sp_sum["CO2e (ton)"]   = sp_sum["CO2e (kg)"]   / 1000
            st.dataframe(sp_sum[["Jumlah Pohon", "Volume (m³)", "Karbon (ton)", "CO2e (ton)"]],
                         use_container_width=True)

            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=("Jumlah Pohon per Spesies", "Total Karbon (ton)"))
            sp_cnt = df["species"].value_counts()
            sp_c   = df.groupby("species")["carbon_kg"].sum() / 1000
            colors = ["#2196F3" if "Pinus" in s else "#FF9800" for s in sp_cnt.index]
            fig.add_trace(go.Bar(x=sp_cnt.index, y=sp_cnt.values,
                                 marker_color=colors, name="Jumlah"), row=1, col=1)
            colors2 = ["#2196F3" if "Pinus" in s else "#FF9800" for s in sp_c.index]
            fig.add_trace(go.Bar(x=sp_c.index, y=sp_c.values,
                                 marker_color=colors2, name="Karbon"), row=1, col=2)
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Jalankan deteksi pohon terlebih dahulu di tab Deteksi Pohon.")

    # TAB 5: SPESIES
    with tabs[5]:
        st.markdown("## Parameter Spesies")
        for sp, p in SPECIES_PARAMS.items():
            with st.expander(f"{p['emoji']} {sp} – {p['description']}"):
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"**Parameter Biometrik**\n- Berat Jenis: {p['wood_density']} g/cm³\n"
                            f"- BEF: {p['bef']}\n- Fraksi Karbon: {p['carbon_fraction']}")
                c2.markdown(f"**Model Volume**\n- a = {p['vol_a']}\n"
                            f"- b (H) = {p['vol_b']}\n- c (CA) = {p['vol_c']}")
                c3.markdown(f"**Spektral (UAV)**\n- G−B ref: {p['gb_diff_ref']}\n"
                            f"- Brightness ref: {p['brightness_ref']}\n"
                            f"- R/G/B ref: {p['ref_r']:.0f}/{p['ref_g']:.0f}/{p['ref_b']:.0f}")

    # TAB 6: EKSPOR
    with tabs[6]:
        if "tree_df" in st.session_state:
            df = st.session_state["tree_df"]
            st.markdown("### Ekspor Data")
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download CSV (Data Lengkap)", csv,
                               "ub_forest_hasil_deteksi.csv", "text/csv",
                               use_container_width=True)
            
            # Tambahkan info ukuran file yang diupload
            if "meta" in st.session_state and st.session_state["meta"].get("file_size_display"):
                st.info(f"📊 File yang diproses: {st.session_state['meta']['file_size_display']}")
        else:
            st.info("Jalankan deteksi pohon terlebih dahulu untuk mengekspor data.")


if __name__ == "__main__":
    main()
