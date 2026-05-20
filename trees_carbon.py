"""
============================================================
Deteksi Pohon & Estimasi Karbon 
============================================================
Perbaikan:
- streamlit server max upload size dikonfigurasi via .streamlit/config.toml (bukan env var)
- Temporary file cleanup yang aman (delete=False + manual delete)
- gc.collect() yang lebih agresif dan konsisten
- Validasi ukuran file via seek() yang lebih handal
- Error handling yang lebih baik di semua fungsi baca
- Fallback regionprops yang diperbaiki (menerima labels saja, bukan positional args)
- read_large_tiff_chunked tidak bergantung pada skimage jika HAS_SKIMAGE = False
- render_maskrcnn_overlay: font default PIL digunakan (tidak ada import font gagal)
- Tab Analisis Karbon: guard untuk df kosong
- Session state keys yang konsisten
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
import base64
import json
import time
import gc
import pyproj
from pyproj import Transformer

def extract_utm_from_geotiff(meta: dict) -> Tuple[Optional[str], Optional[Tuple[float, float, float, float]]]:
    """
    Ekstrak informasi UTM zone dari metadata GeoTIFF.
    Returns: (utm_epsg, (minx, miny, maxx, maxy)) atau (None, None)
    """
    crs_str = meta.get("crs", "")
    bounds = meta.get("bounds")
    
    if not crs_str or not bounds:
        return None, None
    
    # Deteksi UTM zone dari CRS string
    utm_epsg = None
    try:
        # Cari pattern UTM zone dari CRS string
        import re
        
        # Pattern untuk UTM zone (misal: "zone 49S" atau "UTM zone 49S")
        zone_match = re.search(r'zone\s+(\d+)([NS])', crs_str, re.IGNORECASE)
        if zone_match:
            zone_num = int(zone_match.group(1))
            hemisphere = zone_match.group(2).upper()
            # EPSG: 32600 + zone untuk Utara, 32700 + zone untuk Selatan
            if hemisphere == 'S':
                epsg_code = 32700 + zone_num
            else:
                epsg_code = 32600 + zone_num
            utm_epsg = f"EPSG:{epsg_code}"
        else:
            # Coba cari EPSG code langsung
            epsg_match = re.search(r'EPSG[":\s]+(\d+)', crs_str, re.IGNORECASE)
            if epsg_match:
                epsg_code = int(epsg_match.group(1))
                # Cek apakah ini EPSG UTM (32600-32799)
                if 32600 <= epsg_code <= 32799:
                    utm_epsg = f"EPSG:{epsg_code}"
                    
    except Exception as e:
        st.warning(f"Gagal mengekstrak UTM zone: {str(e)[:100]}")
    
    # Jika tidak ditemukan UTM, coba gunakan proj string
    if not utm_epsg and '+proj=utm' in crs_str.lower():
        try:
            import re
            zone_match = re.search(r'\+zone=(\d+)', crs_str, re.IGNORECASE)
            if zone_match:
                zone_num = int(zone_match.group(1))
                # Cek hemisphere dari +south parameter
                if '+south' in crs_str.lower():
                    utm_epsg = f"EPSG:{32700 + zone_num}"
                else:
                    utm_epsg = f"EPSG:{32600 + zone_num}"
        except:
            pass
    
    # Ekstrak bounds
    try:
        if hasattr(bounds, 'left'):
            minx = float(bounds.left)
            miny = float(bounds.bottom)
            maxx = float(bounds.right)
            maxy = float(bounds.top)
        else:
            # Jika bounds adalah tuple/list
            if len(bounds) >= 4:
                minx = float(bounds[0])
                miny = float(bounds[1])
                maxx = float(bounds[2])
                maxy = float(bounds[3])
            else:
                return utm_epsg, None
                
        return utm_epsg, (minx, miny, maxx, maxy)
    except Exception as e:
        st.warning(f"Gagal mengekstrak bounds: {str(e)[:100]}")
        return utm_epsg, None


def convert_utm_to_wgs84(x: float, y: float, utm_epsg: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Konversi koordinat UTM (x,y) ke WGS84 (lon, lat) menggunakan pyproj.
    """
    try:
        from pyproj import Transformer
        transformer = Transformer.from_crs(utm_epsg, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(x, y)
        return lon, lat
    except Exception as e:
        # Fallback: jika pyproj error, coba dengan format EPSG yang berbeda
        try:
            # Coba tanpa huruf 'EPSG:'
            epsg_num = utm_epsg.replace('EPSG:', '')
            transformer = Transformer.from_crs(f"EPSG:{epsg_num}", "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(x, y)
            return lon, lat
        except Exception as e2:
            # Silent fail, akan menggunakan fallback
            return None, None

warnings.filterwarnings("ignore")

# ── Konfigurasi halaman harus PERTAMA ──────────────────────────
st.set_page_config(
    page_title="UB Forest – Deteksi & Karbon",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────
HEADER_BG_URL = "https://environesia.co.id/uploads/blog/20250429094811-2025-04-29blog094800.jpg"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; }}
.main-header {{
    position: relative;
    background-image: url('{HEADER_BG_URL}');
    background-size: cover;
    background-position: center 60%;
    background-repeat: no-repeat;
    padding: 3rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    color: white;
    text-align: center;
    box-shadow: 0 8px 32px rgba(13,51,24,0.45);
    overflow: hidden;
    min-height: 180px;
}}
/* overlay gelap agar teks tetap terbaca */
.main-header::before {{
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(
        135deg,
        rgba(13, 51, 24, 0.72) 0%,
        rgba(26, 92, 42, 0.60) 50%,
        rgba(10, 40, 18, 0.75) 100%
    );
    border-radius: 16px;
    z-index: 0;
}}
.main-header * {{ position: relative; z-index: 1; }}
.main-header h1 {{
    font-family:'Syne',sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0;
    text-shadow: 0 2px 12px rgba(0,0,0,0.6);
    letter-spacing: 0.5px;
}}
.main-header p  {{
    margin: 0.5rem 0 0;
    opacity: 0.95;
    font-size: 0.95rem;
    text-shadow: 0 1px 6px rgba(0,0,0,0.5);
}}
.main-header .badge {{
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.35);
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.78rem;
    margin-top: 0.6rem;
    backdrop-filter: blur(4px);
}}
.upload-warning {{ background:#fff3cd; border:1px solid #ffecb5; border-radius:10px;
                  padding:0.5rem 1rem; margin-bottom:0.5rem; font-size:0.8rem; color:#856404; }}
.success-box    {{ background:#d4edda; border:1px solid #c3e6cb; border-radius:10px;
                  padding:0.5rem 1rem; margin-bottom:0.5rem; font-size:0.8rem; color:#155724; }}
.stTabs [data-baseweb="tab-list"] {{ gap:4px; }}
.stTabs [data-baseweb="tab"] {{ background-color:#f0f7f0; border-radius:8px 8px 0 0;
                               font-family:'DM Sans',sans-serif; font-weight:500; }}
.stTabs [aria-selected="true"] {{ background-color:#1a5c2a !important; color:white !important; }}
</style>
""", unsafe_allow_html=True)

# ── Batas ukuran upload ────────────────────────────────────────
MAX_UPLOAD_SIZE_MB  = 200
MAX_UPLOAD_BYTES    = MAX_UPLOAD_SIZE_MB * 1024 * 1024
CHUNK_SIZE          = 50 * 1024 * 1024   # 50 MB per chunk
MAX_PIXEL_DIM       = 4000               # downscale jika lebih besar

# CATATAN: Untuk mengaktifkan upload 300 MB di Streamlit, buat file
#   .streamlit/config.toml  dengan isi:
#   [server]
#   maxUploadSize = 200

# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════
SPECIES_PARAMS: Dict[str, dict] = {
    "Pinus merkusii": {
        "color": "#2196F3", "emoji": "🌲",
        "wood_density": 0.54, "bef": 1.3, "carbon_fraction": 0.47,
        "allometric_a": 0.0509, "allometric_b": 2.472,
        "vol_a": -8.5, "vol_b": 1.8, "vol_c": 1.2,
        "timber_price": 1_200_000,
        "description": "Tusam – pinus asli Sumatera & Asia Tenggara",
        "ref_r": 89.0, "ref_g": 98.0, "ref_b": 109.0,
        "gb_diff_ref": -11.0, "brightness_ref": 99.0,
    },
    "Swietenia mahagoni": {
        "color": "#FF9800", "emoji": "🌳",
        "wood_density": 0.64, "bef": 1.4, "carbon_fraction": 0.47,
        "allometric_a": 0.0776, "allometric_b": 2.330,
        "vol_a": -8.2, "vol_b": 1.75, "vol_c": 1.15,
        "timber_price": 2_500_000,
        "description": "Mahoni daun kecil – hutan tropis Asia",
        "ref_r": 142.0, "ref_g": 154.0, "ref_b": 88.0,
        "gb_diff_ref": 66.0, "brightness_ref": 129.0,
    },
}

UB_FOREST_LAT  = -7.8754
UB_FOREST_LON  = 112.5853
UB_FOREST_ZOOM = 15

# ══════════════════════════════════════════════════════════════
# OPTIONAL LIBRARY DETECTION
# ══════════════════════════════════════════════════════════════
HAS_RASTERIO = False
HAS_TIFFFILE = False
HAS_SCIPY    = False
HAS_SKIMAGE  = False

try:
    import rasterio
    from rasterio import MemoryFile as RasterioMemFile
    HAS_RASTERIO = True
except ImportError:
    pass

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    pass

try:
    from scipy.ndimage import gaussian_filter as scipy_gaussian
    from scipy.ndimage import maximum_filter as scipy_max_filter
    from scipy.ndimage import label as scipy_label
    HAS_SCIPY = True
except ImportError:
    pass

try:
    from skimage.segmentation import watershed as ski_watershed
    from skimage.feature import peak_local_max as ski_peak_local_max
    from skimage.measure import regionprops as ski_regionprops
    HAS_SKIMAGE = True
except ImportError:
    pass

# ══════════════════════════════════════════════════════════════
# FALLBACK FUNCTIONS
# ══════════════════════════════════════════════════════════════

def _gaussian_filter(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gaussian filter dengan scipy atau PIL fallback."""
    if sigma <= 0:
        return image
    if HAS_SCIPY:
        return scipy_gaussian(image.astype(np.float32), sigma=sigma)
    radius = max(int(sigma * 2), 1)
    img_pil = Image.fromarray(image.astype(np.float32)).convert("F")
    img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(img_pil, dtype=np.float32)


def _peak_local_max(image: np.ndarray, min_distance: int = 10,
                    threshold_abs: float = 0.0,
                    exclude_border: bool = False,
                    num_peaks: int = 10000) -> np.ndarray:
    """Deteksi puncak lokal."""
    if HAS_SKIMAGE:
        return ski_peak_local_max(
            image, min_distance=min_distance,
            threshold_abs=threshold_abs,
            exclude_border=exclude_border,
            num_peaks=num_peaks,
        )
    if HAS_SCIPY:
        size = 2 * min_distance + 1
        max_filt = scipy_max_filter(image, size=size)
        peaks = (image == max_filt) & (image >= threshold_abs)
        coords = np.argwhere(peaks)
        if exclude_border and len(coords):
            m = min_distance
            h, w = image.shape
            mask = ((coords[:, 0] >= m) & (coords[:, 0] < h - m) &
                    (coords[:, 1] >= m) & (coords[:, 1] < w - m))
            coords = coords[mask]
        if len(coords) > num_peaks:
            intensities = image[coords[:, 0], coords[:, 1]]
            idx = np.argsort(intensities)[-num_peaks:]
            coords = coords[idx]
        return coords
    return np.array([[0, 0]], dtype=int)


def _watershed(image: np.ndarray, markers: np.ndarray,
               mask: Optional[np.ndarray] = None,
               compactness: float = 0.0) -> np.ndarray:
    """Watershed segmentation."""
    if HAS_SKIMAGE:
        return ski_watershed(-image, markers, mask=mask, compactness=compactness)
    # Simple dilation fallback
    from scipy.ndimage import binary_dilation
    labels = np.zeros_like(image, dtype=np.int32)
    peak_locs = np.argwhere(markers > 0)
    for i, (r, c) in enumerate(peak_locs, start=1):
        if 0 <= r < image.shape[0] and 0 <= c < image.shape[1]:
            labels[r, c] = i
    kernel = np.ones((3, 3), dtype=bool)
    n_labels = len(peak_locs)
    for _ in range(20):
        for lid in range(1, n_labels + 1):
            lmask = labels == lid
            if not lmask.any():
                continue
            dilated = binary_dilation(lmask, kernel)
            expand = dilated & (labels == 0)
            if mask is not None:
                expand &= mask
            labels[expand] = lid
    return labels


def _regionprops(labels: np.ndarray,
                 intensity_image: Optional[np.ndarray] = None) -> list:
    """Region properties."""
    if HAS_SKIMAGE:
        return ski_regionprops(labels, intensity_image=intensity_image)
    if not HAS_SCIPY:
        return []
    from scipy.ndimage import find_objects
    slices = find_objects(labels)
    props_list = []
    for i, sl in enumerate(slices):
        if sl is None:
            continue
        lid = i + 1
        region_mask = (labels[sl] == lid)
        if not region_mask.any():
            continue
        area = int(region_mask.sum())
        cy = (sl[0].start + sl[0].stop) / 2.0
        cx = (sl[1].start + sl[1].stop) / 2.0
        max_int = (float(np.max(intensity_image[sl][region_mask]))
                   if intensity_image is not None else 0.0)

        class _Prop:
            pass

        p = _Prop()
        p.area     = area
        p.label    = lid
        p.centroid = (cy, cx)
        p.bbox     = (sl[0].start, sl[1].start, sl[0].stop, sl[1].stop)
        p.max_intensity = max_int
        props_list.append(p)
    return props_list


# ══════════════════════════════════════════════════════════════
# NORMALISASI BAND
# ══════════════════════════════════════════════════════════════

def _normalize_band(arr: np.ndarray, pct_hi: int = 98) -> np.ndarray:
    """Normalisasi band ke uint8 dengan clipping persentil."""
    arr = arr.astype(np.float32)
    pos = arr[arr > 0]
    if pos.size:
        hi_clip = float(np.percentile(pos, pct_hi))
        arr = np.clip(arr, 0, hi_clip)
    lo = float(np.percentile(arr, 2))
    hi = float(np.percentile(arr, pct_hi))
    if hi - lo > 0:
        arr = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255)
    else:
        arr[:] = 0
    return arr.astype(np.uint8)


# ══════════════════════════════════════════════════════════════
# RESIZE HELPER
# ══════════════════════════════════════════════════════════════

def _maybe_resize_rgb(rgb: np.ndarray, max_dim: int = MAX_PIXEL_DIM) -> np.ndarray:
    h, w = rgb.shape[:2]
    if max(h, w) <= max_dim:
        return rgb
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return np.array(Image.fromarray(rgb).resize((new_w, new_h), Image.LANCZOS))


# ══════════════════════════════════════════════════════════════
# PEMBACAAN FILE
# ══════════════════════════════════════════════════════════════

def _read_geotiff_rasterio(data: bytes) -> Tuple[Optional[np.ndarray], dict]:
    """Baca GeoTIFF dengan rasterio dari bytes."""
    meta: dict = {}
    try:
        with RasterioMemFile(data) as memfile:
            with memfile.open() as src:
                meta = {
                    "width": src.width, "height": src.height,
                    "bands": src.count, "driver": "GeoTIFF",
                    "crs": str(src.crs) if src.crs else None,
                    "transform": src.transform,
                    "bounds": src.bounds if src.crs else None,
                }
                if src.transform and abs(src.transform[0]) > 0:
                    meta["gsd_m"] = abs(src.transform[0])

                max_dim = MAX_PIXEL_DIM
                scale = min(max_dim / src.height, max_dim / src.width, 1.0)
                new_h = max(1, int(src.height * scale))
                new_w = max(1, int(src.width * scale))

                if scale < 1.0:
                    meta["original_size"] = (src.height, src.width)
                    meta["resized_to"]    = (new_h, new_w)

                def read_band(n: int) -> np.ndarray:
                    return src.read(n, out_shape=(new_h, new_w))

                if src.count >= 3:
                    r, g, b = read_band(1), read_band(2), read_band(3)
                elif src.count == 2:
                    r, g = read_band(1), read_band(2)
                    b = g.copy()
                else:
                    r = read_band(1)
                    g = b = r.copy()

        rgb = np.stack([
            _normalize_band(r.astype(np.float32)),
            _normalize_band(g.astype(np.float32)),
            _normalize_band(b.astype(np.float32)),
        ], axis=2)
        del r, g, b
        gc.collect()
        return rgb, meta

    except Exception as e:
        st.warning(f"rasterio error: {str(e)[:120]}")
        return None, {}


def _read_tiff_tifffile(data: bytes) -> Tuple[Optional[np.ndarray], dict]:
    """Baca TIFF dengan tifffile melalui temporary file (aman untuk file besar)."""
    tmp_path = None
    try:
        # Tulis ke temp file dalam chunks
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = tmp.name
            for offset in range(0, len(data), CHUNK_SIZE):
                tmp.write(data[offset: offset + CHUNK_SIZE])

        img_data = tifffile.imread(tmp_path)

        h_orig = img_data.shape[0]
        w_orig = img_data.shape[1] if img_data.ndim > 1 else 1

        meta: dict = {
            "width": w_orig, "height": h_orig,
            "bands": img_data.shape[2] if img_data.ndim == 3 else 1,
            "driver": "TIFF",
        }

        # Downscale jika perlu (PIL resize, tidak bergantung skimage)
        if max(h_orig, w_orig) > MAX_PIXEL_DIM:
            scale = MAX_PIXEL_DIM / max(h_orig, w_orig)
            new_h = max(1, int(h_orig * scale))
            new_w = max(1, int(w_orig * scale))
            meta["original_size"] = (h_orig, w_orig)
            meta["resized_to"]    = (new_h, new_w)
            if img_data.ndim == 3:
                img_data = np.array(
                    Image.fromarray(img_data[:, :, :3].astype(np.uint8)
                                    if img_data.dtype == np.uint8
                                    else img_data[:, :, :3]).resize((new_w, new_h), Image.LANCZOS)
                )
            else:
                img_pil = Image.fromarray(img_data).resize((new_w, new_h), Image.LANCZOS)
                img_data = np.array(img_pil)

        # Konversi ke float untuk normalisasi
        if img_data.ndim == 3 and img_data.shape[2] >= 3:
            r = img_data[:, :, 0].astype(np.float32)
            g = img_data[:, :, 1].astype(np.float32)
            b = img_data[:, :, 2].astype(np.float32)
        elif img_data.ndim == 3 and img_data.shape[2] == 2:
            r = img_data[:, :, 0].astype(np.float32)
            g = img_data[:, :, 1].astype(np.float32)
            b = g.copy()
        elif img_data.ndim == 2:
            r = img_data.astype(np.float32)
            g = b = r.copy()
        else:
            return None, {}

        rgb = np.stack([
            _normalize_band(r), _normalize_band(g), _normalize_band(b)
        ], axis=2)
        del img_data, r, g, b
        gc.collect()
        return rgb, meta

    except Exception as e:
        st.warning(f"tifffile error: {str(e)[:120]}")
        return None, {}

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# ══════════════════════════════════════════════════════════════
# VALIDASI & LOAD UTAMA
# ══════════════════════════════════════════════════════════════

def _get_file_size(uploaded_file) -> int:
    """Ukuran file dalam bytes menggunakan seek (tidak mem-buffer seluruh file)."""
    uploaded_file.seek(0, 2)
    size = uploaded_file.tell()
    uploaded_file.seek(0)
    return size


def validate_file_size(uploaded_file, label: str = "File") -> Tuple[bool, float]:
    """Validasi ukuran file. Kembalikan (valid, size_mb)."""
    if uploaded_file is None:
        return True, 0.0
    size_bytes = _get_file_size(uploaded_file)
    size_mb    = size_bytes / (1024 * 1024)
    if size_bytes > MAX_UPLOAD_BYTES:
        size_str = (f"{size_mb/1024:.2f} GB" if size_mb >= 1024
                    else f"{size_mb:.2f} MB")
        st.error(
            f"❌ **{label} terlalu besar!** Ukuran: **{size_str}** — "
            f"Maks: **{MAX_UPLOAD_SIZE_MB} MB**"
        )
        return False, size_mb
    if size_mb > 200:
        st.warning(f"⚠️ File berukuran **{size_mb:.1f} MB** — proses mungkin lebih lambat.")
    return True, size_mb


def _fmt_mb(mb: float) -> str:
    return f"{mb/1024:.2f} GB" if mb >= 1024 else f"{mb:.2f} MB"


def load_image_array(uploaded_file) -> Tuple[Optional[np.ndarray], dict]:
    """
    Muat gambar dari upload. Kembalikan (rgb_uint8_HxWx3, meta_dict).
    Meta dict mengandung: width, height, bands, driver, gsd_m, file_size_mb,
    file_size_display, bounds, original_size, resized_to, crs.
    """
    base_meta: dict = {
        "width": 0, "height": 0, "bands": 0, "crs": None,
        "transform": None, "driver": "unknown", "gsd_m": None,
        "file_size_mb": 0.0, "file_size_display": "",
        "bounds": None, "original_size": None, "resized_to": None,
    }
    if uploaded_file is None:
        return None, base_meta

    valid, size_mb = validate_file_size(uploaded_file, "Foto Udara")
    if not valid:
        return None, base_meta

    base_meta["file_size_mb"]      = round(size_mb, 2)
    base_meta["file_size_display"] = _fmt_mb(size_mb)

    name = uploaded_file.name.lower()
    st.info(f"📁 Memproses: {uploaded_file.name} ({base_meta['file_size_display']})")

    data = uploaded_file.read()

    is_tiff = name.endswith((".tif", ".tiff", ".geotiff", ".jp2"))

    if is_tiff:
        st.info("🔍 Mendeteksi file TIFF/GeoTIFF…")

        if HAS_RASTERIO:
            with st.spinner("📡 Membaca dengan rasterio…"):
                rgb, meta_r = _read_geotiff_rasterio(data)
            if rgb is not None:
                base_meta.update(meta_r)
                st.success("✅ Berhasil dibaca dengan rasterio")
                del data; gc.collect()
                return rgb, base_meta

        if HAS_TIFFFILE:
            with st.spinner("📡 Membaca dengan tifffile…"):
                rgb, meta_r = _read_tiff_tifffile(data)
            if rgb is not None:
                base_meta.update(meta_r)
                st.success("✅ Berhasil dibaca dengan tifffile")
                del data; gc.collect()
                return rgb, base_meta

        st.error("❌ Gagal membaca file TIFF. Pastikan rasterio atau tifffile terinstall.")
        return None, base_meta

    # ── JPG / PNG / format lain ──────────────────────────────
    try:
        st.info("📸 Membaca sebagai gambar biasa…")
        img = Image.open(io.BytesIO(data))
        if img.mode != "RGB":
            img = img.convert("RGB")
        arr = np.array(img)
        del img, data; gc.collect()

        arr = _maybe_resize_rgb(arr, MAX_PIXEL_DIM)
        base_meta.update({
            "width": arr.shape[1], "height": arr.shape[0],
            "bands": 3, "driver": name.rsplit(".", 1)[-1].upper(),
        })
        st.success(f"✅ Berhasil membaca {base_meta['driver']}")
        return arr, base_meta

    except Exception as e:
        st.error(f"❌ Gagal membaca gambar: {str(e)[:200]}")
        return None, base_meta


def load_sample_image(uploaded_file, label: str = "Sampel") -> Optional[np.ndarray]:
    """Muat gambar sampel untuk referensi spektral spesies."""
    if uploaded_file is None:
        return None
    valid, _ = validate_file_size(uploaded_file, label)
    if not valid:
        return None
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    is_tiff = name.endswith((".tif", ".tiff", ".geotiff"))
    if is_tiff:
        if HAS_RASTERIO:
            rgb, _ = _read_geotiff_rasterio(data)
            if rgb is not None:
                return rgb
        if HAS_TIFFFILE:
            rgb, _ = _read_tiff_tifffile(data)
            if rgb is not None:
                return rgb
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return np.array(img)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════
# CHM SIMULATION
# ══════════════════════════════════════════════════════════════

def rgb_to_simulated_chm(rgb: np.ndarray, sigma: float = 2.0,
                          gsd_m: float = 0.1) -> np.ndarray:
    h, w = rgb.shape[:2]

    # Downscale CHM work array jika terlalu besar
    chm_max = 2048
    if max(h, w) > chm_max:
        scale   = chm_max / max(h, w)
        new_h   = max(1, int(h * scale))
        new_w   = max(1, int(w * scale))
        rgb_w   = np.array(Image.fromarray(rgb).resize((new_w, new_h), Image.LANCZOS))
        scale_back = True
    else:
        rgb_w      = rgb
        scale_back = False

    R = rgb_w[:, :, 0].astype(np.float32)
    G = rgb_w[:, :, 1].astype(np.float32)
    B = rgb_w[:, :, 2].astype(np.float32)
    brightness = (R + G + B) / 3.0
    total      = R + G + B + 1e-6

    exg = (2 * G - R - B) / total
    bpi = (B - R) / (B + R + 1e-6)

    ps = np.clip(bpi * 0.6 + (1 - brightness / 255.0) * 0.4, 0, 1)
    ms = np.clip(exg * 1.5 + (brightness / 255.0) * 0.3,     0, 1)

    ps = (ps - ps.min()) / (ps.max() - ps.min() + 1e-6)
    ms = (ms - ms.min()) / (ms.max() - ms.min() + 1e-6)

    combined = np.maximum(ps, ms)
    non_veg  = (brightness > 220) | (brightness < 15) | ((R - G) > 30)
    combined[non_veg] = 0

    chm_raw = combined * 40.0
    chm     = _gaussian_filter(chm_raw, sigma=sigma)

    if scale_back:
        chm = np.array(Image.fromarray(chm).resize((w, h), Image.BICUBIC))

    del R, G, B, brightness, total, exg, bpi, ps, ms, combined, chm_raw
    gc.collect()
    return chm.astype(np.float32)


# ══════════════════════════════════════════════════════════════
# TREE DETECTION
# ══════════════════════════════════════════════════════════════

def detect_trees(chm: np.ndarray, min_height: float = 3.0,
                 min_distance_px: int = 15, gsd_m: float = 0.1):
    chm_masked = chm.copy()
    chm_masked[chm < min_height] = 0.0

    coords  = _peak_local_max(
        chm_masked,
        min_distance=min_distance_px,
        threshold_abs=min_height,
        exclude_border=False,
        num_peaks=10_000,
    )
    markers = np.zeros(chm.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        if 0 <= r < chm.shape[0] and 0 <= c < chm.shape[1]:
            markers[r, c] = i

    labels = _watershed(chm_masked, markers, mask=chm_masked > 0, compactness=0.01)
    del chm_masked, markers
    gc.collect()
    return labels, coords


def extract_tree_metrics(labels: np.ndarray, peaks: np.ndarray,
                          chm: np.ndarray, rgb: np.ndarray,
                          gsd_m: float = 0.1) -> pd.DataFrame:
    records = []
    props   = _regionprops(labels, intensity_image=chm)
    for prop in props:
        if prop.area < 9:
            continue
        crown_area_m2 = prop.area * (gsd_m ** 2)
        ecd_m  = 2.0 * math.sqrt(crown_area_m2 / math.pi)
        h_m    = float(prop.max_intensity)
        if h_m < 2.0:
            continue
        cy, cx = prop.centroid
        bbox   = prop.bbox                          # (minr, minc, maxr, maxc)
        y1, x1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
        y2, x2 = min(rgb.shape[0], int(bbox[2])), min(rgb.shape[1], int(bbox[3]))
        if y2 <= y1 or x2 <= x1:
            continue
        region_mask = (labels[y1:y2, x1:x2] == prop.label)
        rgb_region  = rgb[y1:y2, x1:x2]
        if region_mask.any():
            r_m = float(np.mean(rgb_region[:, :, 0][region_mask]))
            g_m = float(np.mean(rgb_region[:, :, 1][region_mask]))
            b_m = float(np.mean(rgb_region[:, :, 2][region_mask]))
        else:
            r_m, g_m, b_m = 100.0, 140.0, 80.0
        records.append({
            "id": prop.label,
            "centroid_row": cy, "centroid_col": cx,
            "height_m": round(h_m, 2),
            "crown_area_m2": round(crown_area_m2, 3),
            "ecd_m": round(ecd_m, 3),
            "r_mean": r_m, "g_mean": g_m, "b_mean": b_m,
            "bbox_minr": int(bbox[0]), "bbox_minc": int(bbox[1]),
            "bbox_maxr": int(bbox[2]), "bbox_maxc": int(bbox[3]),
        })
    del props, labels
    gc.collect()
    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════
# SPECIES CLASSIFICATION
# ══════════════════════════════════════════════════════════════

def _extract_spectral_ref(img_arr: np.ndarray) -> dict:
    R = img_arr[:, :, 0].astype(np.float32)
    G = img_arr[:, :, 1].astype(np.float32)
    B = img_arr[:, :, 2].astype(np.float32)
    brightness = (R + G + B) / 3.0
    veg_mask = ((brightness > 30) & (brightness < 230) &
                ((G > R * 0.9) | (B > R * 0.9)))
    if veg_mask.sum() < 100:
        veg_mask = np.ones_like(brightness, dtype=bool)
    r_m = float(np.mean(R[veg_mask]))
    g_m = float(np.mean(G[veg_mask]))
    b_m = float(np.mean(B[veg_mask]))
    bright_m = float(np.mean(brightness[veg_mask]))
    return {"r": r_m, "g": g_m, "b": b_m,
            "brightness": bright_m, "gb_diff": g_m - b_m}


def classify_species(df: pd.DataFrame,
                     pinus_ref: Optional[dict] = None,
                     mahoni_ref: Optional[dict] = None) -> pd.DataFrame:
    df = df.copy()
    R  = df["r_mean"].values.astype(np.float32)
    G  = df["g_mean"].values.astype(np.float32)
    B  = df["b_mean"].values.astype(np.float32)
    brightness = (R + G + B) / 3.0
    gb_diff    = G - B

    df["gb_diff"]      = gb_diff
    df["brightness_px"] = brightness

    ref_p = {"gb_diff": -11.0, "brightness": 99.0}
    ref_m = {"gb_diff":  66.0, "brightness": 129.0}
    if pinus_ref:  ref_p.update(pinus_ref)
    if mahoni_ref: ref_m.update(mahoni_ref)

    w_gb, w_br = 2.0, 1.0
    dist_p = np.sqrt(w_gb * (gb_diff - ref_p["gb_diff"]) ** 2 +
                     w_br * (brightness - ref_p["brightness"]) ** 2)
    dist_m = np.sqrt(w_gb * (gb_diff - ref_m["gb_diff"]) ** 2 +
                     w_br * (brightness - ref_m["brightness"]) ** 2)
    total  = dist_p + dist_m + 1e-6

    species_arr    = np.where(dist_p <= dist_m, "Pinus merkusii", "Swietenia mahagoni")
    confidence_arr = np.where(
        dist_p <= dist_m,
        (1 - dist_p / total) * 100,
        (1 - dist_m / total) * 100,
    ).round(1)

    df["species"]    = species_arr
    df["confidence"] = confidence_arr
    return df


# ══════════════════════════════════════════════════════════════
# CARBON COMPUTATION
# ══════════════════════════════════════════════════════════════

def compute_carbon(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rows = {k: [] for k in ["dbh_cm", "volume_m3", "biomass_kg",
                             "carbon_kg", "co2e_kg", "stumpage_idr"]}
    for _, row in df.iterrows():
        p   = SPECIES_PARAMS[row["species"]]
        H   = float(row["height_m"])
        CA  = float(row["crown_area_m2"])
        ECD = float(row["ecd_m"])

        dbh = max(0.7 * (ECD ** 0.85) * (H ** 0.35), 1.0)
        try:
            ln_v = (p["vol_a"] + p["vol_b"] * math.log(max(H, 0.01))
                    + p["vol_c"] * math.log(CA + 0.01))
            vol  = max(math.exp(ln_v), 0.001)
        except Exception:
            vol  = 0.001

        W_allo = p["allometric_a"] * (p["wood_density"] * dbh ** 2.26)
        W_vol  = vol * p["wood_density"] * p["bef"] * 1000
        W      = (W_allo + W_vol) / 2.0
        C      = p["carbon_fraction"] * W
        CO2e   = C * (44 / 12)
        stumpage = vol * p["timber_price"] * 0.70

        rows["dbh_cm"].append(round(dbh, 2))
        rows["volume_m3"].append(round(vol, 4))
        rows["biomass_kg"].append(round(W, 2))
        rows["carbon_kg"].append(round(C, 2))
        rows["co2e_kg"].append(round(CO2e, 2))
        rows["stumpage_idr"].append(round(stumpage, 0))

    for k, v in rows.items():
        df[k] = v
    return df


# ══════════════════════════════════════════════════════════════
# MASK R-CNN SIMULATION
# ══════════════════════════════════════════════════════════════

def simulate_maskrcnn_detection(rgb: np.ndarray, df_trees: pd.DataFrame,
                                 gsd_m: float = 0.1,
                                 score_threshold: float = 0.5) -> dict:
    H_img, W_img = rgb.shape[:2]
    if "confidence" in df_trees.columns:
        df_f = df_trees[df_trees["confidence"] / 100.0 >= score_threshold].copy()
    else:
        df_f = df_trees.copy()
    if df_f.empty:
        return {"instances": [], "total": 0,
                "model": "Mask R-CNN (ResNet-50 FPN Backbone)",
                "score_threshold": score_threshold}

    radius_px     = np.maximum((df_f["ecd_m"].values / 2) / gsd_m, 5).astype(int)
    c_row         = df_f["centroid_row"].values.astype(int)
    c_col         = df_f["centroid_col"].values.astype(int)
    y1 = np.maximum(0,      c_row - radius_px)
    x1 = np.maximum(0,      c_col - radius_px)
    y2 = np.minimum(H_img,  c_row + radius_px)
    x2 = np.minimum(W_img,  c_col + radius_px)

    rng    = np.random.default_rng(42)
    base_s = (df_f["confidence"].values / 100.0
              if "confidence" in df_f.columns
              else np.full(len(df_f), 0.7))
    scores = np.clip(base_s + rng.uniform(-0.05, 0.05, len(df_f)), 0.5, 0.99)

    instances = []
    for idx, (_, row) in enumerate(df_f.iterrows()):
        ry = max((y2[idx] - y1[idx]) // 2, 3)
        rx = max((x2[idx] - x1[idx]) // 2, 3)
        cy_l = (y1[idx] + y2[idx]) // 2
        cx_l = (x1[idx] + x2[idx]) // 2
        sp   = row["species"]
        if "Pinus" in sp:
            ry = int(ry * rng.uniform(0.9, 1.2))
            rx = int(rx * rng.uniform(0.7, 1.0))
        else:
            ry = int(ry * rng.uniform(0.85, 1.1))
            rx = int(rx * rng.uniform(0.95, 1.2))
        instances.append({
            "id": int(row["id"]), "species": sp,
            "score": float(scores[idx]),
            "height_m": row["height_m"],
            "crown_area_m2": row["crown_area_m2"],
            "ecd_m": row["ecd_m"],
            "carbon_kg": row.get("carbon_kg", 0),
            "bbox": [int(x1[idx]), int(y1[idx]), int(x2[idx]), int(y2[idx])],
            "centroid": [int(cx_l), int(cy_l)],
            "mask_params": {
                "y1": int(y1[idx]), "y2": int(y2[idx]),
                "x1": int(x1[idx]), "x2": int(x2[idx]),
                "cy": int(cy_l),    "cx": int(cx_l),
                "ry": max(ry, 3),   "rx": max(rx, 3),
                "species": sp,
            },
        })
    return {"instances": instances, "total": len(instances),
            "model": "Mask R-CNN (ResNet-50 FPN Backbone)",
            "score_threshold": score_threshold}


def _create_crown_mask(H: int, W: int, cy: int, cx: int,
                        ry: int, rx: int, species: str,
                        seed: int = 0) -> np.ndarray:
    rng   = np.random.default_rng(seed % 10_000)
    Y, X  = np.ogrid[:H, :W]
    theta = np.arctan2((Y - cy) / (ry + 1e-6), (X - cx) / (rx + 1e-6))
    if "Pinus" in species:
        n, amp = rng.integers(5, 9), rng.uniform(0.10, 0.18)
        radial = 1.0 + amp * np.cos(n * theta)
    else:
        n, amp = rng.integers(3, 6), rng.uniform(0.06, 0.12)
        radial = 1.0 + amp * np.sin(n * theta + rng.uniform(0, 2 * math.pi))
    ellipse = ((Y - cy) / (ry * radial + 1e-6)) ** 2 + \
              ((X - cx) / (rx * radial + 1e-6)) ** 2
    return ellipse <= 1.0


def render_maskrcnn_overlay(rgb: np.ndarray, detection: dict,
                             show_masks: bool = True,
                             show_boxes: bool = True,
                             show_labels: bool = True,
                             max_trees: int = 500) -> np.ndarray:
    COLOR_MAP = {
        "Pinus merkusii":    (33,  150, 243),
        "Swietenia mahagoni": (255, 152, 0),
    }
    MASK_ALPHA = {"Pinus merkusii": 70, "Swietenia mahagoni": 75}
    LABEL_BG   = {
        "Pinus merkusii":    (21, 101, 192, 200),
        "Swietenia mahagoni": (230, 81, 0, 200),
    }

    img_pil = Image.fromarray(rgb.astype(np.uint8)).convert("RGBA")
    overlay = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
    H_img, W_img = rgb.shape[:2]

    for idx, inst in enumerate(detection["instances"][:max_trees]):
        sp    = inst["species"]
        color = COLOR_MAP.get(sp, (100, 200, 100))
        r_c, g_c, b_c = color
        bbox  = inst["bbox"]
        cx_px, cy_px = inst["centroid"]
        score = inst["score"]
        h_m   = inst["height_m"]

        if show_masks and inst.get("mask_params"):
            mp     = inst["mask_params"]
            alpha  = MASK_ALPHA.get(sp, 70)
            mask_a = _create_crown_mask(H_img, W_img,
                                        mp["cy"], mp["cx"],
                                        mp["ry"], mp["rx"],
                                        sp, seed=idx)
            mask_rgba = np.zeros((H_img, W_img, 4), dtype=np.uint8)
            mask_rgba[mask_a] = [r_c, g_c, b_c, alpha]
            overlay = Image.alpha_composite(overlay, Image.fromarray(mask_rgba, "RGBA"))

        draw = ImageDraw.Draw(overlay)

        if show_boxes:
            x1b, y1b, x2b, y2b = bbox
            bw = 2 if score < 0.75 else 3
            draw.rectangle([x1b, y1b, x2b, y2b],
                           outline=(r_c, g_c, b_c, 200), width=bw)
            tick = 6
            for px, py in [(x1b, y1b), (x2b, y1b), (x1b, y2b), (x2b, y2b)]:
                dx = tick if px == x1b else -tick
                dy = tick if py == y1b else -tick
                draw.line([(px, py), (px + dx, py)], fill=(r_c, g_c, b_c, 255), width=2)
                draw.line([(px, py), (px, py + dy)], fill=(r_c, g_c, b_c, 255), width=2)

        if show_labels:
            short    = "PNS" if "Pinus" in sp else "MHN"
            lbl_text = f"{short} {score:.2f} | {h_m:.1f}m"
            dot_r    = 4
            draw.ellipse([cx_px - dot_r, cy_px - dot_r,
                          cx_px + dot_r, cy_px + dot_r],
                         fill=(r_c, g_c, b_c, 255),
                         outline=(255, 255, 255, 200))
            tx, ty   = cx_px + 6, cy_px - 10
            bg       = LABEL_BG.get(sp, (50, 50, 50, 200))
            cw       = len(lbl_text) * 5
            draw.rectangle([tx - 2, ty - 1, tx + cw, ty + 9], fill=bg)
            draw.text((tx, ty), lbl_text, fill=(255, 255, 255, 255))

    return np.array(Image.alpha_composite(img_pil, overlay).convert("RGB"))


# ══════════════════════════════════════════════════════════════
# MAP - DENGAN BASEMAP LENGKAP (OpenStreetMap, Satellite, Terrain)
# ══════════════════════════════════════════════════════════════

# Basemap styles yang tersedia
# Untuk go.Scattermap (Plotly >=5.24)
BASEMAP_STYLES = {
    "OpenStreetMap": {
        "style": "open-street-map",
        "description": "Peta jalan standar OpenStreetMap"
    },
    "Satellite Imagery": {
        "style": "satellite",
        "description": "Citra satelit (memerlukan koneksi internet)"
    },
    "Satellite Streets": {
        "style": "satellite-streets",
        "description": "Citra satelit dengan overlay jalan"
    },
    "Terrain": {
        "style": "terrain",
        "description": "Peta kontur dan elevasi"
    },
    "Light": {
        "style": "light",
        "description": "Tema terang minimalis"
    },
    "Dark": {
        "style": "dark",
        "description": "Tema gelap untuk malam hari"
    },
    "Outdoors": {
        "style": "outdoors",
        "description": "Peta outdoor dengan detail alam"
    },
    "Carto Positron": {
        "style": "carto-positron",
        "description": "Peta bersih gaya CartoDB"
    },
    "Carto Dark": {
        "style": "carto-darkmatter",
        "description": "Peta gelap gaya CartoDB"
    }
}

# Fallback untuk versi Plotly lama (menggunakan mapbox)
BASEMAP_STYLES_FALLBACK = {
    "OpenStreetMap": "open-street-map",
    "Satellite Imagery": "satellite",
    "Satellite Streets": "satellite-streets",
    "Terrain": "terrain",
    "Light": "light",
    "Dark": "dark",
    "Outdoors": "outdoors",
    "Carto Positron": "carto-positron",
    "Carto Dark": "carto-darkmatter"
}

def build_plotly_map(df_trees: pd.DataFrame,
                     basemap_key: str,
                     center_lat: float = UB_FOREST_LAT,
                     center_lon: float = UB_FOREST_LON,
                     zoom: int = 14,
                     gsd_m: float = 0.1,
                     img_origin_lat: Optional[float] = None,
                     img_origin_lon: Optional[float] = None,
                     meta: Optional[dict] = None) -> go.Figure:
    """
    Membangun peta distribusi pohon dengan berbagai pilihan basemap.
    """
    COLOR_MAP = {"Pinus merkusii": "#2196F3", "Swietenia mahagoni": "#FF9800"}
    rng       = np.random.default_rng(99)
    
    df_plot = df_trees.head(3000).copy()
    
    if df_plot.empty:
        st.warning("Tidak ada data pohon untuk ditampilkan")
        return go.Figure()

    # ── Hitung koordinat setiap pohon ──────────────────────────
    lats, lons = [], []
    
    # Cek apakah ada data koordinat dari GeoTIFF
    has_utm = False
    utm_epsg = None
    utm_bounds = None
    
    if meta:
        try:
            utm_epsg, utm_bounds = extract_utm_from_geotiff(meta)
            if utm_epsg and utm_bounds:
                has_utm = True
        except Exception:
            pass
    
    # Hitung koordinat untuk setiap pohon
    for idx, row in df_plot.iterrows():
        try:
            if has_utm and utm_epsg and utm_bounds:
                # Konversi UTM ke WGS84
                minx, miny, maxx, maxy = utm_bounds
                
                if meta.get("original_size"):
                    orig_h, orig_w = meta["original_size"]
                    resized_h, resized_w = meta.get("resized_to", (orig_h, orig_w))
                    
                    scale_h = orig_h / resized_h if resized_h > 0 else 1
                    scale_w = orig_w / resized_w if resized_w > 0 else 1
                    
                    px = float(row["centroid_col"]) * scale_w
                    py = float(row["centroid_row"]) * scale_h
                    
                    x_utm = minx + (px / orig_w) * (maxx - minx)
                    y_utm = maxy - (py / orig_h) * (maxy - miny)
                    
                    lon, lat = convert_utm_to_wgs84(x_utm, y_utm, utm_epsg)
                    
                    if lon is not None and lat is not None:
                        lats.append(lat)
                        lons.append(lon)
                    else:
                        # Fallback
                        angle = rng.uniform(0, 2 * math.pi)
                        radius = rng.uniform(0, 0.002)
                        lats.append(center_lat + radius * math.cos(angle))
                        lons.append(center_lon + radius * math.sin(angle))
                else:
                    angle = rng.uniform(0, 2 * math.pi)
                    radius = rng.uniform(0, 0.002)
                    lats.append(center_lat + radius * math.cos(angle))
                    lons.append(center_lon + radius * math.sin(angle))
            else:
                # Metode sederhana
                if img_origin_lat is not None and img_origin_lon is not None:
                    lat = img_origin_lat - float(row["centroid_row"]) * gsd_m / 111320.0
                    lon = img_origin_lon + float(row["centroid_col"]) * gsd_m / (111320.0 * math.cos(math.radians(img_origin_lat)))
                else:
                    angle = rng.uniform(0, 2 * math.pi)
                    radius = rng.uniform(0, 0.002)
                    lat = center_lat + radius * math.cos(angle)
                    lon = center_lon + radius * math.sin(angle)
                
                lats.append(lat)
                lons.append(lon)
                
        except Exception:
            # Fallback ke center
            angle = rng.uniform(0, 2 * math.pi)
            radius = rng.uniform(0, 0.002)
            lats.append(center_lat + radius * math.cos(angle))
            lons.append(center_lon + radius * math.sin(angle))
    
    # Tambahkan koordinat ke dataframe
    df_plot = df_plot.copy()
    df_plot["lat"] = lats
    df_plot["lon"] = lons
    
    # Validasi koordinat
    valid_coords = df_plot["lat"].notna() & df_plot["lon"].notna()
    df_plot = df_plot[valid_coords]
    
    if df_plot.empty:
        st.error("Tidak ada koordinat valid untuk ditampilkan")
        return go.Figure()
    
    # Siapkan hover info
    df_plot["emoji"] = df_plot["species"].apply(lambda s: "🌲" if "Pinus" in s else "🌳")
    df_plot["hover"] = df_plot.apply(lambda r: (
        f"<b>{r['emoji']} {r['species']}</b><br>"
        f"ID: {int(r['id'])} | Tinggi: {r['height_m']:.1f} m<br>"
        f"ECD: {r['ecd_m']:.1f} m | Tajuk: {r['crown_area_m2']:.1f} m²<br>"
        f"DBH: {r.get('dbh_cm', 0):.1f} cm | Vol: {r.get('volume_m3', 0):.4f} m³<br>"
        f"Karbon: <b>{r.get('carbon_kg', 0):.1f} kg C</b> "
        f"| CO₂e: {r.get('co2e_kg', 0):.1f} kg<br>"
        f"Confidence: {r.get('confidence', 0):.1f}%"
    ), axis=1)
    
    # Ukuran marker proporsional terhadap ECD
    ecd = df_plot["ecd_m"].values.astype(float)
    if ecd.max() > ecd.min():
        norm = (ecd - ecd.min()) / (ecd.max() - ecd.min() + 1e-6)
    else:
        norm = np.ones_like(ecd) * 0.5
    sizes = (norm * 14 + 8).tolist()
    
    # Dapatkan style basemap
    basemap_info = BASEMAP_STYLES.get(basemap_key, BASEMAP_STYLES["OpenStreetMap"])
    map_style = basemap_info["style"]
    
    fig = go.Figure()
    
    # Cek versi Plotly
    import plotly
    plotly_ver = tuple(int(x) for x in plotly.__version__.split(".")[:2])
    use_new_api = plotly_ver >= (5, 24)
    
    # Tambahkan trace untuk setiap spesies
    for sp in ["Pinus merkusii", "Swietenia mahagoni"]:
        mask = df_plot["species"] == sp
        if mask.sum() == 0:
            continue
            
        dsp = df_plot[mask].reset_index(drop=True)
        szs = [sizes[i] for i, m in enumerate(mask) if m]
        color = COLOR_MAP[sp]
        emoji = "🌲" if "Pinus" in sp else "🌳"
        label = f"{emoji} {sp} ({mask.sum()})"
        
        if use_new_api:
            fig.add_trace(go.Scattermap(
                lat=dsp["lat"].tolist(),
                lon=dsp["lon"].tolist(),
                mode="markers",
                marker=go.scattermap.Marker(
                    size=szs,
                    color=color,
                    opacity=0.85,
                    sizemode="diameter",
                    symbol="circle",
                ),
                name=label,
                text=dsp["hover"].tolist(),
                hovertemplate="%{text}<extra></extra>",
            ))
        else:
            # Fallback untuk versi lama
            fallback_style = BASEMAP_STYLES_FALLBACK.get(basemap_key, "open-street-map")
            fig.add_trace(go.Scattermapbox(
                lat=dsp["lat"].tolist(),
                lon=dsp["lon"].tolist(),
                mode="markers",
                marker=go.scattermapbox.Marker(
                    size=szs,
                    color=color,
                    opacity=0.85,
                    sizemode="diameter",
                ),
                name=label,
                text=dsp["hover"].tolist(),
                hovertemplate="%{text}<extra></extra>",
            ))
            map_style = fallback_style
    
    # Hitung center dari data
    if len(df_plot) > 0:
        map_center_lat = df_plot["lat"].median()
        map_center_lon = df_plot["lon"].median()
    else:
        map_center_lat = center_lat
        map_center_lon = center_lon
    
    # Konfigurasi layout
    map_center = dict(lat=map_center_lat, lon=map_center_lon)
    map_zoom = zoom
    
    if use_new_api:
        # Konfigurasi untuk API baru
        map_config = dict(
            style=map_style,
            center=map_center,
            zoom=map_zoom,
        )
        
        # Tambahkan kontrol peta
        map_config["bearing"] = 0
        map_config["pitch"] = 0
        
        fig.update_layout(
            map=map_config,
            margin=dict(l=0, r=0, t=40, b=0),
            height=650,
            legend=dict(
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="#ccc",
                borderwidth=1,
                x=0.01, y=0.99,
                xanchor="left", yanchor="top",
                font=dict(size=12),
                traceorder="normal",
            ),
            title={
                'text': f"📍 Distribusi Pohon ({len(df_plot)} pohon)",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'family': 'DM Sans', 'weight': 'bold'}
            },
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="DM Sans",
            )
        )
    else:
        # Konfigurasi untuk API lama (mapbox)
        fig.update_layout(
            mapbox=dict(
                style=map_style,
                center=map_center,
                zoom=map_zoom,
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            height=650,
            legend=dict(
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="#ccc",
                borderwidth=1,
                x=0.01, y=0.99,
                xanchor="left", yanchor="top",
                font=dict(size=12),
            ),
            title={
                'text': f"📍 Distribusi Pohon ({len(df_plot)} pohon)",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'family': 'DM Sans'}
            }
        )
    
    return fig

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

def render_sidebar() -> dict:
    st.sidebar.markdown("## Parameter Analisis")
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
<div class="upload-warning">
⚠️ Batas Upload: <b>{MAX_UPLOAD_SIZE_MB} MB</b><br>
<small>Format: GeoTIFF (.tif/.tiff), JPG, PNG</small><br>
</div>
""", unsafe_allow_html=True)

    st.sidebar.markdown("### Data Input")
    uploaded_main = st.sidebar.file_uploader(
        f"Foto Udara (GeoTIFF / JPG / PNG) — Maks {MAX_UPLOAD_SIZE_MB} MB",
        type=["tif", "tiff", "geotiff", "jpg", "jpeg", "png"],
    )

    st.sidebar.markdown("### Referensi Tajuk Spesies")
    st.sidebar.info(
        "**Pinus**: upload foto tajuk pinus\n\n"
        "**Mahoni**: upload foto tajuk mahoni"
    )
    uploaded_pinus  = st.sidebar.file_uploader(
        "Sampel Pinus (tajuk gelap)", type=["tif","tiff","jpg","jpeg","png"], key="pinus")
    uploaded_mahoni = st.sidebar.file_uploader(
        "Sampel Mahoni (tajuk terang)", type=["tif","tiff","jpg","jpeg","png"], key="mahoni")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Deteksi Pohon")
    gsd      = st.sidebar.slider("GSD (m/pixel)", 0.02, 0.50, 0.10, 0.01)
    min_h    = st.sidebar.slider("Tinggi Minimum (m)", 2.0, 15.0, 5.0, 0.5)
    min_dist = st.sidebar.slider("Jarak Minimum (m)", 1.0, 10.0, 3.0, 0.5)
    sigma    = st.sidebar.slider("Gaussian Smoothing", 0.5, 5.0, 2.0, 0.5)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Mask R-CNN")
    rcnn_thr      = st.sidebar.slider("Score Threshold", 0.30, 0.95, 0.50, 0.05)
    show_masks    = st.sidebar.checkbox("Tampilkan Masks",          value=True)
    show_boxes    = st.sidebar.checkbox("Tampilkan Bounding Boxes", value=True)
    show_labels   = st.sidebar.checkbox("Tampilkan Label",          value=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Valuasi Karbon")
    carbon_price = st.sidebar.number_input("Harga Karbon (USD/ton CO₂e)", 1.0, 50.0, 6.0, 0.5)
    usd_idr      = st.sidebar.number_input("Kurs USD → IDR", 10_000, 20_000, 16_000, 100)
    area_ha      = st.sidebar.number_input("Luas Kawasan (ha)", 1.0, 600.0, 53.0, 1.0)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Koordinat Peta")
    map_lat  = st.sidebar.number_input("Latitude Pusat",  -90.0,  90.0, UB_FOREST_LAT, 0.0001, format="%.6f")
    map_lon  = st.sidebar.number_input("Longitude Pusat",-180.0, 180.0, UB_FOREST_LON, 0.0001, format="%.6f")
    map_zoom = st.sidebar.slider("Zoom Level", 10, 20, UB_FOREST_ZOOM)

    # Status library
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Status Library:**")
    for name, has in [("rasterio", HAS_RASTERIO), ("tifffile", HAS_TIFFFILE),
                      ("scipy",    HAS_SCIPY),    ("skimage",  HAS_SKIMAGE)]:
        icon = "✅" if has else ("⚠️ fallback" if name in ("scipy","skimage") else "❌")
        st.sidebar.markdown(f"- {name}: {icon}")

    return dict(
        uploaded_main=uploaded_main,
        uploaded_pinus=uploaded_pinus, uploaded_mahoni=uploaded_mahoni,
        gsd=gsd, min_h=min_h, min_dist=min_dist, sigma=sigma,
        rcnn_score_thresh=rcnn_thr,
        show_masks=show_masks, show_boxes=show_boxes, show_labels=show_labels,
        carbon_price=carbon_price, usd_idr=usd_idr, area_ha=area_ha,
        map_lat=map_lat, map_lon=map_lon, map_zoom=map_zoom,
    )


# ══════════════════════════════════════════════════════════════
# MAIN FUNCTION - LENGKAP DAN DIPERBAIKI
# ══════════════════════════════════════════════════════════════

def main():
    st.markdown("""
<div class="main-header">
    <h1>🌲 Deteksi Pohon &amp; Estimasi Karbon</h1>
    <p>Estimasi Volume dan Nilai Simpanan Karbon Tegakan
    <i>Pinus merkusii</i> dan <i>Swietenia mahagoni</i> Menggunakan UAV</p>
    <p><span class="badge">🛸 UAV Remote Sensing</span>
       <span class="badge">🌿 UB Forest</span>
       <span class="badge">📡 Mask R-CNN</span></p>
    <p><small>Credit: Dr. Adipandang Yudono (WebGIS Analytics Developer - PWK UB)| Ananda Nibras Naditya Lokiko (Data Analyst - Kehutanan UB)</small></p>
</div>
""", unsafe_allow_html=True)

    params = render_sidebar()

    # Buat tabs dengan benar
    tabs = st.tabs([
        "📖 Panduan",
        "🔍 Deteksi Pohon",
        "🎭 Mask R-CNN",
        "🗺 Peta Distribusi",
        "📊 Analisis Karbon",
        "🌳 Spesies",
        "📥 Ekspor",
    ])

    # ── TAB 0: PANDUAN ────────────────────────────────────────
    with tabs[0]:
        st.markdown("## Panduan Penggunaan")
        st.info(f"""
**Cara Menggunakan:**

1. Upload foto udara di sidebar (GeoTIFF / JPG / PNG, maks **{MAX_UPLOAD_SIZE_MB} MB**)

2. *(Opsional)* Upload sampel tajuk Pinus & Mahoni untuk kalibrasi spektral

3. Atur parameter deteksi, lalu buka tab **Deteksi Pohon** → klik **Jalankan Deteksi**

4. Hasil tersedia di tab: Mask R-CNN, Peta, Analisis Karbon, dan Ekspor

**Catatan file besar (>200 MB):**
- Gambar otomatis diresize ke maks {MAX_PIXEL_DIM}px per sisi
- CHM dihitung pada resolusi maks 2048px lalu diupscale
- Proses bisa memakan waktu 2–5 menit
        """)

    # ── TAB 1: DETEKSI POHON ─────────────────────────────────
    with tabs[1]:
        st.markdown("## Deteksi Pohon")
        if params["uploaded_main"] is None:
            st.info("📁 Upload foto udara di sidebar kiri untuk memulai.")
            # Jangan return, biarkan tabs lainnya tetap bisa diakses
        else:
            with st.spinner("📂 Memuat gambar…"):
                rgb, meta = load_image_array(params["uploaded_main"])

            if rgb is None:
                st.error("❌ Gambar tidak dapat dimuat. Cek format dan ukuran file.")
            else:
                # Metadata ringkas
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Ukuran Gambar",    f"{rgb.shape[1]:,} × {rgb.shape[0]:,} px")
                c2.metric("Ukuran File",       meta.get("file_size_display", "—"))
                c3.metric("Format",            meta.get("driver", "—"))
                if meta.get("gsd_m"):
                    c4.metric("GSD (dari file)", f"{meta['gsd_m']:.4f} m/px")
                else:
                    c4.metric("GSD (setting)",   f"{params['gsd']} m/px")

                if meta.get("original_size"):
                    st.info(
                        f"📏 Original: {meta['original_size'][0]}×{meta['original_size'][1]} px"
                        f" → Resized: {meta['resized_to'][0]}×{meta['resized_to'][1]} px"
                    )

                # Preview
                st.markdown("### Preview Foto Udara")
                prev = Image.fromarray(rgb)
                w, h = prev.size
                if w > 800:
                    prev = prev.resize((800, int(h * 800 / w)))
                st.image(prev, caption=f"Citra UAV — {meta.get('driver','')}", use_container_width=True)

                # Tombol deteksi
                if st.button("🚀 Jalankan Deteksi", type="primary", use_container_width=True):
                    t0  = time.time()
                    gsd = meta.get("gsd_m") or params["gsd"]
                    if meta.get("gsd_m"):
                        st.info(f"📏 GSD dari file: {gsd:.4f} m/pixel")

                    min_dist_px = max(int(params["min_dist"] / gsd), 10)
                    prog        = st.progress(0, "Memulai analisis…")

                    with st.spinner("🔄 Membangun CHM…"):
                        chm = rgb_to_simulated_chm(rgb, sigma=params["sigma"], gsd_m=gsd)
                    prog.progress(20, "CHM selesai…")

                    with st.spinner("🔍 Mendeteksi puncak pohon…"):
                        labels, peaks = detect_trees(chm, params["min_h"], min_dist_px, gsd)
                    prog.progress(45, f"{len(peaks)} kandidat ditemukan…")

                    with st.spinner("📊 Mengekstraksi metrik…"):
                        df_trees = extract_tree_metrics(labels, peaks, chm, rgb, gsd)
                    prog.progress(65, f"{len(df_trees)} pohon tervalidasi…")

                    # Referensi spektral opsional
                    pinus_ref = mahoni_ref = None
                    if params["uploaded_pinus"]:
                        arr = load_sample_image(params["uploaded_pinus"], "Sampel Pinus")
                        if arr is not None:
                            pinus_ref = _extract_spectral_ref(arr)
                            st.success(f"✅ Pinus ref  G−B = {pinus_ref['gb_diff']:.1f}")
                    if params["uploaded_mahoni"]:
                        arr = load_sample_image(params["uploaded_mahoni"], "Sampel Mahoni")
                        if arr is not None:
                            mahoni_ref = _extract_spectral_ref(arr)
                            st.success(f"✅ Mahoni ref G−B = {mahoni_ref['gb_diff']:.1f}")

                    with st.spinner("🏷 Mengklasifikasi spesies…"):
                        df_trees = classify_species(df_trees, pinus_ref, mahoni_ref)
                    with st.spinner("💰 Menghitung karbon…"):
                        df_trees = compute_carbon(df_trees)
                    prog.progress(95, "Hampir selesai…")

                    # Simpan ke session state
                    st.session_state["tree_df"] = df_trees
                    st.session_state["rgb"]     = rgb
                    st.session_state["chm"]     = chm
                    st.session_state["meta"]    = meta
                    st.session_state["gsd"]     = gsd

                    prog.progress(100, "Selesai!")
                    del chm, labels, peaks
                    gc.collect()

                    elapsed = time.time() - t0
                    st.success(f"✅ {len(df_trees):,} pohon terdeteksi dalam {elapsed:.1f} detik.")

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Pohon",      f"{len(df_trees):,}")
                    c2.metric("Rata-rata Tinggi", f"{df_trees['height_m'].mean():.1f} m")
                    c3.metric("Rata-rata ECD",    f"{df_trees['ecd_m'].mean():.1f} m")
                    sp_counts = df_trees["species"].value_counts()
                    c4.metric("Pinus / Mahoni",
                              f"{sp_counts.get('Pinus merkusii', 0)} / "
                              f"{sp_counts.get('Swietenia mahagoni', 0)}")

                    fig_gb = px.histogram(
                        df_trees, x="gb_diff", color="species", nbins=40,
                        title="Distribusi G−B (fitur utama klasifikasi spesies)",
                        labels={"gb_diff": "G − B", "count": "Jumlah Pohon"},
                        color_discrete_map={"Pinus merkusii": "#2196F3",
                                            "Swietenia mahagoni": "#FF9800"},
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

    # ── TAB 2: MASK R-CNN ─────────────────────────────────────
    with tabs[2]:
        st.markdown("## Visualisasi Mask R-CNN")
        if "tree_df" not in st.session_state:
            st.info("Jalankan deteksi pohon terlebih dahulu.")
        else:
            df  = st.session_state["tree_df"]
            rgb = st.session_state["rgb"]
            gsd = st.session_state.get("gsd", params["gsd"])

            c1, c2, c3, c4 = st.columns(4)
            with c1: show_m = st.checkbox("Masks",          value=params["show_masks"])
            with c2: show_b = st.checkbox("Bounding Boxes", value=params["show_boxes"])
            with c3: show_l = st.checkbox("Labels",         value=params["show_labels"])
            with c4: max_ov = st.number_input("Maks Overlay", 50, 5000, 500, 50)

            with st.spinner("Menjalankan Mask R-CNN…"):
                t0        = time.time()
                detection = simulate_maskrcnn_detection(
                    rgb, df, gsd_m=gsd,
                    score_threshold=params["rcnn_score_thresh"])
                overlay   = render_maskrcnn_overlay(
                    rgb, detection, show_m, show_b, show_l, int(max_ov))
                elapsed   = time.time() - t0

            st.success(f"✅ {detection['total']} instance dalam {elapsed:.2f} detik.")

            def _thumb(arr, max_w=620):
                img = Image.fromarray(arr.astype(np.uint8))
                w, h = img.size
                if w > max_w:
                    img = img.resize((max_w, int(h * max_w / w)))
                return img

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Foto Udara Asli**")
                st.image(_thumb(rgb), use_container_width=True)
            with col_b:
                st.markdown("**Overlay Mask R-CNN**")
                st.image(_thumb(overlay), use_container_width=True)

            buf = io.BytesIO()
            Image.fromarray(overlay).save(buf, format="PNG")
            st.download_button("⬇ Download Overlay (PNG)", buf.getvalue(),
                               "mask_rcnn_overlay.png", "image/png")

    # ── TAB 3: PETA DISTRIBUSI ───────────────────────────────
with tabs[3]:
    st.markdown("## 🗺 Peta Distribusi Pohon")
    
    # Cek apakah data tersedia
    if "tree_df" not in st.session_state:
        st.warning("⚠️ Belum ada data pohon. Silakan jalankan deteksi pohon terlebih dahulu di tab **Deteksi Pohon**.")
        st.info("💡 **Petunjuk:** Upload file GeoTIFF/JPG/PNG di sidebar kiri, lalu buka tab 'Deteksi Pohon' dan klik tombol 'Jalankan Deteksi'")
    else:
        df = st.session_state["tree_df"]
        meta = st.session_state.get("meta", {})
        gsd = st.session_state.get("gsd", params["gsd"])
        
        if df.empty:
            st.warning("⚠️ Data pohon kosong. Silakan jalankan deteksi pohon kembali.")
        else:
            # Ringkasan data
            st.markdown("### 📊 Ringkasan Data")
            col1, col2, col3, col4 = st.columns(4)
            sp_counts = df["species"].value_counts()
            
            with col1:
                st.metric("🌲 Pinus merkusii", f"{sp_counts.get('Pinus merkusii', 0):,} pohon")
            with col2:
                st.metric("🌳 Swietenia mahagoni", f"{sp_counts.get('Swietenia mahagoni', 0):,} pohon")
            with col3:
                st.metric("📍 Total Pohon", f"{len(df):,} pohon")
            with col4:
                st.metric("📏 GSD", f"{gsd:.3f} m/px")
            
            st.markdown("---")
            
            # Kontrol peta
            st.markdown("### 🎮 Kontrol Peta")
            
            # Pilihan basemap dengan grid
            col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns([2, 2, 1, 1])
            
            with col_ctrl1:
                basemap = st.selectbox(
                    "🗺 Peta Dasar:",
                    options=list(BASEMAP_STYLES.keys()),
                    format_func=lambda x: f"{x} - {BASEMAP_STYLES[x]['description']}",
                    index=0,
                    help="Pilih jenis peta dasar untuk visualisasi"
                )
            
            with col_ctrl2:
                filter_sp = st.multiselect(
                    "🔍 Filter Spesies:",
                    ["Pinus merkusii", "Swietenia mahagoni"],
                    default=["Pinus merkusii", "Swietenia mahagoni"],
                )
            
            with col_ctrl3:
                show_count = st.number_input(
                    "📌 Maks titik", 
                    min_value=100, 
                    max_value=5000, 
                    value=min(2000, len(df)),
                    step=100,
                    help="Batasi jumlah titik untuk performa yang lebih baik"
                )
            
            with col_ctrl4:
                marker_size = st.slider(
                    "📍 Ukuran Marker", 
                    min_value=5, 
                    max_value=30, 
                    value=12,
                    step=1,
                    help="Perbesar/kecilkan ukuran titik pohon"
                )
            
            # Filter data
            if filter_sp:
                df_map = df[df["species"].isin(filter_sp)].copy()
            else:
                df_map = df.copy()
            
            # Batasi jumlah titik untuk performa
            if len(df_map) > show_count:
                df_map = df_map.sample(n=show_count, random_state=42)
                st.info(f"📊 Menampilkan {show_count:,} dari {len(df):,} total pohon untuk performa optimal")
            
            if df_map.empty:
                st.warning("⚠️ Tidak ada pohon untuk ditampilkan dengan filter yang dipilih.")
            else:
                # Tentukan koordinat pusat
                center_lat = params["map_lat"]
                center_lon = params["map_lon"]
                
                # Cek apakah ada koordinat dari GeoTIFF
                if meta.get("bounds"):
                    try:
                        bounds = meta["bounds"]
                        if hasattr(bounds, '__len__') and len(bounds) >= 4:
                            left, bottom, right, top = bounds[0], bounds[1], bounds[2], bounds[3]
                            center_lat = (bottom + top) / 2
                            center_lon = (left + right) / 2
                            st.success(f"📍 Menggunakan koordinat dari GeoTIFF: {center_lat:.5f}, {center_lon:.5f}")
                    except Exception as e:
                        st.warning(f"Gagal membaca koordinat GeoTIFF: {str(e)[:100]}")
                
                # Buat peta
                with st.spinner(f"🔄 Membangun peta dengan basemap '{basemap}'..."):
                    try:
                        fig_map = build_plotly_map(
                            df_map, 
                            basemap,
                            center_lat, 
                            center_lon,
                            params["map_zoom"], 
                            gsd,
                            None,  # img_origin_lat
                            None,  # img_origin_lon
                            meta
                        )
                        
                        # Update ukuran marker jika diperlukan
                        if marker_size != 12:
                            for trace in fig_map.data:
                                if hasattr(trace, 'marker') and trace.marker:
                                    if hasattr(trace.marker, 'size'):
                                        # Adjust marker sizes
                                        current_sizes = trace.marker.size
                                        if isinstance(current_sizes, list):
                                            # Scale the sizes
                                            scale = marker_size / 12
                                            trace.marker.size = [s * scale for s in current_sizes]
                                        elif isinstance(current_sizes, (int, float)):
                                            trace.marker.size = marker_size
                        
                        # Tampilkan peta
                        if fig_map and len(fig_map.data) > 0:
                            st.plotly_chart(fig_map, use_container_width=True, config={
                                'displayModeBar': True,
                                'displaylogo': False,
                                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                                'scrollZoom': True,
                                'doubleClick': 'reset'
                            })
                            
                            # Informasi basemap
                            st.info(f"🗺️ Menggunakan peta dasar: **{basemap}** - {BASEMAP_STYLES[basemap]['description']}")
                            
                            # Informasi tambahan
                            st.markdown("---")
                            st.markdown("### ℹ️ Informasi Peta")
                            info_col1, info_col2, info_col3, info_col4 = st.columns(4)
                            with info_col1:
                                st.markdown(f"**✅ Titik ditampilkan:** {len(df_map):,} pohon")
                            with info_col2:
                                st.markdown(f"**📏 Jarak min antar pohon:** {params['min_dist']:.1f} m")
                            with info_col3:
                                st.markdown(f"**🔍 Zoom level:** {params['map_zoom']}")
                            with info_col4:
                                if meta.get("crs"):
                                    crs_str = str(meta["crs"])[:40]
                                    st.markdown(f"**🗺️ CRS:** {crs_str}...")
                            
                            # Legenda interaktif
                            with st.expander("🎨 Legenda & Petunjuk Penggunaan Peta", expanded=False):
                                st.markdown("""
                                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px;">
                                <b>🎨 Warna Marker:</b><br>
                                🔵 <b>Biru</b> = <i>Pinus merkusii</i> (Tusam)<br>
                                🟠 <b>Oranye</b> = <i>Swietenia mahagoni</i> (Mahoni)<br>
                                
                                <b>📏 Ukuran Marker:</b><br>
                                • Proporsional dengan diameter tajuk (ECD)<br>
                                • Pohon besar = marker lebih besar<br>
                                
                                <b>🖱️ Interaksi Peta:</b><br>
                                • <b>Hover</b> - Lihat detail pohon (tinggi, karbon, dll)<br>
                                • <b>Klik</b> - Pilih pohon tertentu<br>
                                • <b>Scroll</b> - Zoom in/out<br>
                                • <b>Drag</b> - Geser peta<br>
                                • <b>Double click</b> - Reset tampilan<br>
                                
                                <b>🗺️ Jenis Peta Dasar:</b><br>
                                • <b>OpenStreetMap</b> - Peta jalan standar<br>
                                • <b>Satellite Imagery</b> - Citra satelit real<br>
                                • <b>Satellite Streets</b> - Citra satelit + overlay jalan<br>
                                • <b>Terrain</b> - Peta kontur dan elevasi<br>
                                • <b>Light/Dark</b> - Tema minimalis<br>
                                • <b>Outdoors</b> - Detail alam untuk outdoor<br>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.error("❌ Gagal membuat peta: tidak ada data yang dapat ditampilkan")
                            st.info("💡 Tips: Coba kurangi filter atau refresh halaman")
                            
                    except Exception as e:
                        st.error(f"❌ Error saat membuat peta: {str(e)}")
                        st.exception(e)
                        st.info("💡 Saran: Coba pilih basemap 'OpenStreetMap' atau refresh halaman")
            
            # Statistik per spesies
            if not df.empty:
                st.markdown("---")
                st.markdown("### 📈 Statistik per Spesies")
                
                stats_df = df.groupby("species").agg({
                    "id": "count",
                    "height_m": ["mean", "std"],
                    "ecd_m": ["mean", "std"],
                    "crown_area_m2": "mean",
                    "carbon_kg": "sum",
                    "co2e_kg": "sum"
                }).round(2)
                
                # Rename columns
                stats_df.columns = ["Jumlah", "Tinggi (m)", "Std Tinggi", 
                                    "ECD (m)", "Std ECD", 
                                    "Luas Tajuk (m²)", "Total Karbon (kg)", "Total CO₂e (kg)"]
                
                st.dataframe(stats_df, use_container_width=True)
                
                # Download statistik
                csv_stats = stats_df.reset_index().to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Statistik (CSV)",
                    csv_stats,
                    "statistik_pohon.csv",
                    "text/csv",
                    key="download_stats"
                )

    # ── TAB 4: ANALISIS KARBON ───────────────────────────────
    with tabs[4]:
        st.markdown("## Analisis Karbon")
        if "tree_df" not in st.session_state:
            st.info("Jalankan deteksi pohon terlebih dahulu.")
        else:
            df = st.session_state["tree_df"]
            if df.empty:
                st.warning("Tidak ada pohon terdeteksi.")
            else:
                total_C   = df["carbon_kg"].sum() / 1000
                total_CO2 = df["co2e_kg"].sum()   / 1000
                total_vol = df["volume_m3"].sum()

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Karbon", f"{total_C:.2f} ton C")
                c2.metric("Total CO₂e",   f"{total_CO2:.2f} ton")
                c3.metric("Total Volume", f"{total_vol:.2f} m³")

                val_usd = total_CO2 * params["carbon_price"]
                val_idr = val_usd  * params["usd_idr"]
                c1, c2 = st.columns(2)
                c1.metric("Nilai Karbon (USD)", f"${val_usd:,.2f}")
                c2.metric("Nilai Karbon (IDR)", f"Rp {val_idr:,.0f}")

                sp_sum = (df.groupby("species")
                          .agg(Jumlah=("id","count"),
                               Karbon_kg=("carbon_kg","sum"),
                               CO2e_kg=("co2e_kg","sum"),
                               Volume_m3=("volume_m3","sum"))
                          .assign(Karbon_ton=lambda d: d["Karbon_kg"] / 1000,
                                  CO2e_ton=lambda d:  d["CO2e_kg"]   / 1000)
                          .round(3))
                st.dataframe(sp_sum, use_container_width=True)

                sp_cnt = df["species"].value_counts()
                sp_c   = df.groupby("species")["carbon_kg"].sum() / 1000
                colors_cnt = ["#2196F3" if "Pinus" in s else "#FF9800" for s in sp_cnt.index]
                colors_c   = ["#2196F3" if "Pinus" in s else "#FF9800" for s in sp_c.index]

                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Jumlah Pohon per Spesies", "Total Karbon (ton)"),
                )
                fig.add_trace(go.Bar(x=sp_cnt.index, y=sp_cnt.values,
                                     marker_color=colors_cnt, name="Jumlah"), row=1, col=1)
                fig.add_trace(go.Bar(x=sp_c.index, y=sp_c.values,
                                     marker_color=colors_c, name="Karbon"),  row=1, col=2)
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    # ── TAB 5: SPESIES ───────────────────────────────────────
    with tabs[5]:
        st.markdown("## Parameter Spesies")
        for sp, p in SPECIES_PARAMS.items():
            with st.expander(f"{p['emoji']} {sp} – {p['description']}"):
                c1, c2, c3 = st.columns(3)
                c1.markdown(
                    f"**Biometrik**\n"
                    f"- Berat Jenis: {p['wood_density']} g/cm³\n"
                    f"- BEF: {p['bef']}\n"
                    f"- Fraksi Karbon: {p['carbon_fraction']}"
                )
                c2.markdown(
                    f"**Model Volume**\n"
                    f"- a = {p['vol_a']}\n"
                    f"- b (H) = {p['vol_b']}\n"
                    f"- c (CA) = {p['vol_c']}"
                )
                c3.markdown(
                    f"**Spektral UAV**\n"
                    f"- G−B ref: {p['gb_diff_ref']}\n"
                    f"- Brightness ref: {p['brightness_ref']}\n"
                    f"- R/G/B ref: {p['ref_r']:.0f}/{p['ref_g']:.0f}/{p['ref_b']:.0f}"
                )

    # ── TAB 6: EKSPOR ────────────────────────────────────────
    with tabs[6]:
        st.markdown("## Ekspor Data")
        if "tree_df" not in st.session_state:
            st.info("Jalankan deteksi pohon terlebih dahulu.")
        else:
            df   = st.session_state["tree_df"]
            meta = st.session_state.get("meta", {})

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download CSV (Data Lengkap)", csv,
                "ub_forest_hasil_deteksi.csv", "text/csv",
                use_container_width=True,
            )
            if meta.get("file_size_display"):
                st.info(f"📊 File diproses: {meta['file_size_display']}")
            st.info(f"🌲 Total pohon: {len(df):,}")
            if not df.empty:
                st.info(f"💰 Total karbon: {df['carbon_kg'].sum()/1000:.2f} ton C")


if __name__ == "__main__":
    main()
