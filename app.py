import streamlit as st
import folium
from streamlit_folium import st_folium
import rasterio
import numpy as np
import glob
import os
from rasterio.warp import transform_bounds
from PIL import Image
import matplotlib.cm as cm

st.set_page_config(page_title="Kinabatangan Deforestation Detection", layout="wide")
st.title("Deforestation Detection | Kinabatangan, Sabah")
st.markdown("U-Net model predictions of forest loss, 2019-2023")

PRED_DIR = "predictions_web"


@st.cache_data
def get_bounds_and_thumbnail(tif_path, downsample=10):
    with rasterio.open(tif_path) as src:
        out_height = max(1, src.height // downsample)
        out_width = max(1, src.width // downsample)
        data = src.read(
            1,
            out_shape=(out_height, out_width),
            resampling=rasterio.enums.Resampling.average,
        )
        bounds = src.bounds
        lon_min, lat_min, lon_max, lat_max = transform_bounds(src.crs, "EPSG:4326", *bounds)
    return data, (lat_min, lon_min, lat_max, lon_max)


tile_paths = sorted(glob.glob(f"{PRED_DIR}/web_pred_*.tif"))
st.sidebar.write(f"Found {len(tile_paths)} prediction tiles")

if len(tile_paths) == 0:
    st.error(f"No prediction tiles found in '{PRED_DIR}'. See the README for how to generate them.")
else:
    threshold = st.sidebar.slider("Deforestation probability threshold", 0.0, 1.0, 0.5, 0.05)
    m = folium.Map(location=[5.4, 118.0], zoom_start=9, tiles="OpenStreetMap")
    for tp in tile_paths:
        data, (lat_min, lon_min, lat_max, lon_max) = get_bounds_and_thumbnail(tp)
        display_data = np.where((~np.isnan(data)) & (data >= threshold), data, np.nan)
        normed = np.clip(np.nan_to_num(display_data, nan=0.0), 0, 1)
        colored = cm.RdYlGn_r(normed, bytes=True)
        nan_mask = np.isnan(display_data)
        colored[nan_mask, 3] = 0
        folium.raster_layers.ImageOverlay(
            image=colored,
            bounds=[[lat_min, lon_min], [lat_max, lon_max]],
            opacity=0.7,
        ).add_to(m)
    st_folium(m, width=1200, height=700)

st.markdown("---")
st.markdown("Built with Sentinel-2 imagery, Google Earth Engine, and a U-Net trained on Hansen Global Forest Change labels.")
