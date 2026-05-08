"""
Visualize groundwater potential maps using folium (interactive) and matplotlib (static).
"""

import numpy as np
import rasterio
import folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from loguru import logger

MAPS_DIR = Path("outputs/maps")
FIGURES_DIR = Path("outputs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {0: "#d73027", 1: "#fee090", 2: "#1a9850"}  # Low=Red, Medium=Yellow, High=Green
LABELS = {0: "Low Potential", 1: "Medium Potential", 2: "High Potential"}


def plot_static_map(raster_path: str, title: str = "Groundwater Potential Map"):
    with rasterio.open(raster_path) as src:
        data = src.read(1).astype(float)
        data[data == src.nodata] = np.nan

    cmap = mcolors.ListedColormap([PALETTE[i] for i in sorted(PALETTE)])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(data, cmap=cmap, norm=norm)
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels([LABELS[i] for i in sorted(LABELS)])
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.axis("off")

    out = FIGURES_DIR / "groundwater_potential_map.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    logger.info(f"Static map saved: {out}")
    plt.show()


def create_interactive_map(raster_path: str) -> folium.Map:
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        center = [(bounds.top + bounds.bottom) / 2, (bounds.left + bounds.right) / 2]

    m = folium.Map(location=center, zoom_start=9, tiles="CartoDB positron")

    folium.raster_layers.ImageOverlay(
        image=raster_path,
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        opacity=0.7,
        name="Groundwater Potential",
    ).add_to(m)

    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background:white; padding:10px; border-radius:8px; font-size:13px;">
        <b>Groundwater Potential</b><br>
        <span style="background:#1a9850; padding:2px 8px;">&nbsp;</span> High<br>
        <span style="background:#fee090; padding:2px 8px;">&nbsp;</span> Medium<br>
        <span style="background:#d73027; padding:2px 8px;">&nbsp;</span> Low
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)

    out = FIGURES_DIR / "interactive_map.html"
    m.save(str(out))
    logger.info(f"Interactive map saved: {out}")
    return m


if __name__ == "__main__":
    raster = str(MAPS_DIR / "groundwater_potential_xgboost.tif")
    plot_static_map(raster)
    create_interactive_map(raster)
