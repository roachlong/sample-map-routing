import osmnx as ox
import numpy as np
import contextily as cx
import geopandas as gpd
from shapely.geometry import box
from PIL import Image

# --- FETCH REAL-WORLD GRID AND BOUNDS ---
def fetch_grid_and_bounds(address, distance, grid_size):
    point = ox.geocode(address)
    G = ox.graph_from_point(point, dist=distance * 2, network_type='drive')  # load a generous area first

    # Get center point
    lat, lon = point

    # Compute bounding box manually
    west, south, east, north = ox.utils_geo.bbox_from_point((lat, lon), dist=distance)

    # Trim graph to exact bounding box
    G = ox.truncate.truncate_graph_bbox(G, bbox=(west, south, east, north), truncate_by_edge=True)

    # Use trimmed nodes to build grid
    nodes = list(G.nodes(data=True))
    edges = list(G.edges())

    # lats = [data['y'] for _, data in nodes]
    # lngs = [data['x'] for _, data in nodes]

    # Recompute bounds from trimmed graph (to match satellite image)
    # north, south = max(lats), min(lats)
    # east, west = max(lngs), min(lngs)
    bounds = (north, south, east, west)
    print("📦 Bounds:", bounds)

    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    connections = []

    node_lookup = {node: data for node, data in nodes}
    for u, v in edges:
        if u in node_lookup and v in node_lookup:
            lat1, lon1 = node_lookup[u]['y'], node_lookup[u]['x']
            lat2, lon2 = node_lookup[v]['y'], node_lookup[v]['x']
            
            x1 = min(grid_size - 1, max(0, int((lon1 - west) / (east - west) * grid_size)))
            y1 = min(grid_size - 1, max(0, int((north - lat1) / (north - south) * grid_size)))
            x2 = min(grid_size - 1, max(0, int((lon2 - west) / (east - west) * grid_size)))
            y2 = min(grid_size - 1, max(0, int((north - lat2) / (north - south) * grid_size)))

            grid[y1, x1] = 1
            grid[y2, x2] = 1
            connections.append(((y1, x1), (y2, x2)))

    return grid, bounds, connections


# --- FETCH SATELLITE IMAGE ---
def get_satellite_image(bounds, zoom):
    west, south, east, north = bounds[3], bounds[1], bounds[2], bounds[0]
    extent = box(west, south, east, north)
    gdf = gpd.GeoDataFrame({'geometry': [extent]}, crs='EPSG:4326')
    gdf = gdf.to_crs(epsg=3857)

    img, ext = cx.bounds2img(*gdf.total_bounds, zoom=zoom, source=cx.providers.Esri.WorldImagery)
    return Image.fromarray(img).transpose(Image.FLIP_TOP_BOTTOM)