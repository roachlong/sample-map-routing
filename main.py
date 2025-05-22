from util.map_utils import fetch_grid_and_bounds, get_satellite_image
from ui.game_board import run_game
import os

# --- SETTINGS ---
ADDRESS = "Main Street, Winnett MT 59087, USA"
DISTANCE = 250  # meters
GRID_SIZE = int(DISTANCE / 2)  # 1 cell per 2 meters
ZOOM_LEVEL = 1
SCALE = 1

# --- ENTRY POINT ---
if __name__ == "__main__":
    print("📡 Fetching grid bounds for map...")
    grid, bounds, connections = fetch_grid_and_bounds(ADDRESS, DISTANCE, GRID_SIZE)
    print("📡 and getting satellite imagery...")
    location = ADDRESS.replace(",", "").replace(" ", "_").lower()
    filename = f"cache/{location}_{DISTANCE}_satellite.png"
    if os.path.exists(filename):
        from PIL import Image
        pil_img = Image.open(filename)
    else:
        pil_img = get_satellite_image(bounds, ZOOM_LEVEL)
        pil_img.save(filename)
    run_game(grid, pil_img, bounds, connections, GRID_SIZE, SCALE)