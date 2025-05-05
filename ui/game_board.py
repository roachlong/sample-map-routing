import pygame
import numpy as np
import scipy.ndimage
from collections import defaultdict, deque
from ai.path_finder import a_star
from ai.dqn_agent import DQNAgent
import time

# --- SETTINGS ---
MIN_SCALE = 0.3
MAX_SCALE = 4.0

# --- CONVERSION HELPERS ---
def grid_to_latlon(i, j, grid_size, bounds):
    north, south, east, west = bounds
    lat = north - (i / (grid_size - 1)) * (north - south)
    lon = west + (j / (grid_size - 1)) * (east - west)
    return lat, lon

def grid_to_pixel(i, j, grid_size, image_width, image_height, bounds):
    lat, lon = grid_to_latlon(i, j, grid_size, bounds)
    north, south, east, west = bounds
    x = (lon - west) / (east - west) * (image_width - 1)
    y = (lat - south) / (north - south) * (image_height - 1)
    return x, y

def scale_centered_on_mouse(mouse_pos, old_scale, new_scale, offset_x, offset_y):
    mx, my = mouse_pos
    dx = mx - offset_x
    dy = my - offset_y
    scale_ratio = new_scale / old_scale
    new_offset_x = mx - dx * scale_ratio
    new_offset_y = my - dy * scale_ratio
    return new_offset_x, new_offset_y

def clamp_offset(offset_x, offset_y, scale, image_width, image_height, screen_width, screen_height):
    max_offset_x = 0
    max_offset_y = 0
    min_offset_x = screen_width - image_width * scale
    min_offset_y = screen_height - image_height * scale
    offset_x = max(min(offset_x, max_offset_x), min_offset_x)
    offset_y = max(min(offset_y, max_offset_y), min_offset_y)
    return offset_x, offset_y

def find_nearest_valid(grid, start):
    if grid[start[0], start[1]] == 1:
        return start
    distance, indices = scipy.ndimage.distance_transform_edt(1 - grid, return_indices=True)
    return [indices[0][start[0], start[1]], indices[1][start[0], start[1]]]

def random_goal(grid, exclude):
    while True:
        g = tuple(find_nearest_valid(grid, [np.random.randint(grid.shape[0]), np.random.randint(grid.shape[1])]))
        if g != exclude:
            return g

def find_closest_node(grid, px, py):
    # Flip Y to match vertically flipped image
    py_flipped = grid.shape[0] - py - 1
    if grid[py_flipped, px] == 1:
        return (py_flipped, px)

    distance, indices = scipy.ndimage.distance_transform_edt(1 - grid, return_indices=True)
    return (int(indices[0][py_flipped, px]), int(indices[1][py_flipped, px]))
                
# --- MAIN GAME LOOP ---
def run_game(grid, sat_image, bounds, connections, screen_size, grid_size, scale=1.0):
    pygame.init()
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption("Zoomable Satellite UI - Wayne, NJ")

    colors = {
        "agent": (0, 102, 255),
        "goal": (0, 255, 100),
        "grid": (255, 255, 0),
        "edge": (255, 0, 0),
        "path": (255, 200, 0)
    }

    zoom_speed = 0.1
    offset_x, offset_y = 0, 0

    # Convert connections to a neighbor map
    neighbor_map = defaultdict(list)
    for a, b in connections:
        neighbor_map[a].append(b)
        neighbor_map[b].append(a)

    trip_count = 0
    flash_toggle = True
    click_mode = 'start'  # toggles between 'start' and 'goal'
    agent_pos = None
    goal_pos = None
    path_trace = []
    path = []
    path_index = 0

    use_rl_agent = True
    model_path = "models/dqn_model.weights.h5"
    rl_agent = DQNAgent(state_size=4, action_size=8)  # max 8 neighbors assumed
    rl_agent.epsilon = 0.0  # inference only

    try:
        rl_agent.load(model_path)
        print("✅ Loaded DQN model from file.")
    except:
        print("⚠️ No pre-trained model found. Starting fresh.")

    step_delay = 0.1  # seconds between steps
    pause_duration = 1.5  # seconds to pause at goal
    last_step_time = time.time()
    
    sat_surface = pygame.image.fromstring(sat_image.tobytes(), sat_image.size, sat_image.mode)

    minimap_size = 150
    minimap_surface = pygame.transform.smoothscale(sat_surface, (minimap_size, minimap_size))
    minimap_rect = pygame.Rect(screen_size - minimap_size - 10, 10, minimap_size, minimap_size)
    show_minimap = True

    state_history = deque(maxlen=rl_agent.sequence_length)
    clock = pygame.time.Clock()
    running = True
    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    step_delay = max(0.01, step_delay - 0.05)
                elif event.key == pygame.K_DOWN:
                    step_delay = min(1.0, step_delay + 0.05)
                elif event.key == pygame.K_m:
                    show_minimap = not show_minimap
                elif event.key == pygame.K_r:
                    use_rl_agent = not use_rl_agent
                    print(f"🔄 Using {'DQN Agent' if use_rl_agent else 'A* Pathfinding'}")
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                gx = int((mx - offset_x) / (scale * sat_surface.get_width()) * grid_size)
                gy = int((my - offset_y) / (scale * sat_surface.get_height()) * grid_size)
                selected = find_closest_node(grid, gx, gy)
                print(f"📍 Selected node: {selected}")

                if grid[selected] == 1:
                    if click_mode == 'start':
                        agent_pos = selected
                        path_trace = [agent_pos]
                        click_mode = 'goal'
                    elif click_mode == 'goal':
                        goal_pos = selected
                        click_mode = None
                        path = a_star(agent_pos, goal_pos, neighbor_map)
                        path_index = 0
                        print("✅ Start and goal set. Beginning inference...")

        keys = pygame.key.get_pressed()

        if keys[pygame.K_EQUALS] or keys[pygame.K_KP_PLUS]:
            new_scale = scale * (1 + zoom_speed)
            offset_x, offset_y = scale_centered_on_mouse(pygame.mouse.get_pos(), scale, new_scale, offset_x, offset_y)
            scale = max(MIN_SCALE, min(MAX_SCALE, new_scale))
        if keys[pygame.K_MINUS] or keys[pygame.K_KP_MINUS]:
            new_scale = scale * (1 - zoom_speed)
            offset_x, offset_y = scale_centered_on_mouse(pygame.mouse.get_pos(), scale, new_scale, offset_x, offset_y)
            scale = max(MIN_SCALE, min(MAX_SCALE, new_scale))

        current_time = time.time()
        if use_rl_agent:
            if agent_pos and goal_pos and current_time - last_step_time > step_delay:
                dx = goal_pos[0] - agent_pos[0]
                dy = goal_pos[1] - agent_pos[1]
                state = np.array([agent_pos[0], agent_pos[1], dx, dy], dtype=np.float32) / grid_size
                state_history.append(state)  # ✅ ADD TO HISTORY

                if len(state_history) < rl_agent.sequence_length:
                    continue  # ✅ WAIT until enough states collected

                state_seq = np.stack(state_history, axis=0)  # Shape: (sequence_length, state_size)
                action_idx = rl_agent.act(state_seq)

                valid_actions = neighbor_map[agent_pos]
                if action_idx >= len(valid_actions):
                    action_idx = np.random.randint(len(valid_actions))

                next_pos = valid_actions[action_idx]
                agent_pos = next_pos
                path_trace.append(agent_pos)
                last_step_time = current_time

                if agent_pos == goal_pos:
                    print("🎯 Goal reached!")
                    goal_pos = None
                    path_trace = []
                    click_mode = 'goal'
                    trip_count += 1
                    last_step_time = time.time()
                    state_history.clear()  # ✅ Reset sequence after goal

        else:
            if path and path_index < len(path) and current_time - last_step_time > step_delay:
                agent_pos = path[path_index]
                path_index += 1
                last_step_time = current_time
            elif path_index >= len(path):
                if int(current_time * 2) % 2 == 0:
                    flash_toggle = not flash_toggle
                if current_time - last_step_time > pause_duration:
                    goal_pos = None
                    path_trace = []
                    click_mode = 'goal'
                    trip_count += 1
                    last_step_time = time.time()

        # Drawing code
        scaled_img = pygame.transform.smoothscale(sat_surface,
            (int(sat_surface.get_width() * scale), int(sat_surface.get_height() * scale)))
        offset_x, offset_y = clamp_offset(offset_x, offset_y, scale, sat_surface.get_width(), sat_surface.get_height(), screen_size, screen_size)

        screen.blit(scaled_img, (offset_x, offset_y))

        for i in range(1, len(path_trace)):
            x1, y1 = grid_to_pixel(path_trace[i-1][0], path_trace[i-1][1], grid_size, sat_surface.get_width(), sat_surface.get_height(), bounds)
            x2, y2 = grid_to_pixel(path_trace[i][0], path_trace[i][1], grid_size, sat_surface.get_width(), sat_surface.get_height(), bounds)
            pygame.draw.line(screen, colors["path"], (x1 * scale + offset_x, y1 * scale + offset_y), (x2 * scale + offset_x, y2 * scale + offset_y), 2)

        # Draw road edges
        for (a, b) in connections:
            x1, y1 = grid_to_pixel(a[0], a[1], grid_size, sat_surface.get_width(), sat_surface.get_height(), bounds)
            x2, y2 = grid_to_pixel(b[0], b[1], grid_size, sat_surface.get_width(), sat_surface.get_height(), bounds)
            pygame.draw.line(screen, colors["edge"], (int(x1 * scale + offset_x), int(y1 * scale + offset_y)), (int(x2 * scale + offset_x), int(y2 * scale + offset_y)), 1)

        if use_rl_agent or path_index < len(path):
            for i in range(1, len(path)):
                x1, y1 = grid_to_pixel(path[i - 1][0], path[i - 1][1], grid_size, sat_surface.get_width(), sat_surface.get_height(), bounds)
                x2, y2 = grid_to_pixel(path[i][0], path[i][1], grid_size, sat_surface.get_width(), sat_surface.get_height(), bounds)
                pygame.draw.line(screen, (0, 255, 255), (int(x1 * scale + offset_x), int(y1 * scale + offset_y)), (int(x2 * scale + offset_x), int(y2 * scale + offset_y)), 2)

        ax, ay = 0, 0
        if agent_pos:
            ax, ay = grid_to_pixel(agent_pos[0], agent_pos[1], grid_size, sat_surface.get_width(), sat_surface.get_height(), bounds)
        gx, gy = 0, 0
        if goal_pos:
            gx, gy = grid_to_pixel(goal_pos[0], goal_pos[1], grid_size, sat_surface.get_width(), sat_surface.get_height(), bounds)

        # Auto-pan to keep agent centered in view
        screen_center_x = screen_size / 2
        screen_center_y = screen_size / 2

        # Smooth auto-pan toward the agent
        pan_speed = 0.1  # 0.0 = no movement, 1.0 = instant
        target_offset_x = screen_center_x - ax * scale
        target_offset_y = screen_center_y - ay * scale
        offset_x += (target_offset_x - offset_x) * pan_speed
        offset_y += (target_offset_y - offset_y) * pan_speed

        offset_x, offset_y = clamp_offset(offset_x, offset_y, scale, sat_surface.get_width(), sat_surface.get_height(), screen_size, screen_size)

        pygame.draw.circle(screen, colors["agent"], (int(ax * scale + offset_x), int(ay * scale + offset_y)), 6)
        if flash_toggle or path_index < len(path):
            pygame.draw.circle(screen, colors["goal"], (int(gx * scale + offset_x), int(gy * scale + offset_y)), 6)

        if show_minimap:
            # Mini-map rendering
            screen.blit(minimap_surface, minimap_rect)
            pygame.draw.rect(screen, (255, 255, 255), minimap_rect, 2)  # border

            # Draw agent and goal markers on minimap
            mini_ax, mini_ay = 0, 0
            if agent_pos:
                mini_ax = int(agent_pos[1] / grid_size * minimap_size)
                mini_ay = minimap_size - int(agent_pos[0] / grid_size * minimap_size)
            mini_gx, mini_gy = 0, 0
            if goal_pos:
                mini_gx = int(goal_pos[1] / grid_size * minimap_size)
                mini_gy = minimap_size - int(goal_pos[0] / grid_size * minimap_size)
            pygame.draw.circle(screen, colors["agent"], (minimap_rect.x + mini_ax, minimap_rect.y + mini_ay), 3)
            pygame.draw.circle(screen, colors["goal"], (minimap_rect.x + mini_gx, minimap_rect.y + mini_gy), 3)

            # Draw viewport rectangle
            view_w = screen_size / (sat_surface.get_width() * scale) * minimap_size
            view_h = screen_size / (sat_surface.get_height() * scale) * minimap_size
            view_x = (-offset_x / (sat_surface.get_width() * scale)) * minimap_size + minimap_rect.x
            view_y = minimap_rect.y + (-offset_y / (sat_surface.get_height() * scale)) * minimap_size
            pygame.draw.rect(screen, (255, 255, 0), pygame.Rect(view_x, view_y, view_w, view_h), 1)

        font = pygame.font.SysFont(None, 24)
        info_text = font.render(f"Speed: {step_delay:.2f}s | Trips: {trip_count} | M = Mini-Map | R = Toggle AI | Agent: {'RL' if use_rl_agent else 'A*'}", True, (255, 255, 255))
        screen.blit(info_text, (10, 10))
        
        pygame.display.flip()

    pygame.quit()
