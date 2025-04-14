from util.map_utils import fetch_grid_and_bounds
from ai.path_finder import a_star
from ai.dqn_agent import DQNAgent
from collections import defaultdict
import numpy as np
import os
import random
import sys
import time

ADDRESS = "Main Street, Winnett MT 59087, USA"
DISTANCE = 1000
GRID_SIZE = 250
EPISODES = 100
MODEL_PATH = "models/dqn_model.weights.h5"

# --- ENVIRONMENT WRAPPER ---
class GraphEnv:
    def __init__(self, grid, neighbor_map):
        self.grid = grid
        self.neighbor_map = neighbor_map
        self.agent_pos = self._random_valid()
        self.goal_pos = None
        self.path = None
        self.visited_edges = None
        self.steps = 0
        self.max_steps = 500

    def reset(self):
        self.goal_pos = self._random_valid(exclude=self.agent_pos)
        self.path = a_star(self.agent_pos, self.goal_pos, self.neighbor_map)
        self.steps = 0
        self.visited_edges = set()
        return self._get_state()

    def _get_state(self):
        dx = self.goal_pos[0] - self.agent_pos[0]
        dy = self.goal_pos[1] - self.agent_pos[1]
        return np.array([self.agent_pos[0], self.agent_pos[1], dx, dy], dtype=np.float32) / GRID_SIZE

    def _random_valid(self, exclude=None):
        while True:
            pos = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
            if self.grid[pos] == 1 and pos != exclude:
                return pos

    def step(self, action_idx):
        neighbors = self.neighbor_map[self.agent_pos]
        if not neighbors:
            return self._get_state(), -1, True, {}

        next_pos = neighbors[action_idx % len(neighbors)]
        prev_dist = np.linalg.norm(np.array(self.agent_pos) - np.array(self.goal_pos))
        new_dist = np.linalg.norm(np.array(next_pos) - np.array(self.goal_pos))

        #######
        # Each step the agent takes leads to an outcome
        # based on a Q-value associated with the state change.
        # You can adjust the Q-values with the reward calculation system,
        # i.e. positive poits for moving closer to the goal,
        # negative points for moving further away.
        #
        # i.e. I don't want the agent to traverse a loop, so I deduct points
        # when the agent follows an edge we've already visited
        #
        # If you add a traffic indicator to your state then you'll want to
        # assign points for no traffic, light, moderate, heavy, blocked, etc.
        #
        # But be careful, overly complex reward systems can confuse the agent,
        # i.e. we don't reward for following the A* path because it
        # may conflict with other incentives we use to find the goal.
        #######
        reward = (prev_dist - new_dist)  # distance improvement
        if reward < 0:
            reward *= 2; # punishment for moving further away

        reward -= 0.5  # step penalty
        # if next_pos in self.path:  # optional A* path bonus
        #     reward += 1 # this reward will confuse the agent
        if next_pos == self.goal_pos:
            reward += 100

        edge = (self.agent_pos, next_pos)
        if edge in self.visited_edges:
            reward -= 10  # discourage repeat
        else:
            self.visited_edges.add(edge)
    
        self.agent_pos = next_pos
        self.steps += 1
        done = self.agent_pos == self.goal_pos or self.steps >= self.max_steps
        return self._get_state(), reward, done, {}

# --- TRAINING LOOP ---
def train_dqn(grid, connections):
    neighbor_map = defaultdict(list)
    for a, b in connections:
        neighbor_map[a].append(b)
        neighbor_map[b].append(a)

    env = GraphEnv(grid, neighbor_map)
    agent = DQNAgent(state_size=4, action_size=8)

    try:
        agent.load(MODEL_PATH)
        print("✅ Loaded DQN model from file.")
    except:
        print("⚠️ No pre-trained model found. Starting fresh.")

    for ep in range(EPISODES):
        state = env.reset()
        done = False
        total_reward = 0
        start_time = time.time()

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            for _ in range(3):
                agent.replay(batch_size=32)
            state = next_state
            total_reward += reward
            if env.steps % 5 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nEpisode {ep+1}/{EPISODES} - Total reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.3f} - Time (min): {elapsed_time/60:.2f} - Steps: {env.steps}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    agent.save(MODEL_PATH)
    print("✅ Model saved to", MODEL_PATH)


if __name__ == "__main__":
    print("📡 Offline training starting...")
    grid, _, connections = fetch_grid_and_bounds(ADDRESS, DISTANCE, GRID_SIZE)
    train_dqn(grid, connections)
