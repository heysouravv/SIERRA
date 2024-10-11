import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

class ForestEnvironment:
    def __init__(self, size=100, tree_density=0.3):
        self.size = size
        self.grid = np.random.choice([0, 1, 2], size=(size, size), p=[1-tree_density, tree_density*0.7, tree_density*0.3])
        self.pheromone_map = np.zeros((size, size))
        self.exploration_map = np.zeros((size, size))

    def get_features(self, x, y, radius=2):
        x1, x2 = max(0, x-radius), min(self.size, x+radius+1)
        y1, y2 = max(0, y-radius), min(self.size, y+radius+1)
        area = self.grid[x1:x2, y1:y2]
        return np.array([np.sum(area == 1), np.sum(area == 2)]) / area.size

    def update_pheromones(self):
        self.pheromone_map *= 0.99  # Slower decay
        self.pheromone_map = np.clip(self.pheromone_map, 0, 1)

class Drone:
    def __init__(self, env, x, y, id):
        self.env = env
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.random.randn(2) * 0.1
        self.id = id
        self.history = [self.position.copy()]
        self.mapping_data = []
        self.max_speed = 1.0
        self.max_force = 0.1
        self.perception_radius = 5
        self.curiosity = 1.0  # Increased curiosity

    def update(self, swarm):
        acceleration = self.swarm_behavior(swarm)
        self.velocity += acceleration
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = self.velocity / speed * self.max_speed
        new_position = self.position + self.velocity
        if self.is_valid_move(new_position):
            self.position = new_position
        else:
            self.velocity = np.random.randn(2) * self.max_speed  # Random direction if blocked
        self.position = np.clip(self.position, 0, self.env.size - 1)
        self.history.append(self.position.copy())
        self.scan()
        self.leave_pheromone()

    def is_valid_move(self, position):
        x, y = int(position[0]), int(position[1])
        if 0 <= x < self.env.size and 0 <= y < self.env.size:
            return self.env.grid[x, y] == 0  # Only move to open spaces
        return False

    def swarm_behavior(self, swarm):
        separation = self.separate(swarm) * 1.5
        alignment = self.align(swarm) * 0.5
        cohesion = self.cohere(swarm) * 0.3
        obstacle_avoidance = self.avoid_obstacles() * 2.0
        pheromone_following = self.follow_pheromone() * 1.5
        exploration = self.explore() * self.curiosity
        return separation + alignment + cohesion + obstacle_avoidance + pheromone_following + exploration

    def separate(self, swarm):
        steering = np.zeros(2)
        for other in swarm:
            if other.id != self.id:
                diff = self.position - other.position
                d = np.linalg.norm(diff)
                if 0 < d < self.perception_radius:
                    steering += diff / d
        return steering

    def align(self, swarm):
        steering = np.zeros(2)
        count = 0
        for other in swarm:
            if other.id != self.id:
                d = np.linalg.norm(self.position - other.position)
                if d < self.perception_radius:
                    steering += other.velocity
                    count += 1
        if count > 0:
            steering /= count
            steering = self.limit(steering, self.max_force)
        return steering

    def cohere(self, swarm):
        steering = np.zeros(2)
        count = 0
        for other in swarm:
            if other.id != self.id:
                d = np.linalg.norm(self.position - other.position)
                if d < self.perception_radius:
                    steering += other.position
                    count += 1
        if count > 0:
            steering /= count
            steering -= self.position
            steering = self.limit(steering, self.max_force)
        return steering

    def avoid_obstacles(self):
        look_ahead = self.velocity * 3
        future_position = self.position + look_ahead
        x, y = int(future_position[0]), int(future_position[1])
        if 0 <= x < self.env.size and 0 <= y < self.env.size:
            if self.env.grid[x, y] > 0:  # If there's a tree
                return -look_ahead
        return np.zeros(2)

    def follow_pheromone(self):
        x, y = int(self.position[0]), int(self.position[1])
        pheromone_area = self.env.pheromone_map[max(0, x-2):min(self.env.size, x+3),
                                                max(0, y-2):min(self.env.size, y+3)]
        gradient = np.unravel_index(pheromone_area.argmin(), pheromone_area.shape)
        return np.array([gradient[0] - 2, gradient[1] - 2])

    def explore(self):
        x, y = int(self.position[0]), int(self.position[1])
        exploration_area = self.env.exploration_map[max(0, x-3):min(self.env.size, x+4),
                                                    max(0, y-3):min(self.env.size, y+4)]
        least_explored = np.unravel_index(exploration_area.argmin(), exploration_area.shape)
        return np.array([least_explored[0] - 3, least_explored[1] - 3])

    def limit(self, vector, max_magnitude):
        magnitude = np.linalg.norm(vector)
        if magnitude > max_magnitude:
            return vector / magnitude * max_magnitude
        return vector

    def scan(self):
        x, y = int(self.position[0]), int(self.position[1])
        features = self.env.get_features(x, y)
        self.mapping_data.append((x, y, features))
        self.env.exploration_map[max(0, x-1):min(self.env.size, x+2),
                                 max(0, y-1):min(self.env.size, y+2)] += 0.2
        self.env.exploration_map = np.clip(self.env.exploration_map, 0, 1)

    def leave_pheromone(self):
        x, y = int(self.position[0]), int(self.position[1])
        self.env.pheromone_map[x, y] += 0.5
        self.env.pheromone_map = np.clip(self.env.pheromone_map, 0, 1)

class Simulation:
    def __init__(self, env_size=100, num_drones=50, iterations=3000):
        self.env = ForestEnvironment(env_size)
        self.drones = self.initialize_drones(num_drones, env_size)
        self.iterations = iterations

    def initialize_drones(self, num_drones, env_size):
        drones = []
        for i in range(num_drones):
            x, y = self.find_open_space(env_size)
            drones.append(Drone(self.env, x, y, i))
        return drones

    def find_open_space(self, env_size):
        while True:
            x, y = np.random.randint(0, env_size), np.random.randint(0, env_size)
            if self.env.grid[x, y] == 0:
                return x, y

    def run(self):
        for _ in range(self.iterations):
            for drone in self.drones:
                drone.update(self.drones)
            self.env.update_pheromones()

    def visualize(self):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
        
        # Actual Forest
        forest_cmap = plt.cm.colors.ListedColormap(['white', 'limegreen', 'darkgreen'])
        ax1.imshow(self.env.grid, cmap=forest_cmap)
        ax1.set_title('Actual Forest')
        
        # Generated Forest Map
        all_data = [data for drone in self.drones for data in drone.mapping_data]
        positions = np.array([(data[0], data[1]) for data in all_data])
        features = np.array([data[2] for data in all_data])
        
        grid_x, grid_y = np.mgrid[0:self.env.size:1, 0:self.env.size:1]
        grid_z = griddata(positions, features[:, 0] - features[:, 1], (grid_x, grid_y), method='linear', fill_value=0)
        
        im = ax2.imshow(grid_z, cmap='RdYlBu', extent=[0, self.env.size, 0, self.env.size], vmin=-1, vmax=1)
        ax2.set_title('Generated Forest Map')
        plt.colorbar(im, ax=ax2, label='Deciduous - Coniferous')
        
        # Pheromone Map
        ax3.imshow(self.env.pheromone_map, cmap='viridis')
        ax3.set_title('Pheromone Map')
        
        # Exploration Density
        ax4.imshow(self.env.exploration_map, cmap='hot')
        ax4.set_title('Exploration Density')
        
        plt.tight_layout()
        plt.show()

# Run the simulation
sim = Simulation(env_size=100, num_drones=50, iterations=3000)
sim.run()
sim.visualize()



