import numpy as np

class GridWorld3D_10x10x10:
    def __init__(self):
        self.size = 10
        self.state = np.array([0, 0, 0])  # Start at (0,0,0)
        self.goal = np.array([9, 9, 9])   # Goal at (9,9,9)
        self.obstacles = [
            np.array([3, 3, 3]),
            np.array([5, 5, 5]),
            np.array([7, 7, 7]),
            np.array([2, 4, 6]),
            np.array([6, 4, 2])
        ]
        self.time_step = 0
        self.max_steps = 500  # Increased due to larger grid

    def reset(self):
        self.state = np.array([0, 0, 0])
        self.time_step = 0
        return self.state.copy()

    def _move_obstacles(self):
        # Define more complex movement patterns for obstacles
        t = self.time_step
        
        # Circular pattern in XY plane for first obstacle
        self.obstacles[0] = np.array([
            3 + int(2 * np.cos(t * 0.1)),
            3 + int(2 * np.sin(t * 0.1)),
            3 + int(np.sin(t * 0.05))
        ])

        # Diamond pattern for second obstacle
        self.obstacles[1] = np.array([
            5 + int(2 * np.cos(t * 0.15)),
            5 + int(2 * np.sin(t * 0.15)),
            5 + int(np.cos(t * 0.1))
        ])

        # Vertical pattern for third obstacle
        self.obstacles[2] = np.array([
            7,
            7,
            (7 + t % 3) % self.size
        ])

        # Diagonal pattern for fourth obstacle
        self.obstacles[3] = np.array([
            (2 + t) % self.size,
            (4 + t) % self.size,
            (6 + t) % self.size
        ])

        # Spiral pattern for fifth obstacle
        self.obstacles[4] = np.array([
            6 + int(np.cos(t * 0.2) * (t % 3)),
            4 + int(np.sin(t * 0.2) * (t % 3)),
            (2 + t) % self.size
        ])

    def _is_valid_position(self, pos):
        return all(0 <= p < self.size for p in pos)

    def _check_collision(self, pos):
        return any(np.array_equal(pos, obs) for obs in self.obstacles)

    def step(self, action):
        self.time_step += 1
        
        # Define possible actions (up, down, north, south, east, west)
        action_map = {
            0: np.array([0, 0, 1]),   # up
            1: np.array([0, 0, -1]),  # down
            2: np.array([0, 1, 0]),   # north
            3: np.array([0, -1, 0]),  # south
            4: np.array([1, 0, 0]),   # east
            5: np.array([-1, 0, 0])   # west
        }

        # Get the movement vector
        move = action_map[action]
        new_pos = self.state + move

        # Check if the move is valid
        if self._is_valid_position(new_pos) and not self._check_collision(new_pos):
            self.state = new_pos

        # Move obstacles after the agent's move
        self._move_obstacles()

        # Check if we hit an obstacle after they moved
        done = False
        reward = -1  # Small negative reward for each step

        if self._check_collision(self.state):
            reward = -10  # Larger negative reward for hitting an obstacle
            done = True
        elif np.array_equal(self.state, self.goal):
            reward = 100  # Large positive reward for reaching the goal
            done = True
        elif self.time_step >= self.max_steps:
            done = True

        return self.state.copy(), reward, done

    def get_state_size(self):
        return self.size ** 3

    def get_action_size(self):
        return 6  # 6 possible actions

    def get_current_state(self):
        return self.state.copy()

    def get_obstacle_positions(self):
        return [obs.copy() for obs in self.obstacles] 