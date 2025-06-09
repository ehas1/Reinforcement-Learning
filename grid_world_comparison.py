"""Compare Value Iteration, Policy Iteration, SARSA and Q-Learning on multi-dimensional grid worlds.

This script implements and analyzes different RL algorithms on custom grid environments
of varying dimensionality to compare their performance and convergence properties.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import gymnasium as gym
from gymnasium import spaces
import itertools
from mpl_toolkits.mplot3d import Axes3D
import random
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import partial

class MultiDimGrid(gym.Env):
    """Multi-dimensional grid world environment with obstacles."""
    
    def __init__(self, dims=(4, 4, 4), movement_cost=-0.01, n_obstacles=None):
        super(MultiDimGrid, self).__init__()
        
        self.dims = dims
        self.n_dims = len(dims)
        self.movement_cost = movement_cost
        
        # Action space: 2 actions per dimension (positive and negative movement)
        self.action_space = spaces.Discrete(2 * self.n_dims)
        
        # Observation space: tuple of coordinates
        self.observation_space = spaces.MultiDiscrete(dims)
        
        # Start state: (0, 0, ..., 0)
        self.start_state = tuple([0] * self.n_dims)
        
        # Goal state: (max, max, ..., max)
        self.goal_state = tuple([d-1 for d in dims])
        
        # Generate obstacles
        if n_obstacles is None:
            # Default to 10% of states being obstacles
            total_states = np.prod(dims)
            n_obstacles = int(0.1 * total_states)
        
        self.obstacles = self._generate_obstacles(n_obstacles)
        
        # Store visited states for visualization
        self.visited_states = defaultdict(int)
        self.current_state = None
        
        self.reset()
    
    def _generate_obstacles(self, n_obstacles):
        """Generate random obstacle positions, avoiding start and goal states."""
        obstacles = set()
        all_states = list(itertools.product(*[range(d) for d in self.dims]))
        valid_states = [s for s in all_states if s != self.start_state and s != self.goal_state]
        
        if n_obstacles > len(valid_states):
            n_obstacles = len(valid_states)
        
        obstacle_states = random.sample(valid_states, n_obstacles)
        obstacles.update(obstacle_states)
        
        return obstacles
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_state = self.start_state
        self.visited_states.clear()
        self.visited_states[self.current_state] += 1
        return self.current_state, {}
    
    def step(self, action):
        # Convert action to dimension and direction
        dim = action // 2
        direction = 1 if action % 2 == 0 else -1
        
        # Create list from current state for modification
        new_state = list(self.current_state)
        
        # Apply action
        new_state[dim] = min(max(0, new_state[dim] + direction), self.dims[dim] - 1)
        new_state = tuple(new_state)
        
        # Check if hit obstacle
        if new_state in self.obstacles:
            new_state = self.current_state  # Bounce back
            reward = -0.5  # Penalty for hitting obstacle
        else:
            # Check if reached goal
            done = new_state == self.goal_state
            reward = 1.0 if done else self.movement_cost
        
        self.current_state = new_state
        self.visited_states[self.current_state] += 1
        
        done = self.current_state == self.goal_state
        return new_state, reward, done, False, {}

def value_iteration(env, gamma=0.99, theta=1e-8, max_iterations=1000):
    """Optimized value iteration implementation."""
    nS = np.prod(env.dims)
    nA = env.action_space.n
    V = np.zeros(nS)
    rewards_over_time = []
    
    # Pre-compute state mappings
    state_to_idx = {state: i for i, state in enumerate(itertools.product(*[range(d) for d in env.dims]))}
    idx_to_state = {i: state for state, i in state_to_idx.items()}
    
    # Pre-compute valid states (non-obstacle states)
    valid_states = [s for s in state_to_idx.keys() if s not in env.obstacles]
    valid_indices = [state_to_idx[s] for s in valid_states]
    
    for iteration in range(max_iterations):
        delta = 0
        V_old = V.copy()
        
        # Vectorized update for all valid states
        for state_idx in valid_indices:
            state = idx_to_state[state_idx]
            values = np.zeros(nA)
            
            for action in range(nA):
                env.current_state = state
                next_state, reward, done, _, _ = env.step(action)
                values[action] = reward + gamma * V_old[state_to_idx[next_state]]
            
            V[state_idx] = np.max(values)
            delta = max(delta, abs(V[state_idx] - V_old[state_idx]))
        
        if iteration % 10 == 0:  # Reduce frequency of reward tracking
            policy = {idx_to_state[i]: np.argmax([_get_action_value(env, idx_to_state[i], a, V, state_to_idx, gamma) 
                                                 for a in range(nA)]) for i in valid_indices}
            rewards_over_time.append(_evaluate_policy(env, policy))
        
        if delta < theta:
            break
    
    # Get final policy
    policy = {idx_to_state[i]: np.argmax([_get_action_value(env, idx_to_state[i], a, V, state_to_idx, gamma) 
                                         for a in range(nA)]) for i in valid_indices}
    
    return V, policy, rewards_over_time

def _get_action_value(env, state, action, V, state_to_idx, gamma):
    """Helper function to compute action value."""
    env.current_state = state
    next_state, reward, done, _, _ = env.step(action)
    return reward + gamma * V[state_to_idx[next_state]]

def _evaluate_policy(env, policy):
    """Quick policy evaluation."""
    state = env.reset()[0]
    total_reward = 0
    done = False
    max_steps = 100  # Prevent infinite loops
    steps = 0
    
    while not done and steps < max_steps:
        action = policy.get(state, 0)
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        steps += 1
    
    return total_reward

def policy_iteration(env, gamma=0.99, max_iterations=100):
    """Optimized policy iteration implementation."""
    nS = np.prod(env.dims)
    nA = env.action_space.n
    
    # Pre-compute state mappings
    state_to_idx = {state: i for i, state in enumerate(itertools.product(*[range(d) for d in env.dims]))}
    idx_to_state = {i: state for state, i in state_to_idx.items()}
    valid_states = [s for s in state_to_idx.keys() if s not in env.obstacles]
    
    # Initialize random policy
    policy = {state: env.action_space.sample() for state in valid_states}
    rewards_over_time = []
    
    for iteration in range(max_iterations):
        # Policy Evaluation (simplified)
        V = defaultdict(float)
        for _ in range(10):  # Reduced iterations for faster convergence
            for state in valid_states:
                action = policy[state]
                env.current_state = state
                next_state, reward, done, _, _ = env.step(action)
                V[state] = reward + gamma * V[next_state]
        
        # Policy Improvement
        policy_stable = True
        for state in valid_states:
            old_action = policy[state]
            values = np.zeros(nA)
            for action in range(nA):
                env.current_state = state
                next_state, reward, done, _, _ = env.step(action)
                values[action] = reward + gamma * V[next_state]
            
            policy[state] = np.argmax(values)
            if old_action != policy[state]:
                policy_stable = False
        
        rewards_over_time.append(_evaluate_policy(env, policy))
        
        if policy_stable:
            break
    
    return V, policy, rewards_over_time

class QLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.1, epsilon=0.1, gamma=0.99):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = defaultdict(lambda: np.zeros(n_actions))
    
    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state):
        """Q-Learning update."""
        best_next_value = np.max(self.Q[next_state])
        td_target = reward + self.gamma * best_next_value
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.lr * td_error

class SarsaAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.1, epsilon=0.1, gamma=0.99):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = defaultdict(lambda: np.zeros(n_actions))
    
    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state, next_action):
        """SARSA update."""
        next_value = self.Q[next_state][next_action]
        td_target = reward + self.gamma * next_value
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.lr * td_error

def run_episode(env, agent, is_sarsa=False):
    """Run one episode with either Q-Learning or SARSA agent."""
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    if is_sarsa:
        action = agent.choose_action(state)
    
    while not done:
        if not is_sarsa:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.update(state, action, reward, next_state)
        else:
            next_state, reward, done, _, _ = env.step(action)
            next_action = agent.choose_action(next_state)
            agent.update(state, action, reward, next_state, next_action)
            action = next_action
            
        state = next_state
        total_reward += reward
    
    return total_reward

def visualize_state_visitation(env, policy, title):
    """Visualize state visitation patterns for a given policy."""
    plt.figure(figsize=(10, 8))
    
    ax = plt.axes(projection='3d')
    
    # Convert visited states to coordinates and counts
    x, y, z, counts = [], [], [], []
    for state, count in env.visited_states.items():
        x.append(state[0])
        y.append(state[1])
        z.append(state[2])
        counts.append(count)
    
    # Normalize counts for visualization
    counts = np.array(counts)
    counts = counts / counts.max()
    
    scatter = ax.scatter(x, y, z, c=counts, cmap='viridis', s=100)
    plt.colorbar(scatter, label='Normalized Visit Count')
    
    # Plot start and goal
    ax.scatter(*env.start_state, color='green', s=200, label='Start')
    ax.scatter(*env.goal_state, color='red', s=200, label='Goal')
    
    # Plot obstacles
    for obs in env.obstacles:
        ax.scatter(*obs, color='black', s=100, marker='x')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(f'{title} State Visitation Pattern')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}_visitation.png')
    plt.close()

def visualize_optimal_paths(env, policies, title="Optimal Paths Comparison"):
    """Optimized path visualization."""
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = {
        'Value Iteration': 'blue',
        'Policy Iteration': 'green',
        'Q-Learning': 'red',
        'SARSA': 'purple'
    }
    
    # Plot static elements
    ax.scatter(*env.start_state, color='lime', s=200, marker='*', label='Start')
    ax.scatter(*env.goal_state, color='gold', s=200, marker='*', label='Goal')
    
    # Plot obstacles efficiently
    if env.obstacles:
        obstacles = np.array(list(env.obstacles))
        ax.scatter(obstacles[:, 0], obstacles[:, 1], obstacles[:, 2], 
                  color='black', s=100, marker='x')
    
    # Plot paths efficiently
    for algo_name, policy in policies.items():
        path = []
        state = env.start_state
        visited = set()
        
        while state != env.goal_state and state not in visited and len(path) < 100:
            path.append(state)
            visited.add(state)
            
            # Handle different policy types
            if isinstance(policy, dict):
                if isinstance(policy[state], np.ndarray):
                    action = int(np.argmax(policy[state]))
                else:
                    action = policy[state]
            else:
                action = policy.choose_action(state)
            
            env.current_state = state
            state, _, _, _, _ = env.step(action)
        
        path.append(env.goal_state)
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], 
                color=colors[algo_name], linewidth=2, 
                label=f'{algo_name} Path', marker='o')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    ax.view_init(elev=20, azim=45)
    
    plt.savefig('optimal_paths_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_learning_curves(results, dims):
    """Create enhanced visualizations comparing all algorithms."""
    plt.figure(figsize=(20, 20))
    
    # Plot 1: Learning Curves with Confidence Intervals
    plt.subplot(3, 2, 1)
    window_size = 20
    for algo, rewards in results.items():
        rewards_array = np.array(rewards)
        avg_rewards = np.convolve(rewards_array, np.ones(window_size)/window_size, mode='valid')
        std_rewards = np.std([rewards_array[i:i+window_size] for i in range(len(rewards_array)-window_size+1)], axis=1)
        x = range(len(avg_rewards))
        plt.plot(x, avg_rewards, label=algo)
        plt.fill_between(x, avg_rewards-std_rewards, avg_rewards+std_rewards, alpha=0.2)
    
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title(f'Learning Curves with Confidence Intervals\n{dims}-dimensional Grid World')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Final Performance Distribution
    plt.subplot(3, 2, 2)
    data = [results[algo] for algo in results.keys()]
    violin_parts = plt.violinplot(data, showmeans=True)
    plt.xticks(range(1, len(results.keys())+1), results.keys(), rotation=45)
    plt.ylabel('Total Episode Reward')
    plt.title('Performance Distribution (Violin Plot)')
    
    # Plot 3: Convergence Rate
    plt.subplot(3, 2, 3)
    optimal_reward = max([max(results[a]) for a in results.keys()])
    for algo, rewards in results.items():
        convergence = [max(rewards[:i+1])/optimal_reward for i in range(len(rewards))]
        plt.plot(convergence, label=algo)
    
    plt.xlabel('Episode')
    plt.ylabel('% of Optimal Performance')
    plt.title('Convergence to Optimal Performance')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Success Rate Over Time
    plt.subplot(3, 2, 4)
    window_size = 20
    for algo, rewards in results.items():
        successes = [1 if r > 0 else 0 for r in rewards]
        success_rate = np.convolve(successes, np.ones(window_size)/window_size, mode='valid')
        plt.plot(success_rate, label=algo)
    
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.title(f'Success Rate (Moving Average, Window={window_size})')
    plt.legend()
    plt.grid(True)
    
    # Plot 5: Sample Efficiency (Cumulative Reward)
    plt.subplot(3, 2, 5)
    for algo, rewards in results.items():
        cumulative = np.cumsum(rewards)
        plt.plot(cumulative, label=algo)
    
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Sample Efficiency')
    plt.legend()
    plt.grid(True)
    
    # Plot 6: Stability (Rolling Variance)
    plt.subplot(3, 2, 6)
    window_size = 20
    for algo, rewards in results.items():
        rolling_var = [np.var(rewards[max(0, i-window_size):i+1]) for i in range(len(rewards))]
        plt.plot(rolling_var, label=algo)
    
    plt.xlabel('Episode')
    plt.ylabel('Rolling Variance')
    plt.title(f'Learning Stability (Window={window_size})')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'enhanced_comparison_{len(dims)}d.png')
    plt.close()

def compare_algorithms(dims=(4, 4, 4), num_episodes=100):
    """Optimized algorithm comparison."""
    env = MultiDimGrid(dims=dims)
    results = {
        'Value Iteration': [],
        'Policy Iteration': [],
        'Q-Learning': [],
        'SARSA': []
    }
    policies = {}
    
    # Run model-based methods
    print("Running Value Iteration...")
    vi_V, vi_policy, vi_rewards = value_iteration(env)
    results['Value Iteration'] = vi_rewards + [vi_rewards[-1]] * (num_episodes - len(vi_rewards))
    policies['Value Iteration'] = vi_policy
    
    print("Running Policy Iteration...")
    pi_V, pi_policy, pi_rewards = policy_iteration(env)
    results['Policy Iteration'] = pi_rewards + [pi_rewards[-1]] * (num_episodes - len(pi_rewards))
    policies['Policy Iteration'] = pi_policy
    
    # Run Q-Learning (without parallel processing for now)
    print("Running Q-Learning...")
    q_agent = QLearningAgent(env.n_dims, env.action_space.n, learning_rate=0.2, epsilon=0.2)
    q_rewards = []
    for episode in range(num_episodes):
        reward = run_episode(env, q_agent, False)
        q_rewards.append(reward)
        if episode % 20 == 0:
            print(f"Q-Learning Episode {episode}, Reward: {reward:.3f}")
    results['Q-Learning'] = q_rewards
    policies['Q-Learning'] = q_agent.Q
    
    # Run SARSA (without parallel processing for now)
    print("Running SARSA...")
    sarsa_agent = SarsaAgent(env.n_dims, env.action_space.n, learning_rate=0.2, epsilon=0.2)
    sarsa_rewards = []
    for episode in range(num_episodes):
        reward = run_episode(env, sarsa_agent, True)
        sarsa_rewards.append(reward)
        if episode % 20 == 0:
            print(f"SARSA Episode {episode}, Reward: {reward:.3f}")
    results['SARSA'] = sarsa_rewards
    policies['SARSA'] = sarsa_agent.Q
    
    # Batch process visualizations
    plt.ioff()  # Turn off interactive mode for faster plotting
    
    # Create all visualizations
    visualize_state_visitation(env, vi_policy, "Value Iteration")
    env.reset()
    visualize_state_visitation(env, pi_policy, "Policy Iteration")
    env.reset()
    visualize_state_visitation(env, q_agent.Q, "Q-Learning")
    env.reset()
    visualize_state_visitation(env, sarsa_agent.Q, "SARSA")
    
    # Visualize optimal paths
    visualize_optimal_paths(env, policies)
    
    # Plot learning curves
    plot_learning_curves(results, dims)
    
    plt.ion()  # Turn interactive mode back on
    
    # Print convergence information
    print("\nConvergence Information:")
    print(f"Value Iteration converged in {len(vi_rewards)} iterations")
    print(f"Policy Iteration converged in {len(pi_rewards)} iterations")
    
    return results

class DynamicMultiDimGrid(gym.Env):
    """Multi-dimensional grid world with dynamic obstacles and changing rewards."""
    
    def __init__(self, dims=(4, 4, 4), movement_cost=-0.01, n_obstacles=None, 
                 obstacle_move_prob=0.1, reward_change_prob=0.05):
        super(DynamicMultiDimGrid, self).__init__()
        
        self.dims = dims
        self.n_dims = len(dims)
        self.movement_cost = movement_cost
        self.obstacle_move_prob = obstacle_move_prob
        self.reward_change_prob = reward_change_prob
        
        # Action space: 2 actions per dimension (positive and negative movement)
        self.action_space = spaces.Discrete(2 * self.n_dims)
        
        # Observation space: tuple of coordinates
        self.observation_space = spaces.MultiDiscrete(dims)
        
        # Start state: (0, 0, ..., 0)
        self.start_state = tuple([0] * self.n_dims)
        
        # Goal state: (max, max, ..., max)
        self.goal_state = tuple([d-1 for d in dims])
        
        # Generate initial obstacles
        if n_obstacles is None:
            total_states = np.prod(dims)
            n_obstacles = int(0.1 * total_states)
        self.n_obstacles = n_obstacles
        
        # Initialize obstacles and rewards
        self.reset()
    
    def _generate_obstacles(self):
        """Generate random obstacle positions, avoiding start and goal states."""
        obstacles = set()
        all_states = list(itertools.product(*[range(d) for d in self.dims]))
        valid_states = [s for s in all_states if s != self.start_state and s != self.goal_state]
        
        if self.n_obstacles > len(valid_states):
            self.n_obstacles = len(valid_states)
        
        obstacle_states = random.sample(valid_states, self.n_obstacles)
        obstacles.update(obstacle_states)
        
        return obstacles
    
    def _move_obstacles(self):
        """Randomly move obstacles to adjacent cells."""
        new_obstacles = set()
        for obs in self.obstacles:
            if random.random() < self.obstacle_move_prob:
                # Try to move obstacle to adjacent cell
                dim = random.randint(0, self.n_dims - 1)
                direction = random.choice([-1, 1])
                new_pos = list(obs)
                new_pos[dim] = max(0, min(self.dims[dim] - 1, new_pos[dim] + direction))
                new_pos = tuple(new_pos)
                
                # Only move if not blocking start or goal
                if new_pos != self.start_state and new_pos != self.goal_state:
                    new_obstacles.add(new_pos)
                else:
                    new_obstacles.add(obs)
            else:
                new_obstacles.add(obs)
        
        self.obstacles = new_obstacles
    
    def _update_rewards(self):
        """Randomly change rewards in some states."""
        if random.random() < self.reward_change_prob:
            self.movement_cost = random.uniform(-0.02, 0)
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.obstacles = self._generate_obstacles()
        self.current_state = self.start_state
        self.visited_states = defaultdict(int)
        self.visited_states[self.current_state] += 1
        return self.current_state, {}
    
    def step(self, action):
        # Move obstacles and update rewards
        self._move_obstacles()
        self._update_rewards()
        
        # Convert action to dimension and direction
        dim = action // 2
        direction = 1 if action % 2 == 0 else -1
        
        # Create list from current state for modification
        new_state = list(self.current_state)
        
        # Apply action
        new_state[dim] = min(max(0, new_state[dim] + direction), self.dims[dim] - 1)
        new_state = tuple(new_state)
        
        # Check if hit obstacle
        if new_state in self.obstacles:
            new_state = self.current_state  # Bounce back
            reward = -0.5  # Penalty for hitting obstacle
        else:
            # Check if reached goal
            done = new_state == self.goal_state
            reward = 1.0 if done else self.movement_cost
        
        self.current_state = new_state
        self.visited_states[self.current_state] += 1
        
        done = self.current_state == self.goal_state
        return new_state, reward, done, False, {}

def create_interactive_path_visualization(env, policies, title, is_dynamic=False):
    """Create interactive 3D plot of optimal paths using Plotly."""
    fig = go.Figure()
    
    # Define colors for each algorithm
    colors = {
        'Value Iteration': 'blue',
        'Policy Iteration': 'green',
        'Q-Learning': 'red',
        'SARSA': 'purple'
    }
    
    # Plot start and goal points
    fig.add_trace(go.Scatter3d(
        x=[env.start_state[0]], y=[env.start_state[1]], z=[env.start_state[2]],
        mode='markers',
        marker=dict(size=15, color='lime', symbol='diamond'),
        name='Start'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[env.goal_state[0]], y=[env.goal_state[1]], z=[env.goal_state[2]],
        mode='markers',
        marker=dict(size=15, color='gold', symbol='diamond'),
        name='Goal'
    ))
    
    # Plot obstacles
    if env.obstacles:
        obstacles = np.array(list(env.obstacles))
        fig.add_trace(go.Scatter3d(
            x=obstacles[:, 0], y=obstacles[:, 1], z=obstacles[:, 2],
            mode='markers',
            marker=dict(size=8, color='black', symbol='x'),
            name='Obstacles'
        ))
    
    # Plot paths for each algorithm
    for algo_name, policy in policies.items():
        path = []
        state = env.start_state
        visited = set()
        
        while state != env.goal_state and state not in visited and len(path) < 100:
            path.append(state)
            visited.add(state)
            
            if algo_name in ['Q-Learning', 'SARSA']:
                # Initialize Q-values if state not in Q-table
                if state not in policy:
                    policy[state] = np.zeros(env.action_space.n)
                action = np.argmax(policy[state])
            else:
                action = policy[state]
            
            env.current_state = state
            state, _, _, _, _ = env.step(action)
        
        path.append(env.goal_state)
        path = np.array(path)
        
        fig.add_trace(go.Scatter3d(
            x=path[:, 0], y=path[:, 1], z=path[:, 2],
            mode='lines+markers',
            line=dict(color=colors[algo_name], width=4),
            marker=dict(size=4, symbol='circle'),
            name=f'{algo_name} Path'
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1000,
        height=800,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Save as HTML for interactivity
    fig.write_html(f'{"dynamic_" if is_dynamic else ""}optimal_paths_3d.html')

def create_performance_dashboard(results, is_dynamic=False):
    """Create interactive dashboard comparing algorithm performance metrics."""
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Learning Curves',
            'Performance Distribution',
            'Convergence Rate',
            'Cumulative Rewards'
        )
    )
    
    # Colors for consistency
    colors = {
        'Value Iteration': 'blue',
        'Policy Iteration': 'green',
        'Q-Learning': 'red',
        'SARSA': 'purple'
    }
    
    # 1. Learning Curves (Moving Average)
    window_size = 10
    for algo, rewards in results.items():
        avg_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        fig.add_trace(
            go.Scatter(x=list(range(len(avg_rewards))), y=avg_rewards,
                      name=f'{algo} (MA)', line=dict(color=colors[algo])),
            row=1, col=1
        )
    
    # 2. Performance Distribution (Violin Plot)
    for i, (algo, rewards) in enumerate(results.items()):
        fig.add_trace(
            go.Violin(y=rewards, name=algo, line_color=colors[algo],
                     side='positive', meanline_visible=True),
            row=1, col=2
        )
    
    # 3. Convergence Rate
    optimal_reward = max([max(results[a]) for a in results.keys()])
    for algo, rewards in results.items():
        convergence = [max(rewards[:i+1])/optimal_reward for i in range(len(rewards))]
        fig.add_trace(
            go.Scatter(x=list(range(len(convergence))), y=convergence,
                      name=f'{algo} Conv', line=dict(color=colors[algo])),
            row=2, col=1
        )
    
    # 4. Cumulative Rewards
    for algo, rewards in results.items():
        cumulative = np.cumsum(rewards)
        fig.add_trace(
            go.Scatter(x=list(range(len(cumulative))), y=cumulative,
                      name=f'{algo} Cum', line=dict(color=colors[algo])),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=1000,
        width=1200,
        title_text=f"{'Dynamic ' if is_dynamic else ''}Environment Performance Metrics",
        showlegend=True
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Episode", row=1, col=1)
    fig.update_xaxes(title_text="Episode", row=2, col=1)
    fig.update_xaxes(title_text="Episode", row=2, col=2)
    fig.update_yaxes(title_text="Average Reward", row=1, col=1)
    fig.update_yaxes(title_text="Reward Distribution", row=1, col=2)
    fig.update_yaxes(title_text="Convergence Rate", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative Reward", row=2, col=2)
    
    # Save as HTML for interactivity
    fig.write_html(f'{"dynamic_" if is_dynamic else ""}performance_dashboard.html')

def compare_dynamic_environment(dims=(4, 4, 4), num_episodes=50):
    """Optimized dynamic environment comparison."""
    env = DynamicMultiDimGrid(dims=dims, obstacle_move_prob=0.15, reward_change_prob=0.1)
    results = {
        'Value Iteration': [],
        'Policy Iteration': [],
        'Q-Learning': [],
        'SARSA': []
    }
    policies = {}
    
    # Initialize agents once
    q_agent = QLearningAgent(env.n_dims, env.action_space.n, learning_rate=0.2, epsilon=0.2)
    sarsa_agent = SarsaAgent(env.n_dims, env.action_space.n, learning_rate=0.2, epsilon=0.2)
    
    # Initialize state space
    state_space = list(itertools.product(*[range(d) for d in dims]))
    
    # Run episodes for each algorithm
    for episode in range(num_episodes):
        # Value Iteration (recompute every 5 episodes instead of 10)
        if episode % 5 == 0:
            vi_V, vi_policy = value_iteration(env, max_iterations=30)[:2]
            policies['Value Iteration'] = vi_policy
            # Initialize policy for all states
            for state in state_space:
                if state not in vi_policy:
                    vi_policy[state] = 0  # Default action
        
        state = env.reset()[0]
        vi_reward = 0
        done = False
        while not done and vi_reward > -10:  # Early stopping for very poor performance
            action = vi_policy.get(state, 0)  # Use default action if state not in policy
            state, reward, done, _, _ = env.step(action)
            vi_reward += reward
        results['Value Iteration'].append(vi_reward)
        
        # Policy Iteration (recompute every 5 episodes)
        if episode % 5 == 0:
            pi_V, pi_policy = policy_iteration(env, max_iterations=15)[:2]
            policies['Policy Iteration'] = pi_policy
            # Initialize policy for all states
            for state in state_space:
                if state not in pi_policy:
                    pi_policy[state] = 0  # Default action
        
        state = env.reset()[0]
        pi_reward = 0
        done = False
        while not done and pi_reward > -10:
            action = pi_policy.get(state, 0)  # Use default action if state not in policy
            state, reward, done, _, _ = env.step(action)
            pi_reward += reward
        results['Policy Iteration'].append(pi_reward)
        
        # Q-Learning
        q_reward = run_episode(env, q_agent, False)
        results['Q-Learning'].append(q_reward)
        policies['Q-Learning'] = q_agent.Q
        
        # SARSA
        sarsa_reward = run_episode(env, sarsa_agent, True)
        results['SARSA'].append(sarsa_reward)
        policies['SARSA'] = sarsa_agent.Q
        
        if episode % 5 == 0:  # More frequent updates
            print(f"\nEpisode {episode}")
            for algo, rewards in results.items():
                avg_reward = np.mean(rewards[-5:] if len(rewards) >= 5 else rewards)
                print(f"{algo}: Average Reward = {avg_reward:.3f}")
    
    # Create interactive visualizations
    create_interactive_path_visualization(env, policies, 
                                       "Dynamic Environment Optimal Paths", 
                                       is_dynamic=True)
    create_performance_dashboard(results, is_dynamic=True)
    
    return results

if __name__ == "__main__":
    # Test with 4x4x4 grid
    print("\nTesting Standard 4x4x4 Grid...")
    results_3d = compare_algorithms(dims=(4, 4, 4), num_episodes=100)
    
    # Create interactive visualizations for standard environment
    env = MultiDimGrid(dims=(4, 4, 4))
    
    # Initialize Q-Learning and SARSA agents with adjusted parameters
    q_agent = QLearningAgent(env.n_dims, env.action_space.n, learning_rate=0.2, epsilon=0.2)
    sarsa_agent = SarsaAgent(env.n_dims, env.action_space.n, learning_rate=0.15, epsilon=0.15)  # Slightly more conservative parameters
    
    print("\nTraining Q-Learning and SARSA for visualization...")
    # Train agents for more episodes to find optimal paths
    best_q_reward = float('-inf')
    best_sarsa_reward = float('-inf')
    no_improvement_count = 0
    
    for episode in range(300):  # Increased from 100 to 300
        q_reward = run_episode(env, q_agent, False)
        sarsa_reward = run_episode(env, sarsa_agent, True)
        
        # Track best performance
        best_q_reward = max(best_q_reward, q_reward)
        best_sarsa_reward = max(best_sarsa_reward, sarsa_reward)
        
        # Gradually reduce exploration as agents learn
        if episode > 0 and episode % 50 == 0:
            q_agent.epsilon *= 0.9
            sarsa_agent.epsilon *= 0.9
            q_agent.lr *= 0.95  # Using lr instead of learning_rate
            sarsa_agent.lr *= 0.95  # Using lr instead of learning_rate
        
        if episode % 20 == 0:
            print(f"Episode {episode}: Q-Learning reward = {q_reward:.3f} (best: {best_q_reward:.3f}), "
                  f"SARSA reward = {sarsa_reward:.3f} (best: {best_sarsa_reward:.3f})")
            print(f"Current exploration rates - Q: {q_agent.epsilon:.3f}, SARSA: {sarsa_agent.epsilon:.3f}")
            print(f"Current learning rates - Q: {q_agent.lr:.3f}, SARSA: {sarsa_agent.lr:.3f}")
    
    # Set exploration to 0 for visualization
    q_agent.epsilon = 0.0
    sarsa_agent.epsilon = 0.0
    
    # Run a few episodes with zero exploration to ensure we get the best paths
    print("\nFinal evaluation with no exploration:")
    for _ in range(5):
        q_reward = run_episode(env, q_agent, False)
        sarsa_reward = run_episode(env, sarsa_agent, True)
        print(f"Q-Learning reward = {q_reward:.3f}, SARSA reward = {sarsa_reward:.3f}")
    
    policies = {
        'Value Iteration': value_iteration(env)[1],
        'Policy Iteration': policy_iteration(env)[1],
        'Q-Learning': q_agent.Q,
        'SARSA': sarsa_agent.Q
    }
    create_interactive_path_visualization(env, policies, 
                                       "Standard Environment Optimal Paths")
    create_performance_dashboard(results_3d)
    
    print("\nTesting Dynamic Environment...")
    dynamic_results = compare_dynamic_environment(dims=(4, 4, 4), num_episodes=50) 