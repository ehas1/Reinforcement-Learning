import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from grid_world_comparison import QLearningAgent, SarsaAgent
from rl_algorithms import value_iteration_generic as value_iteration
from rl_algorithms import policy_iteration_generic as policy_iteration
from rl_algorithms import run_episode_generic as run_episode

class StochasticRewardGrid(gym.Env):
    
    def __init__(self, dims=(4, 4), reward_variance=0.5, delay_prob=0.2):
        super().__init__()
        self.dims = dims
        self.reward_variance = reward_variance
        self.delay_prob = delay_prob
        
        # Action space: up, down, left, right
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete(dims)
        
        # Start state and goal state
        self.start_state = (0, 0)
        self.goal_state = (dims[0]-1, dims[1]-1)
        
        # Delayed rewards storage
        self.delayed_rewards = []
        
        # Initialize reward distributions for each state
        self.reward_means = {}
        for i in range(dims[0]):
            for j in range(dims[1]):
                if (i, j) == self.goal_state:
                    self.reward_means[(i, j)] = 1.0
                else:
                    self.reward_means[(i, j)] = -0.01
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_state = self.start_state
        self.delayed_rewards = []
        return self.current_state, {}
    
    def step(self, action):
        # Store current state
        old_state = self.current_state
        
        # Movement dynamics
        if action == 0:  # up
            new_state = (min(self.dims[0]-1, old_state[0]+1), old_state[1])
        elif action == 1:  # down
            new_state = (max(0, old_state[0]-1), old_state[1])
        elif action == 2:  # right
            new_state = (old_state[0], min(self.dims[1]-1, old_state[1]+1))
        else:  # left
            new_state = (old_state[0], max(0, old_state[1]-1))
        
        self.current_state = new_state
        done = (new_state == self.goal_state)
        
        # Generate stochastic reward
        base_reward = self.reward_means[new_state]
        noise = np.random.normal(0, self.reward_variance)
        reward = base_reward + noise
        
        # Handle delayed rewards
        if random.random() < self.delay_prob and not done:
            self.delayed_rewards.append(reward)
            reward = 0
        elif self.delayed_rewards and (done or random.random() < 0.3):
            # Release some delayed rewards
            reward += sum(self.delayed_rewards)
            self.delayed_rewards = []
        
        return new_state, reward, done, False, {}

class MonteCarloAgent:
    """Monte Carlo agent with exploring starts."""
    
    def __init__(self, action_space, gamma=0.99, epsilon=0.1):
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = {}
        self.returns = defaultdict(list)
        self.policy = {}
    
    def get_action(self, state):
        if state not in self.policy or random.random() < self.epsilon:
            return self.action_space.sample()
        return self.policy[state]
    
    def update(self, episode):
        states, actions, rewards = zip(*episode)
        G = 0
        
        # Process episode in reverse order
        for t in range(len(states) - 1, -1, -1):
            G = rewards[t] + self.gamma * G
            state = states[t]
            action = actions[t]
            
            # First-visit MC
            if (state, action) not in [(states[i], actions[i]) for i in range(t)]:
                if state not in self.Q:
                    self.Q[state] = np.zeros(self.action_space.n)
                self.returns[(state, action)].append(G)
                self.Q[state][action] = np.mean(self.returns[(state, action)])
                self.policy[state] = np.argmax(self.Q[state])

def run_comparison(env, num_episodes=1000):
    # Initialize agents
    mc_agent = MonteCarloAgent(env.action_space)
    q_agent = QLearningAgent(env.action_space)
    sarsa_agent = SarsaAgent(env.action_space)
    
    results = {
        'Monte Carlo': [],
        'Q-Learning': [],
        'SARSA': [],
        'Value Iteration': [],
        'Policy Iteration': []
    }
    
    # Training loop
    for episode in range(num_episodes):
        # Monte Carlo episode
        state = env.reset()[0]
        mc_episode = []
        done = False
        while not done:
            action = mc_agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            mc_episode.append((state, action, reward))
            state = next_state
        mc_agent.update(mc_episode)
        results['Monte Carlo'].append(sum(r for _, _, r in mc_episode))
        
        # Q-Learning episode
        results['Q-Learning'].append(run_episode(env, q_agent, False))
        
        # SARSA episode
        results['SARSA'].append(run_episode(env, sarsa_agent, True))
        
        # Value/Policy Iteration (recompute periodically)
        if episode % 10 == 0:
            # Save environment state
            old_state = env.current_state
            
            # Run VI/PI
            vi_V, vi_policy = value_iteration(env)
            pi_V, pi_policy = policy_iteration(env)
            
            # Evaluate policies
            env.current_state = env.start_state
            vi_reward = evaluate_policy(env, vi_policy)
            
            env.current_state = env.start_state
            pi_reward = evaluate_policy(env, pi_policy)
            
            # Restore environment state
            env.current_state = old_state
            
            results['Value Iteration'].append(vi_reward)
            results['Policy Iteration'].append(pi_reward)
            
            # Fill in missing values
            if episode > 0:
                last_vi = results['Value Iteration'][-2]
                last_pi = results['Policy Iteration'][-2]
                for i in range(10):
                    if episode - 9 + i < 0:
                        continue
                    results['Value Iteration'].append(last_vi)
                    results['Policy Iteration'].append(last_pi)
    
    return results

def evaluate_policy(env, policy):
    """Evaluate a policy in the environment."""
    state = env.reset()[0]
    total_reward = 0
    done = False
    while not done:
        action = policy[state]
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
    return total_reward

def plot_comparison_results(results, title):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Learning Curves',
            'Final Performance Distribution',
            'Sample Efficiency',
            'Stability Analysis'
        )
    )
    
    colors = {
        'Monte Carlo': 'purple',
        'Q-Learning': 'red',
        'SARSA': 'blue',
        'Value Iteration': 'green',
        'Policy Iteration': 'orange'
    }
    
    # Learning Curves
    window = 20
    for method, rewards in results.items():
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        fig.add_trace(
            go.Scatter(x=list(range(len(smoothed))), y=smoothed,
                      name=method, line=dict(color=colors[method])),
            row=1, col=1
        )
    
    # Performance Distribution
    for method, rewards in results.items():
        fig.add_trace(
            go.Violin(y=rewards[-100:], name=method, line_color=colors[method],
                     side='positive', meanline_visible=True),
            row=1, col=2
        )
    
    # Sample Efficiency (episodes to reach 90% of max performance)
    max_rewards = {m: max(r) for m, r in results.items()}
    episodes_to_threshold = {}
    for method, rewards in results.items():
        threshold = 0.9 * max_rewards[method]
        for i, r in enumerate(rewards):
            if r >= threshold:
                episodes_to_threshold[method] = i
                break
    
    fig.add_trace(
        go.Bar(x=list(episodes_to_threshold.keys()),
               y=list(episodes_to_threshold.values()),
               marker_color=list(colors.values())),
        row=2, col=1
    )
    
    # Stability Analysis (rolling standard deviation)
    for method, rewards in results.items():
        std = [np.std(rewards[max(0, i-window):i+1])
               for i in range(len(rewards))]
        fig.add_trace(
            go.Scatter(x=list(range(len(std))), y=std,
                      name=f"{method} Stability",
                      line=dict(color=colors[method])),
            row=2, col=2
        )
    
    fig.update_layout(
        height=1000,
        width=1200,
        title_text=title,
        showlegend=True
    )
    
    fig.write_html(f"{title.lower().replace(' ', '_')}.html")

if __name__ == "__main__":
    # Test environments with different characteristics
    print("\nTesting Stochastic Rewards Environment...")
    stochastic_env = StochasticRewardGrid(reward_variance=0.5, delay_prob=0.2)
    stochastic_results = run_comparison(stochastic_env)
    plot_comparison_results(stochastic_results, "Stochastic Rewards Comparison") 
