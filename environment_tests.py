import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from collections import defaultdict
import plotly.graph_objects as go
import json
from grid_world_comparison import QLearningAgent, SarsaAgent
from rl_algorithms import value_iteration_generic as value_iteration
from rl_algorithms import policy_iteration_generic as policy_iteration
from rl_algorithms import run_episode_generic as run_episode
from monte_carlo import MonteCarloES

def evaluate_policy(env, policy, max_steps=50):  # Reduced max steps
    """Evaluate a policy in the environment with a step limit."""
    state = env.reset()[0]
    total_reward = 0
    steps = 0
    done = False
    while not done and steps < max_steps:
        action = policy[state]
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        steps += 1
    return total_reward

class SparseRewardGrid(gym.Env):
    """Environment with very sparse rewards to test exploration strategies."""
    
    def __init__(self, dims=(3, 3), sparsity=0.7):
        super().__init__()
        self.dims = dims
        self.sparsity = sparsity
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete(dims)
        
        self.start_state = (0, 0)
        self.goal_state = (dims[0]-1, dims[1]-1)
        
        # Generate sparse reward locations
        self.reward_locations = self._generate_reward_locations()
    
    def _generate_reward_locations(self):
        locations = {}
        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                if (i, j) == self.goal_state:
                    locations[(i, j)] = 1.0
                elif random.random() > self.sparsity:
                    locations[(i, j)] = random.uniform(0.1, 0.3)
                else:
                    locations[(i, j)] = 0.0
        return locations
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_state = self.start_state
        return self.current_state, {}
    
    def step(self, action):
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
        reward = self.reward_locations[new_state]
        
        return new_state, reward, done, False, {}

class NonStationaryGrid(gym.Env):
    """Environment with changing dynamics over time."""
    
    def __init__(self, dims=(4, 4), change_interval=100):
        super().__init__()
        self.dims = dims
        self.change_interval = change_interval
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete(dims)
        
        self.start_state = (0, 0)
        self.goal_state = (dims[0]-1, dims[1]-1)
        
        self.episode_count = 0
        self.current_phase = 0
        
        # Different phases of environment dynamics
        self.phases = [
            {'slip_prob': 0.0, 'wind': (0, 0)},
            {'slip_prob': 0.2, 'wind': (0, 1)},
            {'slip_prob': 0.0, 'wind': (1, 0)},
            {'slip_prob': 0.3, 'wind': (-1, -1)}
        ]
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_state = self.start_state
        self.episode_count += 1
        
        # Update phase
        if self.episode_count % self.change_interval == 0:
            self.current_phase = (self.current_phase + 1) % len(self.phases)
        
        return self.current_state, {}
    
    def step(self, action):
        old_state = self.current_state
        phase = self.phases[self.current_phase]
        
        # Random slip
        if random.random() < phase['slip_prob']:
            action = random.randint(0, 3)
        
        # Base movement
        if action == 0:  # up
            new_state = (min(self.dims[0]-1, old_state[0]+1), old_state[1])
        elif action == 1:  # down
            new_state = (max(0, old_state[0]-1), old_state[1])
        elif action == 2:  # right
            new_state = (old_state[0], min(self.dims[1]-1, old_state[1]+1))
        else:  # left
            new_state = (old_state[0], max(0, old_state[1]-1))
        
        # Apply wind effect
        wind = phase['wind']
        new_state = (
            min(max(0, new_state[0] + wind[0]), self.dims[0]-1),
            min(max(0, new_state[1] + wind[1]), self.dims[1]-1)
        )
        
        self.current_state = new_state
        done = (new_state == self.goal_state)
        reward = 1.0 if done else -0.01
        
        return new_state, reward, done, False, {}

class LongHorizonGrid(gym.Env):
    """Environment with long action sequences and delayed rewards."""
    
    def __init__(self, dims=(3, 3), max_steps=30):
        super().__init__()
        self.dims = dims
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete(dims)
        
        self.start_state = (0, 0)
        self.goal_state = (dims[0]-1, dims[1]-1)
        
        # Generate checkpoints
        self.checkpoints = self._generate_checkpoints()
        self.visited_checkpoints = set()
    
    def _generate_checkpoints(self):
        checkpoints = {}
        num_checkpoints = 2  # Reduced number of checkpoints
        for i in range(num_checkpoints):
            x = (i + 1) * self.dims[0] // (num_checkpoints + 1)
            y = (i + 1) * self.dims[1] // (num_checkpoints + 1)
            checkpoints[(x, y)] = 0.3
        return checkpoints
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_state = self.start_state
        self.steps = 0
        self.visited_checkpoints = set()
        return self.current_state, {}
    
    def step(self, action):
        self.steps += 1
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
        done = (new_state == self.goal_state) or (self.steps >= self.max_steps)
        
        # Reward structure
        reward = -0.01  # Small negative reward for each step
        
        # Checkpoint rewards
        if new_state in self.checkpoints and new_state not in self.visited_checkpoints:
            reward += self.checkpoints[new_state]
            self.visited_checkpoints.add(new_state)
        
        # Goal reward
        if new_state == self.goal_state:
            reward += 1.0
        
        return new_state, reward, done, False, {}

class StochasticRewardGrid(gym.Env):
    """Grid world with stochastic rewards and delayed feedback."""
    
    def __init__(self, dims=(3, 3), reward_variance=0.2, delay_prob=0.1):
        super().__init__()
        self.dims = dims
        self.reward_variance = reward_variance
        self.delay_prob = delay_prob
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete(dims)
        
        self.start_state = (0, 0)
        self.goal_state = (dims[0]-1, dims[1]-1)
        
        # Delayed rewards storage
        self.delayed_rewards = []
        
        # Initialize reward distributions
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
            reward += sum(self.delayed_rewards)
            self.delayed_rewards = []
        
        return new_state, reward, done, False, {}

def save_plot_data(fig, filename):
    """Save plot data in a format suitable for the combined view."""
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj

    plot_json = {
        'data': [trace.to_plotly_json() for trace in fig.data],
        'layout': fig.layout.to_plotly_json()
    }
    plot_data = convert_numpy(plot_json)
    
    with open(f'{filename}.json', 'w') as f:
        json.dump(plot_data, f)

def visualize_environment(env, title, description, value_function=None):
    """Create an enhanced visualization of the environment."""
    env.reset()
    grid = np.zeros(env.dims)
    value_grid = np.zeros(env.dims) if value_function is None else value_function.reshape(env.dims)
    
    # Create subplots for different visualizations
    fig = go.Figure()
    
    # 1. State Types (as discrete heatmap)
    # Mark special states
    grid[env.start_state] = 1
    grid[env.goal_state] = 2
    
    # Mark special features
    if hasattr(env, 'reward_locations'):
        for pos, reward in env.reward_locations.items():
            if reward > 0 and pos != env.goal_state:
                grid[pos] = 3
    if hasattr(env, 'checkpoints'):
        for pos in env.checkpoints:
            grid[pos] = 4
    
    # Create custom text annotations
    text_matrix = []
    for i in range(env.dims[0]):
        row_text = []
        for j in range(env.dims[1]):
            cell_type = grid[i, j]
            base_text = f'({i},{j})'
            if cell_type == 1:
                cell_text = f'{base_text}<br>Start'
            elif cell_type == 2:
                cell_text = f'{base_text}<br>Goal'
            elif cell_type == 3:
                cell_text = f'{base_text}<br>Reward'
            elif cell_type == 4:
                cell_text = f'{base_text}<br>Checkpoint'
            else:
                cell_text = base_text
            row_text.append(cell_text)
        text_matrix.append(row_text)
    
    # State types heatmap with improved colors and annotations
    fig.add_trace(go.Heatmap(
        z=grid,
        text=text_matrix,
        texttemplate='%{text}',
        textfont={"size": 12},
        colorscale=[
            [0, 'white'],
            [0.2, '#90EE90'],  # Light green for start
            [0.4, '#FFB6C1'],  # Light red for goal
            [0.6, '#FFFFE0'],  # Light yellow for rewards
            [0.8, '#ADD8E6'],  # Light blue for checkpoints
        ],
        showscale=False,
        name='State Types'
    ))
    
    # 2. Value Function (as contour)
    if value_function is not None:
        fig.add_trace(go.Contour(
            z=value_grid,
            colorscale='Viridis',
            showscale=True,
            opacity=0.7,
            name='Value Function',
            contours=dict(
                start=np.min(value_grid),
                end=np.max(value_grid),
                size=(np.max(value_grid) - np.min(value_grid)) / 10,
                showlabels=True
            ),
            colorbar=dict(
                title=dict(
                    text='Value',
                    side='right'
                )
            )
        ))
    
    fig.update_layout(
        title={'text': f"{title}<br><sup>{description}</sup>",
               'y':0.95, 'x':0.5,
               'xanchor': 'center',
               'yanchor': 'top'},
        height=500,  # Even larger
        width=700,   # Even larger
        margin=dict(l=50, r=50, t=80, b=50),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Save plot data
    save_plot_data(fig, f'{title.lower().replace(" ", "_")}_viz')

def run_quick_comparison(env, num_episodes=30):
    """Run a quick comparison of different algorithms."""
    # Initialize agents
    q_learning = QLearningAgent(
        n_states=env.dims[0] * env.dims[1],
        n_actions=4,
        learning_rate=0.5,
        epsilon=0.4,
        gamma=0.99
    )
    
    sarsa = SarsaAgent(
        n_states=env.dims[0] * env.dims[1],
        n_actions=4,
        learning_rate=0.5,
        epsilon=0.4,
        gamma=0.99
    )
    
    mces = MonteCarloES(
        n_states=env.dims[0] * env.dims[1],
        n_actions=4,
        gamma=0.99
    )
    
    # Run episodes
    results = {
        'Q-Learning': [],
        'SARSA': [],
        'Monte Carlo ES': []
    }
    
    # Training loop
    for episode in range(num_episodes):
        # Q-Learning episode
        state = env.reset()[0]
        total_reward = 0
        done = False
        while not done:
            action = q_learning.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            q_learning.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        results['Q-Learning'].append(total_reward)
        
        # SARSA episode
        state = env.reset()[0]
        total_reward = 0
        done = False
        action = sarsa.choose_action(state)
        while not done:
            next_state, reward, done, _, _ = env.step(action)
            next_action = sarsa.choose_action(next_state)
            sarsa.update(state, action, reward, next_state, next_action)
            state = next_state
            action = next_action
            total_reward += reward
        results['SARSA'].append(total_reward)
        
        # Monte Carlo ES episode
        episode_data = mces.generate_episode(env)
        mces.update(episode_data)
        total_reward = sum(reward for _, _, reward in episode_data)
        results['Monte Carlo ES'].append(total_reward)
    
    # Run Value Iteration for visualization only
    V = value_iteration(env, gamma=0.99)
    
    return results, V

def plot_quick_comparison(results, title, env_description):
    """Create enhanced comparison plots with detailed descriptions."""
    fig = go.Figure()
    
    colors = {
        'Q-Learning': '#FF9999',
        'SARSA': '#66B2FF',
        'Monte Carlo ES': '#99FF99'
    }
    
    # Learning Curves with Rolling Mean
    window = 5  # Larger window for smoother curves
    for method in ['Q-Learning', 'SARSA', 'Monte Carlo ES']:
        rewards = np.array(results[method])
        # Calculate rolling statistics
        rolling_mean = np.convolve(rewards, np.ones(window)/window, mode='valid')
        rolling_std = np.array([np.std(rewards[max(0, i-window):i+1]) 
                              for i in range(window-1, len(rewards))])
        
        # Plot main line (rolling mean)
        fig.add_trace(go.Scatter(
            x=list(range(len(rolling_mean))),
            y=rolling_mean,
            name=f"{method}",
            line=dict(color=colors[method], width=2),
            legendgroup=method
        ))
        
        # Add confidence intervals
        fig.add_trace(go.Scatter(
            x=list(range(len(rolling_mean))) + list(range(len(rolling_mean)))[::-1],
            y=list(rolling_mean + rolling_std) + list(rolling_mean - rolling_std)[::-1],
            fill='toself',
            fillcolor=colors[method],
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False,
            legendgroup=method,
            opacity=0.2
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Episode",
        yaxis_title="Total Reward",
        height=400,  # Reduced size
        width=600,   # Reduced size
        margin=dict(l=50, r=50, t=50, b=50),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='LightGray'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='LightGray'
        )
    )
    
    # Save plot data
    save_plot_data(fig, f'{title.lower().replace(" ", "_")}_comparison')

def generate_combined_html():
    """Generate the combined HTML view with embedded plot data."""
    # Read all plot data
    plot_files = [
        'sparse_rewards_comparison.json',
        'long_horizon_comparison.json',
        'stochastic_rewards_comparison.json'
    ]
    
    plot_data = {}
    for filename in plot_files:
        with open(filename) as f:
            key = filename.replace('.json', '')
            plot_data[key] = json.load(f)
    
    # Generate HTML with embedded data
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Reinforcement Learning Algorithm Comparison</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 900px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
                line-height: 1.6;
            }
            .environment-section {
                background-color: white;
                border-radius: 8px;
                padding: 30px;
                margin-bottom: 40px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .plot-container {
                min-height: 400px;
                background-color: white;
                padding: 10px;
                border-radius: 4px;
                margin-top: 20px;
            }
            h1 {
                color: #2c3e50;
                text-align: center;
                margin-bottom: 30px;
                font-size: 28px;
            }
            h2 {
                color: #34495e;
                margin-top: 0;
                font-size: 24px;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }
            .description {
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 6px;
                margin-bottom: 20px;
                border-left: 4px solid #3498db;
            }
            .description h3 {
                color: #2c3e50;
                margin-top: 0;
                margin-bottom: 10px;
                font-size: 18px;
            }
            .description p {
                margin: 0 0 15px 0;
                color: #444;
            }
            .analysis {
                background-color: #fff3e0;
                padding: 20px;
                border-radius: 6px;
                margin-bottom: 20px;
                border-left: 4px solid #ff9800;
            }
            .analysis h3 {
                color: #e65100;
                margin-top: 0;
                margin-bottom: 10px;
                font-size: 18px;
            }
            .analysis p {
                margin: 0;
                color: #444;
            }
            .algorithm-comparison {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin-bottom: 20px;
            }
            .algorithm-card {
                flex: 1;
                min-width: 250px;
                background-color: #e3f2fd;
                padding: 15px;
                border-radius: 6px;
                border-left: 4px solid #1976d2;
            }
            .algorithm-card h4 {
                color: #1976d2;
                margin-top: 0;
                margin-bottom: 10px;
                font-size: 16px;
            }
            .algorithm-card p {
                margin: 0;
                color: #444;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <h1>Reinforcement Learning Algorithm Comparison</h1>
        
        <!-- Sparse Reward Environment -->
        <div class="environment-section">
            <h2>Sparse Reward Environment</h2>
            <div class="description">
                <h3>Environment Setup</h3>
                <p>A 3x3 grid world where 70% of states have zero reward. This environment is designed to test each algorithm's ability to explore efficiently and discover sparse rewards. The agent must navigate from a fixed start state to a goal state, but most intermediate states provide no feedback, making exploration crucial.</p>
            </div>
            <div class="analysis">
                <h3>Performance Analysis</h3>
                <p>Q-Learning typically excels in this environment due to its off-policy nature, which allows for more aggressive exploration while maintaining a greedy policy. SARSA tends to be more conservative in its exploration, while Monte Carlo ES can struggle with the lack of intermediate feedback.</p>
            </div>
            <div class="algorithm-comparison">
                <div class="algorithm-card">
                    <h4>Q-Learning</h4>
                    <p>Benefits from off-policy learning, allowing separate exploration and exploitation strategies. Can learn optimal paths even from exploratory actions.</p>
                </div>
                <div class="algorithm-card">
                    <h4>SARSA</h4>
                    <p>More conservative exploration due to on-policy learning. Takes longer to find optimal paths but generally more stable once found.</p>
                </div>
                <div class="algorithm-card">
                    <h4>Monte Carlo ES</h4>
                    <p>Uses complete episode returns, which can be sparse in this environment. Exploring starts help with initial exploration but delayed feedback is challenging.</p>
                </div>
            </div>
            <div class="plot-container" id="sparse-comparison"></div>
        </div>

        <!-- Long Horizon Environment -->
        <div class="environment-section">
            <h2>Long Horizon Environment</h2>
            <div class="description">
                <h3>Environment Setup</h3>
                <p>A 3x3 grid with strategically placed checkpoints that provide partial rewards. This environment tests each algorithm's ability to handle temporal credit assignment and learn from delayed rewards. The agent must learn which checkpoint sequence leads to the optimal path.</p>
            </div>
            <div class="analysis">
                <h3>Performance Analysis</h3>
                <p>Temporal difference methods (Q-Learning and SARSA) handle credit assignment effectively through bootstrapping. Monte Carlo methods face challenges with long-term dependencies but benefit from complete episode evaluation.</p>
            </div>
            <div class="algorithm-comparison">
                <div class="algorithm-card">
                    <h4>Q-Learning</h4>
                    <p>Effective at propagating rewards through bootstrapping. Can quickly learn the value of checkpoint sequences.</p>
                </div>
                <div class="algorithm-card">
                    <h4>SARSA</h4>
                    <p>Similar bootstrapping benefits to Q-Learning, but with more conservative policy updates. Good stability in learning checkpoint values.</p>
                </div>
                <div class="algorithm-card">
                    <h4>Monte Carlo ES</h4>
                    <p>Evaluates complete episodes, which can be beneficial for understanding the full checkpoint sequence but slower to learn.</p>
                </div>
            </div>
            <div class="plot-container" id="horizon-comparison"></div>
        </div>

        <!-- Stochastic Environment -->
        <div class="environment-section">
            <h2>Stochastic Environment</h2>
            <div class="description">
                <h3>Environment Setup</h3>
                <p>A 3x3 grid with noisy rewards (σ=0.2) and random delays (p=0.1). This environment tests each algorithm's robustness to uncertainty and ability to learn stable policies under stochastic conditions. The agent must learn to distinguish signal from noise.</p>
            </div>
            <div class="analysis">
                <h3>Performance Analysis</h3>
                <p>TD methods show strong performance due to their ability to average out noise through bootstrapping. Monte Carlo methods benefit from full episode evaluation but can be more sensitive to reward variance.</p>
            </div>
            <div class="algorithm-comparison">
                <div class="algorithm-card">
                    <h4>Q-Learning</h4>
                    <p>Robust to noise through TD learning. Can maintain good performance despite stochastic rewards and delays.</p>
                </div>
                <div class="algorithm-card">
                    <h4>SARSA</h4>
                    <p>Similar noise robustness to Q-Learning. On-policy nature can provide more stable learning in stochastic environments.</p>
                </div>
                <div class="algorithm-card">
                    <h4>Monte Carlo ES</h4>
                    <p>Full episode evaluation helps average out noise, but can be more sensitive to reward delays and variance.</p>
                </div>
            </div>
            <div class="plot-container" id="stochastic-comparison"></div>
        </div>

        <script>
            const plotData = ''' + json.dumps(plot_data) + ''';
            
            // Create all plots
            Plotly.newPlot('sparse-comparison', 
                plotData.sparse_rewards_comparison.data, 
                plotData.sparse_rewards_comparison.layout);
            Plotly.newPlot('horizon-comparison', 
                plotData.long_horizon_comparison.data, 
                plotData.long_horizon_comparison.layout);
            Plotly.newPlot('stochastic-comparison', 
                plotData.stochastic_rewards_comparison.data, 
                plotData.stochastic_rewards_comparison.layout);
        </script>
    </body>
    </html>
    '''
    
    with open('combined_results.html', 'w') as f:
        f.write(html_template)

if __name__ == "__main__":
    # Test environments with smaller dimensions
    print("\nTesting Sparse Reward Environment...")
    sparse_env = SparseRewardGrid(dims=(3, 3), sparsity=0.7)
    results, V = run_quick_comparison(sparse_env)
    plot_quick_comparison(
        results,
        "Sparse Rewards",
        "A 3x3 grid world where 70% of states have zero reward. Tests exploration efficiency and ability to discover sparse rewards. Start and goal states are fixed, but most intermediate states provide no feedback."
    )
    
    print("\nTesting Long Horizon Environment...")
    longhorizon_env = LongHorizonGrid(dims=(3, 3), max_steps=30)
    results, V = run_quick_comparison(longhorizon_env)
    plot_quick_comparison(
        results,
        "Long Horizon",
        "A 3x3 grid with checkpoints providing partial rewards. Tests temporal credit assignment and ability to learn from delayed rewards. Requires learning which checkpoints lead to the goal."
    )
    
    print("\nTesting Stochastic Reward Environment...")
    stochastic_env = StochasticRewardGrid(dims=(3, 3), reward_variance=0.2, delay_prob=0.1)
    results, V = run_quick_comparison(stochastic_env)
    plot_quick_comparison(
        results,
        "Stochastic Rewards",
        "A 3x3 grid with noisy rewards (σ=0.2) and random delays (p=0.1). Tests robustness to uncertainty and ability to learn stable policies under stochastic conditions."
    )
    
    # Generate combined view
    generate_combined_html() 