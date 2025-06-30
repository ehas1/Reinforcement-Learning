import numpy as np
import json
from environment_10x10x10 import GridWorld3D_10x10x10
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class NStepQLearning:
    def __init__(self, state_size, action_size, n_step=1, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.q_table = np.zeros((state_size, action_size))
        self.n_step = n_step
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []

    def get_action(self, state_idx):
        if np.random.random() < self.epsilon:
            return np.random.randint(6)
        return np.argmax(self.q_table[state_idx])

    def state_to_idx(self, state):
        return state[0] * 100 + state[1] * 10 + state[2]

    def update(self, state, action, reward, next_state, done):
        state_idx = self.state_to_idx(state)
        next_state_idx = self.state_to_idx(next_state)

        self.state_buffer.append(state_idx)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)

        if len(self.state_buffer) < self.n_step and not done:
            return

        if done:
            while self.state_buffer:
                self._update_q_value()
        elif len(self.state_buffer) == self.n_step:
            self._update_q_value()

    def _update_q_value(self):
        state_idx = self.state_buffer.pop(0)
        action = self.action_buffer.pop(0)
        
        G = 0
        for i, r in enumerate(self.reward_buffer):
            G += self.gamma ** i * r

        if len(self.state_buffer) > 0:
            next_state_idx = self.state_buffer[-1]
            G += self.gamma ** len(self.reward_buffer) * np.max(self.q_table[next_state_idx])

        self.q_table[state_idx][action] += self.alpha * (G - self.q_table[state_idx][action])
        self.reward_buffer.pop(0)

class NStepSARSA:
    def __init__(self, state_size, action_size, n_step=1, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.q_table = np.zeros((state_size, action_size))
        self.n_step = n_step
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []

    def get_action(self, state_idx):
        if np.random.random() < self.epsilon:
            return np.random.randint(6)
        return np.argmax(self.q_table[state_idx])

    def state_to_idx(self, state):
        return state[0] * 100 + state[1] * 10 + state[2]

    def update(self, state, action, reward, next_state, next_action, done):
        state_idx = self.state_to_idx(state)
        next_state_idx = self.state_to_idx(next_state)

        self.state_buffer.append(state_idx)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)

        if len(self.state_buffer) < self.n_step and not done:
            return

        if done:
            while self.state_buffer:
                self._update_q_value()
        elif len(self.state_buffer) == self.n_step:
            self._update_q_value()

    def _update_q_value(self):
        state_idx = self.state_buffer.pop(0)
        action = self.action_buffer.pop(0)
        
        G = 0
        for i, r in enumerate(self.reward_buffer):
            G += self.gamma ** i * r

        if len(self.state_buffer) > 0:
            next_state_idx = self.state_buffer[-1]
            next_action = self.action_buffer[-1]
            G += self.gamma ** len(self.reward_buffer) * self.q_table[next_state_idx][next_action]

        self.q_table[state_idx][action] += self.alpha * (G - self.q_table[state_idx][action])
        self.reward_buffer.pop(0)

def run_episode(env, agent, n_step):
    state = env.reset()
    done = False
    total_reward = 0
    path = [state.copy()]
    obstacle_frames = []

    if isinstance(agent, NStepQLearning):
        while not done:
            state_idx = agent.state_to_idx(state)
            action = agent.get_action(state_idx)
            next_state, reward, done = env.step(action)
            
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            path.append(state.copy())
            obstacle_frames.append(env.get_obstacle_positions())
    else:  # SARSA
        state_idx = agent.state_to_idx(state)
        action = agent.get_action(state_idx)
        
        while not done:
            next_state, reward, done = env.step(action)
            next_state_idx = agent.state_to_idx(next_state)
            next_action = agent.get_action(next_state_idx)
            
            agent.update(state, action, reward, next_state, next_action, done)
            
            state = next_state
            action = next_action
            total_reward += reward
            path.append(state.copy())
            obstacle_frames.append(env.get_obstacle_positions())

    return total_reward, path, obstacle_frames

def run_experiments():
    env = GridWorld3D_10x10x10()
    n_steps = [1, 3, 5]
    n_episodes = 200
    
    results = {
        'qlearning': {
            'rewards': {}, 
            'paths': {}, 
            'obstacle_frames': {},
            'n1': None,  # Will store Q-learning agent for n=1
            'n3': None,  # Will store Q-learning agent for n=3
            'n5': None   # Will store Q-learning agent for n=5
        },
        'sarsa': {
            'rewards': {}, 
            'paths': {}, 
            'obstacle_frames': {},
            'n1': None,  # Will store SARSA agent for n=1
            'n3': None,  # Will store SARSA agent for n=3
            'n5': None   # Will store SARSA agent for n=5
        }
    }
    
    for n in n_steps:
        print(f"Running {n}-step Q-learning...")
        q_agent = NStepQLearning(1000, 6, n_step=n)
        q_rewards = []
        
        for episode in range(n_episodes):
            reward, path, frames = run_episode(env, q_agent, n)
            q_rewards.append(reward)
            if episode == n_episodes - 1:  # Save only the last episode's path
                results['qlearning']['paths'][f'n{n}'] = path
                results['qlearning']['obstacle_frames'][f'n{n}'] = frames
        
        results['qlearning']['rewards'][f'n{n}'] = q_rewards
        results['qlearning'][f'n{n}'] = q_agent  # Store the trained agent
        
        print(f"Running {n}-step SARSA...")
        sarsa_agent = NStepSARSA(1000, 6, n_step=n)
        sarsa_rewards = []
        
        for episode in range(n_episodes):
            reward, path, frames = run_episode(env, sarsa_agent, n)
            sarsa_rewards.append(reward)
            if episode == n_episodes - 1:  # Save only the last episode's path
                results['sarsa']['paths'][f'n{n}'] = path
                results['sarsa']['obstacle_frames'][f'n{n}'] = frames
        
        results['sarsa']['rewards'][f'n{n}'] = sarsa_rewards
        results['sarsa'][f'n{n}'] = sarsa_agent  # Store the trained agent

    return results

def plot_learning_curves(results):
    plt.figure(figsize=(15, 8))
    
    # Style settings
    colors = {
        'n1': '#FF6B6B',  # Coral red
        'n3': '#4ECDC4',  # Turquoise
        'n5': '#45B7D1'   # Sky blue
    }
    algo_styles = {
        'qlearning': {'linestyle': '-', 'alpha': 0.8},
        'sarsa': {'linestyle': '--', 'alpha': 0.8}
    }
    
    # Set background color
    plt.gca().set_facecolor('#f8f9fa')
    plt.gcf().set_facecolor('white')
    
    # Calculate moving averages for smoother curves
    window = 20
    for algo in ['qlearning', 'sarsa']:
        for n in ['n1', 'n3', 'n5']:
            rewards = results[algo]['rewards'][n]
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            
            # Plot with enhanced styling
            plt.plot(moving_avg, 
                    label=f'{algo.upper()} (n={n[-1]})',
                    color=colors[n],
                    linestyle=algo_styles[algo]['linestyle'],
                    alpha=algo_styles[algo]['alpha'],
                    linewidth=2)
    
    # Enhance the plot appearance
    plt.xlabel('Episode', fontsize=12, fontweight='bold')
    plt.ylabel('Average Reward (Moving Window)', fontsize=12, fontweight='bold')
    plt.title('Learning Curves - 10x10x10 Grid World\n(20-Episode Moving Average)', 
             fontsize=14, fontweight='bold', pad=20)
    
    # Customize grid
    plt.grid(True, linestyle='--', alpha=0.7, color='gray')
    
    # Customize legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
              borderaxespad=0., frameon=True, 
              fancybox=True, shadow=True)
    
    # Add text box with environment details
    info_text = 'Environment Details:\n' + \
                '• Grid Size: 10x10x10\n' + \
                '• Moving Obstacles: 5\n' + \
                '• Max Steps: 500\n' + \
                '• Goal Reward: +100\n' + \
                '• Collision Penalty: -10\n' + \
                '• Step Penalty: -1'
    
    plt.text(1.25, 0.3, info_text,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='center')
    
    # Adjust layout to prevent text cutoff
    plt.tight_layout()
    
    # Save with high quality
    plt.savefig('learning_curves_10x10x10.png', 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.close()

def plot_value_convergence(results):
    plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2)
    
    # Plot 1: Average Q-values over time
    plt.subplot(gs[0, 0])
    for n in ['n1', 'n3', 'n5']:
        avg_q_values = []
        for episode in range(len(results['qlearning']['rewards'][n])):
            if episode % 10 == 0:  # Sample every 10 episodes to reduce noise
                avg_q_values.append(np.mean([np.max(q) for q in results['qlearning'][n].q_table]))
        plt.plot(range(0, len(results['qlearning']['rewards'][n]), 10), avg_q_values, 
                label=f'Q-Learning (n={n[-1]})')
    plt.xlabel('Episode')
    plt.ylabel('Average Max Q-Value')
    plt.title('Q-Value Convergence (Q-Learning)')
    plt.legend()
    plt.grid(True)

    # Plot 2: Moving average of rewards
    plt.subplot(gs[0, 1])
    window = 20
    for algo in ['qlearning', 'sarsa']:
        for n in ['n1', 'n3', 'n5']:
            rewards = results[algo]['rewards'][n]
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(moving_avg, 
                    label=f'{algo.upper()} (n={n[-1]})',
                    linestyle='-' if algo == 'qlearning' else '--')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (Moving Window)')
    plt.title(f'Moving Average Reward (Window={window})')
    plt.legend()
    plt.grid(True)

    # Plot 3: Success rate over time
    plt.subplot(gs[1, 0])
    window = 50
    for algo in ['qlearning', 'sarsa']:
        for n in ['n1', 'n3', 'n5']:
            rewards = results[algo]['rewards'][n]
            # Consider episode successful if reward > 0 (reached goal)
            successes = [1 if r > 0 else 0 for r in rewards]
            success_rate = np.convolve(successes, np.ones(window)/window, mode='valid')
            plt.plot(success_rate, 
                    label=f'{algo.upper()} (n={n[-1]})',
                    linestyle='-' if algo == 'qlearning' else '--')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.title(f'Success Rate (Moving Window={window})')
    plt.legend()
    plt.grid(True)

    # Plot 4: Steps to goal distribution
    plt.subplot(gs[1, 1])
    data = []
    labels = []
    for algo in ['qlearning', 'sarsa']:
        for n in ['n1', 'n3', 'n5']:
            path = results[algo]['paths'][n]
            steps = len(path)
            data.append(steps)
            labels.append(f'{algo.upper()} (n={n[-1]})')
    
    plt.bar(labels, data)
    plt.xticks(rotation=45)
    plt.ylabel('Steps to Goal')
    plt.title('Final Path Length Comparison')
    plt.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig('convergence_analysis_10x10x10.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_3d_visualization(results):
    # Create the visualization data structure
    vis_data = {
        'qlearning': {
            'path': [],
            'obstacle_frames': [[list(map(int, obs)) for obs in frame] for frame in results['qlearning']['obstacle_frames']['n3']]  # Convert numpy arrays to lists
        },
        'sarsa': {
            'path': [],
            'obstacle_frames': [[list(map(int, obs)) for obs in frame] for frame in results['sarsa']['obstacle_frames']['n3']]  # Convert numpy arrays to lists
        }
    }

    # Add paths for each n-step value
    colors = {'n1': 'red', 'n3': 'blue', 'n5': 'green'}
    for algo in ['qlearning', 'sarsa']:
        for n in ['n1', 'n3', 'n5']:
            path = results[algo]['paths'][n]
            trace = {
                'type': 'scatter3d',
                'x': [int(p[0]) for p in path],  # Convert numpy values to Python ints
                'y': [int(p[1]) for p in path],
                'z': [int(p[2]) for p in path],
                'mode': 'lines+markers',
                'name': f'{n}',
                'line': {'color': colors[n], 'width': 4},
                'marker': {'size': 4},
                'showlegend': True
            }
            vis_data[algo]['path'].append(trace)

        # Add goal point
        vis_data[algo]['path'].append({
            'type': 'scatter3d',
            'x': [9],
            'y': [9],
            'z': [9],
            'mode': 'markers',
            'name': 'Goal',
            'marker': {'color': 'purple', 'size': 15, 'symbol': 'diamond'},
            'showlegend': True
        })

    # Save visualization data
    with open('visualization_data_10x10x10.js', 'w') as f:
        f.write('const plotData = ' + json.dumps(vis_data))

def main():
    results = run_experiments()
    plot_learning_curves(results)
    plot_value_convergence(results)
    plot_reward_distributions(results)
    generate_3d_visualization(results)

if __name__ == "__main__":
    main() 
