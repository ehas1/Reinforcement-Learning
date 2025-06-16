import numpy as np
import plotly.graph_objects as go
import json
from environments import StochasticObstacle3DWorld
from n_step_algorithms import NStepQLearning, NStepSARSA, NStepExpectedSARSA

def run_experiment(env, agent, n_episodes=100, max_steps=200):
    """Run a single experiment with an agent."""
    episode_rewards = []
    success_rates = []
    value_history = []
    best_path = None
    best_path_reward = float('-inf')
    
    successes = 0
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        path = [state]
        
        for step in range(max_steps):
            action = agent._get_epsilon_greedy_action(state)
            next_state, reward, done, _, _ = env.step(action)
            
            agent.update(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state
            path.append(state)
            
            if done:
                if reward > 0:  # Reached goal
                    successes += 1
                    if episode_reward > best_path_reward:
                        best_path_reward = episode_reward
                        best_path = path
                break
        
        episode_rewards.append(episode_reward)
        success_rates.append(successes / (episode + 1))
        
        # Store average value function
        value_history.append(np.mean(np.max(agent.Q, axis=1)))
    
    return {
        'rewards': episode_rewards,
        'success_rates': success_rates,
        'value_history': value_history,
        'paths': [best_path] if best_path is not None else []  # Only keep the best path
    }

def create_path_traces(paths, color):
    """Create path visualization traces."""
    traces = []
    
    for path in paths:
        x = [p[0] for p in path]
        y = [p[1] for p in path]
        z = [p[2] for p in path]
        
        traces.append(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='lines+markers',
                line=dict(color=color, width=4),
                marker=dict(size=4),
                showlegend=False
            )
        )
    
    return traces

def create_value_function_trace(agent, dims):
    """Create value function visualization trace."""
    x, y, z = np.meshgrid(
        np.arange(dims[0]),
        np.arange(dims[1]),
        np.arange(dims[2])
    )
    
    values = np.max(agent.Q, axis=1).reshape(dims)
    
    return [go.Volume(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=values.flatten(),
        isomin=np.min(values),
        isomax=np.max(values),
        opacity=0.1,
        surface_count=20,
        colorscale='Viridis'
    )]

def create_obstacle_movement_data(env, n_frames=50):
    """Create animation data for obstacle movement."""
    obstacle_positions = []
    
    # Store initial positions
    obstacle_positions.append([list(map(int, pos)) for pos in env.obstacles])
    
    # Generate movement frames
    for _ in range(n_frames - 1):
        env._move_obstacles()
        obstacle_positions.append([list(map(int, pos)) for pos in env.obstacles])
    
    return obstacle_positions

def main():
    # Environment setup
    env = StochasticObstacle3DWorld()
    n_states = env.dims[0] * env.dims[1] * env.dims[2]
    n_actions = 6
    
    # Algorithm parameters
    n_steps = [1, 3, 5]
    algorithms = {
        'qlearning': NStepQLearning,
        'sarsa': NStepSARSA,
        'expectedsarsa': NStepExpectedSARSA
    }
    
    # Colors for different n-step values
    colors = {1: 'red', 3: 'blue', 5: 'green'}
    
    # Store results
    all_results = {algo: {} for algo in algorithms.keys()}
    plot_data = {
        algo: {
            'path': [],
            'value': None,
            'obstacle_frames': None
        } for algo in algorithms.keys()
    }
    
    # Generate obstacle movement data
    obstacle_frames = create_obstacle_movement_data(env)
    
    # Run experiments
    for algo_name, algo_class in algorithms.items():
        for n in n_steps:
            print(f"Running {n}-step {algo_name}...")
            agent = algo_class(
                n_states=n_states,
                n_actions=n_actions,
                n_step=n,
                gamma=0.99,
                alpha=0.1,
                epsilon=0.1
            )
            
            results = run_experiment(env, agent)
            all_results[algo_name][n] = results
            
            # Add path traces
            plot_data[algo_name]['path'].extend(
                create_path_traces(results['paths'], colors[n])
            )
            
            # Store value function for the largest n-step
            if n == max(n_steps):
                plot_data[algo_name]['value'] = create_value_function_trace(agent, env.dims)
                plot_data[algo_name]['obstacle_frames'] = obstacle_frames
    
    # Create learning curves data
    learning_curves_data = []
    success_rate_data = []
    value_convergence_data = []
    
    for algo_name in algorithms.keys():
        for n in n_steps:
            results = all_results[algo_name][n]
            
            # Learning curves
            learning_curves_data.append(
                go.Scatter(
                    y=results['rewards'],
                    name=f'{algo_name} (n={n})',
                    line=dict(color=colors[n])
                )
            )
            
            # Success rates
            success_rate_data.append(
                go.Scatter(
                    y=results['success_rates'],
                    name=f'{algo_name} (n={n})',
                    line=dict(color=colors[n])
                )
            )
            
            # Value convergence
            value_convergence_data.append(
                go.Scatter(
                    y=results['value_history'],
                    name=f'{algo_name} (n={n})',
                    line=dict(color=colors[n])
                )
            )
    
    # Add environment elements to path plots
    for algo_name in algorithms.keys():
        # Add goal state
        plot_data[algo_name]['path'].append(
            go.Scatter3d(
                x=[env.goal_state[0]],
                y=[env.goal_state[1]],
                z=[env.goal_state[2]],
                mode='markers',
                marker=dict(size=15, color='purple', symbol='diamond'),
                name='Goal'
            )
        )
        
        # Add obstacles
        if hasattr(env, 'obstacles'):
            obstacles = np.array(list(env.obstacles))
            plot_data[algo_name]['path'].append(
                go.Scatter3d(
                    x=obstacles[:, 0],
                    y=obstacles[:, 1],
                    z=obstacles[:, 2],
                    mode='markers',
                    marker=dict(size=8, color='orange', symbol='x'),
                    name='Obstacles'
                )
            )
    
    # Combine all visualization data
    visualization_data = {
        'qlearning': {
            'path': plot_data['qlearning']['path'],
            'value': plot_data['qlearning']['value'],
            'obstacle_frames': plot_data['qlearning']['obstacle_frames']
        },
        'sarsa': {
            'path': plot_data['sarsa']['path'],
            'value': plot_data['sarsa']['value'],
            'obstacle_frames': plot_data['sarsa']['obstacle_frames']
        },
        'expectedsarsa': {
            'path': plot_data['expectedsarsa']['path'],
            'value': plot_data['expectedsarsa']['value'],
            'obstacle_frames': plot_data['expectedsarsa']['obstacle_frames']
        },
        'learningCurves': learning_curves_data,
        'successRate': success_rate_data,
        'valueConvergence': value_convergence_data
    }
    
    # Save visualization data
    with open('visualization_data.js', 'w') as f:
        f.write('const plotData = ' + json.dumps(visualization_data, cls=PlotlyJSONEncoder) + ';')
    
    print("Visualization data generated successfully!")

class PlotlyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy and Plotly types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (go.Scatter, go.Scatter3d, go.Volume)):
            return obj.to_plotly_json()
        return json.JSONEncoder.default(self, obj)

if __name__ == "__main__":
    main() 