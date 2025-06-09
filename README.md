# RL Policy Iteration Comparison

A comprehensive implementation and visualization of various reinforcement learning algorithms, focusing on policy iteration methods across different test environments. This project compares the performance of Q-Learning, SARSA, Monte Carlo ES, and Value Iteration algorithms in challenging grid world environments.

## Features

- **Multiple RL Algorithms Implementation**
  - Q-Learning
  - SARSA
  - Monte Carlo ES
  - Value Iteration (baseline)

- **Test Environments**
  - Sparse Rewards Environment
  - Long Horizon Environment
  - Stochastic Rewards Environment

- **Rich Visualizations**
  - Interactive learning curve plots with confidence intervals
  - Dynamic 3D optimal path visualizations
  - Real-time performance comparisons
  - Interactive HTML dashboards

## Key Findings

### Dynamic Performance Analysis
Our dynamic performance dashboard (`dynamic_performance_dashboard.html`) reveals several interesting patterns:

- **Algorithm Convergence**:
  - Q-Learning shows fastest initial learning in sparse reward environments
  - SARSA demonstrates more stable learning curves with lower variance
  - Monte Carlo ES exhibits strong final performance but slower learning
  - Value Iteration provides consistent baseline performance

- **Environment-Specific Insights**:
  - Stochastic environments: SARSA shows more robust performance
  - Long horizon tasks: Q-Learning achieves better asymptotic performance
  - Sparse rewards: Monte Carlo ES demonstrates competitive final results

### Optimal Path Analysis
The dynamic optimal paths visualization (`dynamic_optimal_paths_3d.html`) shows:

- **Path Optimization**:
  - Progressive improvement in path efficiency over training
  - Clear visualization of how algorithms adapt to environment changes
  - Comparison of exploration vs exploitation strategies

- **Algorithm Characteristics**:
  - Q-Learning: More direct paths after convergence
  - SARSA: Safer paths avoiding risky states
  - Monte Carlo ES: Diverse path exploration early in training

## Requirements

```bash
numpy>=1.21.0
matplotlib>=3.4.0
plotly>=5.3.0
pandas>=1.3.0
seaborn>=0.11.0
pytest>=6.2.0
```

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rl-policy-iteration-comparison.git
cd rl-policy-iteration-comparison
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the main comparison:
```bash
python grid_world_comparison.py
```

## Interactive Visualizations

This repository includes interactive HTML visualizations that can be viewed in any modern web browser:

1. `dynamic_performance_dashboard.html`
   - Real-time performance metrics
   - Learning curve comparisons
   - Environment-specific analysis

2. `dynamic_optimal_paths_3d.html`
   - 3D visualization of optimal paths
   - Training progression visualization
   - Algorithm behavior comparison

To view the visualizations:
1. Clone the repository
2. Open the HTML files in a web browser
3. Interact with the plots to explore different aspects of the results

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please open an issue in the repository. 
