# RL Policy Iteration Comparison

A comprehensive implementation and visualization of various reinforcement learning algorithms, focusing on policy iteration methods across different test environments. This project compares the performance of Q-Learning, SARSA, Monte Carlo ES, and Value Iteration algorithms in challenging grid world environments.

Ideas based on Sutton and Barto's Second Edition on Reinforcement Learning.

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
  - Learning curve plots with confidence intervals
  - Grid world comparisons in 2D and 3D
  - Performance analysis visualizations
  - Optimal path comparisons

## Key Findings

### Performance Analysis
Our analysis reveals several interesting patterns:

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
The optimal paths visualization shows:

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
git clone https://github.com/ehas1/Reinforcement-Learning.git
cd Reinforcement-Learning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the main comparison:
```bash
python grid_world_comparison.py
```

## Visualizations

This repository includes key visualizations:

1. `grid_world_comparison_2d.png` and `grid_world_comparison_3d.png`
   - 2D and 3D representations of the grid world
   - Algorithm performance comparisons
   - State-value function visualization

2. `enhanced_comparison_3d.png`
   - Detailed 3D visualization of algorithm performance
   - Comparative analysis across environments

3. `optimal_paths_comparison.png`
   - Visual comparison of optimal paths
   - Algorithm behavior analysis
   - Policy convergence demonstration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please open an issue in the repository. 
