# Reinforcement-Learning
Exploring Sutton and Barlo's Second Edition of Reinforcement Learning. 

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
  - Learning curve plots with confidence intervals
  - Value function contours
  - Performance comparisons
  - Interactive HTML dashboards

## Requirements

```bash
numpy
matplotlib
plotly
pandas
seaborn
json
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

## 📊 Project Structure

- `grid_world_comparison.py`: Main script for running algorithm comparisons
- `rl_algorithms.py`: Core implementations of RL algorithms
- `monte_carlo.py`: Monte Carlo ES implementation
- `environment_tests.py`: Environment definitions and test configurations
- `combined_results.html`: Interactive visualization dashboard

## Results

The project includes comprehensive visualizations and comparisons of algorithm performance across different environments:

- Learning curves with confidence intervals
- Environment-specific performance metrics
- Algorithm comparison analysis
- Interactive HTML dashboards for detailed exploration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or feedback, please open an issue in the repository. 
