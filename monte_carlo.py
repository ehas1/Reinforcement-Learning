import numpy as np
from collections import defaultdict

class MonteCarloES:
    def __init__(self, n_states, n_actions, gamma=0.99):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        
        # Initialize Q-values and returns
        self.Q = defaultdict(lambda: np.zeros(n_actions))
        self.returns = defaultdict(lambda: [[] for _ in range(n_actions)])
        self.policy = defaultdict(lambda: np.ones(n_actions) / n_actions)
        
    def choose_action(self, state):
        """Choose action based on current policy."""
        return np.random.choice(self.n_actions, p=self.policy[state])
    
    def update(self, episode):
        """Update policy based on generated episode."""
        state_action_pairs = [(state, action) for state, action, _ in episode]
        returns = 0
        
        # Process episode in reverse order
        for t in range(len(episode)-1, -1, -1):
            state, action, reward = episode[t]
            returns = self.gamma * returns + reward
            
            # First-visit MC: only update if this is the first occurrence
            if (state, action) not in state_action_pairs[:t]:
                self.returns[state][action].append(returns)
                self.Q[state][action] = np.mean(self.returns[state][action])
                
                # Update policy (greedy with respect to Q)
                best_action = np.argmax(self.Q[state])
                self.policy[state] = np.zeros(self.n_actions)
                self.policy[state][best_action] = 1.0
    
    def generate_episode(self, env, max_steps=100):
        """Generate episode using exploring starts."""
        episode = []
        state = env.reset()[0]  # Get initial state
        
        # Exploring starts: choose random first action
        action = np.random.randint(self.n_actions)
        
        for _ in range(max_steps):
            next_state, reward, done, _, _ = env.step(action)
            episode.append((state, action, reward))
            
            if done:
                break
                
            state = next_state
            action = self.choose_action(state)
            
        return episode 