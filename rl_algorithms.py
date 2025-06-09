import numpy as np
from collections import defaultdict
import itertools

def value_iteration_generic(env, gamma=0.99, theta=0.0001, max_iterations=1000):
    """
    Generic value iteration algorithm.
    Returns the optimal value function as a numpy array.
    """
    # Initialize value function
    n_states = env.dims[0] * env.dims[1]
    V = np.zeros(n_states)
    
    for _ in range(max_iterations):
        delta = 0
        for state in range(n_states):
            v = V[state]
            # Try all possible actions
            values = []
            for action in range(4):  # Assuming 4 actions: up, down, left, right
                # Get row, col from state index
                row = state // env.dims[1]
                col = state % env.dims[1]
                
                # Save current state
                current_state = env.current_state
                env.current_state = (row, col)
                
                # Take action
                next_state, reward, done, _, _ = env.step(action)
                next_state_idx = next_state[0] * env.dims[1] + next_state[1]
                values.append(reward + gamma * V[next_state_idx])
                
                # Restore state
                env.current_state = current_state
            
            V[state] = max(values)
            delta = max(delta, abs(v - V[state]))
        
        if delta < theta:
            break
    
    return V

def policy_iteration_generic(env, gamma=0.99, theta=0.0001, max_iterations=1000):
    """
    Generic policy iteration algorithm.
    Returns the optimal value function and policy.
    """
    # Initialize value function and policy
    n_states = env.dims[0] * env.dims[1]
    V = np.zeros(n_states)
    policy = np.zeros(n_states, dtype=int)
    
    for _ in range(max_iterations):
        # Policy Evaluation
        while True:
            delta = 0
            for state in range(n_states):
                v = V[state]
                row = state // env.dims[1]
                col = state % env.dims[1]
                
                # Save current state
                current_state = env.current_state
                env.current_state = (row, col)
                
                # Take action according to policy
                action = policy[state]
                next_state, reward, done, _, _ = env.step(action)
                next_state_idx = next_state[0] * env.dims[1] + next_state[1]
                
                # Update value
                V[state] = reward + gamma * V[next_state_idx]
                
                # Restore state
                env.current_state = current_state
                
                delta = max(delta, abs(v - V[state]))
            
            if delta < theta:
                break
        
        # Policy Improvement
        policy_stable = True
        for state in range(n_states):
            old_action = policy[state]
            row = state // env.dims[1]
            col = state % env.dims[1]
            
            # Try all actions
            values = []
            for action in range(4):
                # Save current state
                current_state = env.current_state
                env.current_state = (row, col)
                
                # Take action
                next_state, reward, done, _, _ = env.step(action)
                next_state_idx = next_state[0] * env.dims[1] + next_state[1]
                values.append(reward + gamma * V[next_state_idx])
                
                # Restore state
                env.current_state = current_state
            
            policy[state] = np.argmax(values)
            
            if old_action != policy[state]:
                policy_stable = False
        
        if policy_stable:
            break
    
    return V, policy

def run_episode_generic(env, agent, is_sarsa=False):
    """Generic episode runner that works with any environment."""
    state = env.reset()[0]
    done = False
    total_reward = 0
    
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _, _ = env.step(action)
        
        if is_sarsa:
            next_action = agent.get_action(next_state)
            agent.learn(state, action, reward, next_state, next_action, done)
        else:
            agent.learn(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
    
    return total_reward 