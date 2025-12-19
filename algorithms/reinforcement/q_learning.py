"""
Q-Learning Demo - Taxi Navigation in Grid World
"""
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import time

import sys
sys.path.append('..')
from utils.explanations import get_explanation
from data.datasets import DatasetGenerator


class QLearningDemo:
    def __init__(self):
        self.explanation = get_explanation('q_learning')
        
    def render(self):
        st.markdown(f"# 🚕 {self.explanation['name']}")
        st.markdown(f"**Project:** {self.explanation['project']}")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Learn", "🎮 Demo", "🔮 Predict", "💻 Code", "📊 Results"])
        
        with tab1:
            self._render_theory()
            st.markdown("📖 **For more details:** [Click here for geeksforgeeks](https://www.geeksforgeeks.org/deep-learning/deep-q-learning/)")
        with tab2:
            self._render_demo()
        with tab3:
            self._render_predict()
        with tab4:
            self._render_code()
        with tab5:
            if 'ql_results' in st.session_state:
                self._render_results()
            else:
                st.info("👆 Run the demo first!")
    
    def _render_theory(self):
        st.markdown("## Theory")
        st.markdown(self.explanation['theory'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### ✅ Pros")
            for pro in self.explanation['pros']:
                st.markdown(f"- {pro}")
        with col2:
            st.markdown("### ❌ Cons")
            for con in self.explanation['cons']:
                st.markdown(f"- {con}")
    
    def _render_demo(self):
        st.markdown("## Interactive Demo: Grid World Navigation")
        
        st.info("🚕 The agent learns to navigate from start (green) to goal (red) while avoiding obstacles (black)")
        
        col1, col2 = st.columns(2)
        with col1:
            grid_size = st.slider("Grid Size", 4, 8, 5)
            episodes = st.slider("Training Episodes", 100, 2000, 500)
        with col2:
            learning_rate = st.slider("Learning Rate (α)", 0.01, 1.0, 0.1)
            discount = st.slider("Discount Factor (γ)", 0.1, 1.0, 0.95)
        
        epsilon = st.slider("Epsilon (Exploration)", 0.01, 1.0, 0.1)
        
        if st.button("🚀 Train Agent", type="primary"):
            results = self._train_agent(grid_size, episodes, learning_rate, discount, epsilon)
            st.session_state['ql_results'] = results
            st.success("✅ Training complete!")
    
    def _train_agent(self, grid_size, episodes, alpha, gamma, epsilon):
        # Initialize Q-table
        n_states = grid_size * grid_size
        n_actions = 4  # Up, Down, Left, Right
        Q = np.zeros((n_states, n_actions))
        
        # Define environment
        goal = n_states - 1
        obstacles = [grid_size + 1, 2 * grid_size + grid_size - 2, 3 * grid_size + 1]
        
        rewards_history = []
        steps_history = []
        
        for episode in range(episodes):
            state = 0  # Start
            total_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 100:
                # Epsilon-greedy action selection
                if np.random.random() < epsilon:
                    action = np.random.randint(n_actions)
                else:
                    action = np.argmax(Q[state])
                
                # Take action
                row, col = state // grid_size, state % grid_size
                if action == 0:  # Up
                    new_row = max(0, row - 1)
                    new_col = col
                elif action == 1:  # Down
                    new_row = min(grid_size - 1, row + 1)
                    new_col = col
                elif action == 2:  # Left
                    new_row = row
                    new_col = max(0, col - 1)
                else:  # Right
                    new_row = row
                    new_col = min(grid_size - 1, col + 1)
                
                next_state = new_row * grid_size + new_col
                
                # Get reward
                if next_state == goal:
                    reward = 100
                    done = True
                elif next_state in obstacles:
                    reward = -10
                else:
                    reward = -1
                
                # Q-learning update
                Q[state, action] = Q[state, action] + alpha * (
                    reward + gamma * np.max(Q[next_state]) - Q[state, action]
                )
                
                state = next_state
                total_reward += reward
                steps += 1
            
            rewards_history.append(total_reward)
            steps_history.append(steps)
        
        return {
            'Q': Q, 'grid_size': grid_size, 'goal': goal,
            'obstacles': obstacles, 'rewards': rewards_history,
            'steps': steps_history
        }
    
    def _render_results(self):
        r = st.session_state['ql_results']
        
        st.markdown("## 📊 Training Results")
        
        # Plot rewards over episodes
        fig = go.Figure()
        window = 50
        smoothed = np.convolve(r['rewards'], np.ones(window)/window, mode='valid')
        fig.add_trace(go.Scatter(y=smoothed, mode='lines', name='Avg Reward'))
        fig.update_layout(
            title="Rewards Over Episodes (Smoothed)",
            xaxis_title="Episode", yaxis_title="Total Reward",
            template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, width='stretch')
        
        # Visualize learned policy
        st.markdown("## 🗺️ Learned Policy")
        
        grid_size = r['grid_size']
        Q = r['Q']
        arrows = ['↑', '↓', '←', '→']
        
        policy_grid = []
        for i in range(grid_size):
            row = []
            for j in range(grid_size):
                state = i * grid_size + j
                if state == 0:
                    row.append('🟢')
                elif state == r['goal']:
                    row.append('🔴')
                elif state in r['obstacles']:
                    row.append('⬛')
                else:
                    best_action = np.argmax(Q[state])
                    row.append(arrows[best_action])
            policy_grid.append(row)
        
        # Display as table
        st.table(policy_grid)
        
        st.success("🟢 = Start | 🔴 = Goal | ⬛ = Obstacle | Arrows = Best Action")
    
    def _render_code(self):
        st.markdown("## 💻 Complete Implementation Code")
        
        st.markdown("### 📦 Required Dependencies")
        st.code("pip install numpy matplotlib", language="bash")
        
        st.markdown("### 🐍 Full Python Code")
        
        code = '''# ============================================
# Q-LEARNING - Grid World Navigation
# ============================================
# Run: pip install numpy matplotlib

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------
# 1. ENVIRONMENT SETUP
# --------------------------------------------
grid_size = 5
n_states = grid_size * grid_size
n_actions = 4  # Up=0, Down=1, Left=2, Right=3

# Define start, goal, obstacles
start = 0
goal = n_states - 1
obstacles = [6, 12, 18]  # Some blocked cells

print("="*50)
print("GRID WORLD ENVIRONMENT")
print("="*50)
print(f"Grid Size: {grid_size}x{grid_size}")
print(f"Start: {start}, Goal: {goal}")
print(f"Obstacles: {obstacles}")

# --------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------
def state_to_pos(state):
    return state // grid_size, state % grid_size

def pos_to_state(row, col):
    return row * grid_size + col

def get_next_state(state, action):
    row, col = state_to_pos(state)
    if action == 0:    # Up
        row = max(0, row - 1)
    elif action == 1:  # Down
        row = min(grid_size - 1, row + 1)
    elif action == 2:  # Left
        col = max(0, col - 1)
    elif action == 3:  # Right
        col = min(grid_size - 1, col + 1)
    return pos_to_state(row, col)

def get_reward(state):
    if state == goal:
        return 100
    elif state in obstacles:
        return -10
    else:
        return -1

# --------------------------------------------
# 3. Q-LEARNING PARAMETERS
# --------------------------------------------
alpha = 0.1      # Learning rate
gamma = 0.95     # Discount factor
epsilon = 0.1    # Exploration rate
episodes = 1000

# Initialize Q-table
Q = np.zeros((n_states, n_actions))

# --------------------------------------------
# 4. TRAINING LOOP
# --------------------------------------------
rewards_per_episode = []

for episode in range(episodes):
    state = start
    total_reward = 0
    steps = 0
    done = False
    
    while not done and steps < 100:
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = np.random.randint(n_actions)
        else:
            action = np.argmax(Q[state])
        
        # Take action
        next_state = get_next_state(state, action)
        reward = get_reward(next_state)
        
        # Q-learning update
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )
        
        state = next_state
        total_reward += reward
        steps += 1
        
        if state == goal:
            done = True
    
    rewards_per_episode.append(total_reward)

print("\\n" + "="*50)
print("TRAINING COMPLETE")
print("="*50)
print(f"Episodes: {episodes}")
print(f"Final Avg Reward (last 100): {np.mean(rewards_per_episode[-100:]):.2f}")

# --------------------------------------------
# 5. EXTRACT LEARNED POLICY
# --------------------------------------------
action_symbols = ['↑', '↓', '←', '→']
policy = np.argmax(Q, axis=1)

print("\\n" + "="*50)
print("LEARNED POLICY")
print("="*50)
for row in range(grid_size):
    row_str = ""
    for col in range(grid_size):
        state = pos_to_state(row, col)
        if state == start:
            row_str += " S "
        elif state == goal:
            row_str += " G "
        elif state in obstacles:
            row_str += " # "
        else:
            row_str += f" {action_symbols[policy[state]]} "
    print(row_str)

print("\\nLegend: S=Start, G=Goal, #=Obstacle, Arrows=Best Action")

# --------------------------------------------
# 6. VISUALIZATION
# --------------------------------------------
plt.figure(figsize=(10, 4))

# Rewards over episodes
plt.subplot(1, 2, 1)
window = 50
smoothed = np.convolve(rewards_per_episode, np.ones(window)/window, mode='valid')
plt.plot(smoothed)
plt.xlabel('Episode')
plt.ylabel('Total Reward (smoothed)')
plt.title('Learning Curve')

# Q-value heatmap
plt.subplot(1, 2, 2)
q_max = np.max(Q, axis=1).reshape(grid_size, grid_size)
plt.imshow(q_max, cmap='viridis')
plt.colorbar(label='Max Q-value')
plt.title('Learned Value Function')

plt.tight_layout()
plt.savefig('q_learning_results.png')
plt.show()

print("\\n✅ Results saved to 'q_learning_results.png'")
'''
        st.code(code, language='python')
        
        st.download_button(
            label="📥 Download Complete Code",
            data=code,
            file_name="q_learning_complete.py",
            mime="text/plain"
        )
    
    def _render_predict(self):
        """Render action selection interface"""
        st.markdown("## 🔮 Select Action for State")
        
        if 'ql_results' not in st.session_state:
            st.warning("⚠️ Please train the agent first in the Demo tab!")
            return
        
        r = st.session_state['ql_results']
        Q = r['Q']
        grid_size = r['grid_size']
        
        st.success("✅ Agent trained! Select a grid position to see the recommended action.")
        
        col1, col2 = st.columns(2)
        with col1:
            row = st.slider("Row", 0, grid_size - 1, 0)
        with col2:
            col = st.slider("Column", 0, grid_size - 1, 0)
        
        state = row * grid_size + col
        actions = ['↑ Up', '↓ Down', '← Left', '→ Right']
        
        if st.button("🎯 Get Best Action", type="primary"):
            q_values = Q[state]
            best_action = np.argmax(q_values)
            
            st.markdown("---")
            st.markdown(f"### State: ({row}, {col})")
            st.markdown(f"### 🎯 Best Action: **{actions[best_action]}**")
            
            st.markdown("#### Q-Values for all actions:")
            for i, (action, q) in enumerate(zip(actions, q_values)):
                if i == best_action:
                    st.success(f"{action}: {q:.2f} ⭐")
                else:
                    st.info(f"{action}: {q:.2f}")
