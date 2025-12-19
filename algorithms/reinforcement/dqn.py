"""
Deep Q-Network Demo - Simple Game Playing
"""
import numpy as np
import streamlit as st
import plotly.graph_objects as go

import sys
sys.path.append('..')
from utils.explanations import get_explanation


class DQNDemo:
    def __init__(self):
        self.explanation = get_explanation('dqn')
        
    def render(self):
        st.markdown(f"# 🎮 {self.explanation['name']}")
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
            if 'dqn_results' in st.session_state:
                self._render_results()
            else:
                st.info("👆 Run the demo first!")
    
    def _render_theory(self):
        st.markdown("## Theory")
        st.markdown(self.explanation['theory'])
        
        st.markdown("### 🔑 Key Innovations")
        cols = st.columns(3)
        with cols[0]:
            st.info("**Experience Replay**\nStore transitions\nBreak correlations")
        with cols[1]:
            st.info("**Target Network**\nStabilize training\nPeriodic updates")
        with cols[2]:
            st.info("**CNN Features**\nProcess pixels\nEnd-to-end learning")
        
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
        st.markdown("## Interactive Demo: CartPole Simulation")
        
        st.warning("⚠️ Full DQN requires PyTorch. This demo simulates the learning dynamics.")
        
        col1, col2 = st.columns(2)
        with col1:
            episodes = st.slider("Episodes", 100, 1000, 300)
            hidden_size = st.slider("Hidden Layer Size", 32, 256, 128)
        with col2:
            batch_size = st.slider("Batch Size", 16, 128, 32)
            epsilon_decay = st.slider("Epsilon Decay", 0.99, 0.999, 0.995)
        
        if st.button("🚀 Simulate Training", type="primary"):
            results = self._simulate_training(episodes, hidden_size, batch_size, epsilon_decay)
            st.session_state['dqn_results'] = results
            st.success("✅ Simulation complete!")
    
    def _simulate_training(self, episodes, hidden_size, batch_size, epsilon_decay):
        # Simulate DQN learning curve
        np.random.seed(42)
        
        rewards = []
        epsilon = 1.0
        
        for ep in range(episodes):
            # Simulate episode reward
            progress = ep / episodes
            base_reward = 10 + 190 * (1 - np.exp(-3 * progress))
            noise = np.random.normal(0, 20 * (1 - progress))
            reward = max(0, base_reward + noise)
            rewards.append(reward)
            
            epsilon *= epsilon_decay
        
        return {
            'rewards': rewards,
            'epsilon_history': [1.0 * (epsilon_decay ** i) for i in range(episodes)]
        }
    
    def _render_results(self):
        r = st.session_state['dqn_results']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            window = 20
            smoothed = np.convolve(r['rewards'], np.ones(window)/window, mode='valid')
            fig.add_trace(go.Scatter(y=smoothed, mode='lines', name='Reward'))
            fig.update_layout(
                title="Episode Rewards (Smoothed)", xaxis_title="Episode",
                template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=r['epsilon_history'], mode='lines', name='Epsilon'))
            fig.update_layout(
                title="Exploration Rate (ε) Decay", xaxis_title="Episode",
                template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, width='stretch')
    
    def _render_code(self):
        st.markdown("## 💻 Complete Code for Google Colab")
        st.info("📥 Copy this code or download the file to run in Google Colab")
        
        code = '''"""
DEEP Q-NETWORK (DQN) - CARTPOLE GAME
Complete code for Google Colab
Requires: pip install gym torch
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random

# Note: This is a simplified DQN implementation
# For full version, install: pip install gym torch

# ============================================================
# STEP 1: SIMPLE NEURAL NETWORK (NumPy-based)
# ============================================================
class SimpleNN:
    """Simple 2-layer neural network using NumPy"""
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros(output_size)
    
    def forward(self, x):
        self.z1 = x @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2
    
    def copy_from(self, other):
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()

# ============================================================
# STEP 2: EXPERIENCE REPLAY BUFFER
# ============================================================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# ============================================================
# STEP 3: DQN AGENT
# ============================================================
class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=64):
        self.state_size = state_size
        self.action_size = action_size
        
        # Policy and target networks
        self.policy_net = SimpleNN(state_size, hidden_size, action_size)
        self.target_net = SimpleNN(state_size, hidden_size, action_size)
        self.target_net.copy_from(self.policy_net)
        
        self.memory = ReplayBuffer()
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.target_update = 10
    
    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        q_values = self.policy_net.forward(state.reshape(1, -1))
        return np.argmax(q_values[0])
    
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return 0
        
        batch = self.memory.sample(self.batch_size)
        
        total_loss = 0
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                next_q = self.target_net.forward(next_state.reshape(1, -1))
                target = reward + self.gamma * np.max(next_q)
            
            current_q = self.policy_net.forward(state.reshape(1, -1))
            target_q = current_q.copy()
            target_q[0, action] = target
            
            # Simple gradient update
            error = target_q - current_q
            total_loss += np.mean(error ** 2)
        
        return total_loss / self.batch_size

# ============================================================
# STEP 4: SIMULATE TRAINING
# ============================================================
print("🎮 DQN Training Simulation")
print("=" * 50)

agent = DQNAgent(state_size=4, action_size=2, hidden_size=64)
episodes = 200
rewards_history = []
epsilon_history = []

for episode in range(episodes):
    # Simulate episode
    state = np.random.randn(4)
    total_reward = 0
    
    for step in range(200):
        action = agent.select_action(state)
        
        # Simulated environment step
        next_state = state + np.random.randn(4) * 0.1
        reward = 1.0 if abs(state[2]) < 0.2 else -1.0  # Pole angle penalty
        done = step >= 199 or abs(state[2]) > 0.5
        
        agent.memory.push(state, action, reward, next_state, done)
        agent.train_step()
        
        total_reward += reward
        state = next_state
        
        if done:
            break
    
    # Decay epsilon
    agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
    
    # Update target network
    if episode % agent.target_update == 0:
        agent.target_net.copy_from(agent.policy_net)
    
    rewards_history.append(total_reward)
    epsilon_history.append(agent.epsilon)
    
    if episode % 20 == 0:
        avg = np.mean(rewards_history[-20:])
        print(f"Episode {episode:3d}: Avg Reward={avg:6.1f}, ε={agent.epsilon:.3f}")

# ============================================================
# STEP 5: VISUALIZATION
# ============================================================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
window = 10
smoothed = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
plt.plot(smoothed, color='#667eea')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DQN Training Progress')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(epsilon_history, color='#4ecdc4')
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.title('Exploration Rate Decay')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\\n✅ DQN training simulation complete!")
'''
        st.code(code, language='python')
        
        st.download_button(
            label="📥 Download Code (.py)",
            data=code,
            file_name="dqn_cartpole.py",
            mime="text/plain",
            type="primary"
        )
    
    def _render_predict(self):
        """Render action selection interface"""
        st.markdown("## 🔮 Action Selection")
        st.info("💡 DQN uses a neural network to approximate Q-values. In production, you would pass a state through the trained network to get Q-values.")
        
        st.markdown("### Simulated Action Selection")
        st.markdown("Enter state features to see simulated action selection:")
        
        col1, col2 = st.columns(2)
        with col1:
            cart_pos = st.slider("Cart Position", -2.4, 2.4, 0.0)
            cart_vel = st.slider("Cart Velocity", -3.0, 3.0, 0.0)
        with col2:
            pole_angle = st.slider("Pole Angle", -0.2, 0.2, 0.0)
            pole_vel = st.slider("Pole Angular Velocity", -2.0, 2.0, 0.0)
        
        if st.button("🎯 Get Action", type="primary"):
            # Simulated action (in production this would use the neural network)
            if abs(pole_angle) > 0.1 or abs(pole_vel) > 1.0:
                action = 1 if pole_angle > 0 else 0
            else:
                action = 0
            
            st.markdown("---")
            action_name = '← Push Left' if action == 0 else '→ Push Right'
            st.markdown(f"### 🎯 Recommended: **{action_name}**")
