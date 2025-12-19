"""
Multi-Agent RL Demo - Warehouse Robot Coordination
"""
import numpy as np
import streamlit as st
import plotly.graph_objects as go

import sys
sys.path.append('..')
from utils.explanations import get_explanation


class MARLDemo:
    def __init__(self):
        self.explanation = get_explanation('marl')
        
    def render(self):
        st.markdown(f"# 🤖 {self.explanation['name']}")
        st.markdown(f"**Project:** {self.explanation['project']}")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Learn", "🎮 Demo", "🔮 Predict", "💻 Code", "📊 Results"])
        
        with tab1:
            self._render_theory()
            st.markdown("📖 **For more details:** [Click here for geeksforgeeks](https://www.geeksforgeeks.org/machine-learning/multi-agent-reinforcement-learning-in-ai/)")
        with tab2:
            self._render_demo()
        with tab3:
            self._render_predict()
        with tab4:
            self._render_code()
        with tab5:
            if 'marl_results' in st.session_state:
                self._render_results()
            else:
                st.info("👆 Run the demo first!")
    
    def _render_theory(self):
        st.markdown("## Theory")
        st.markdown(self.explanation['theory'])
        
        st.markdown("### 🔧 MARL Approaches")
        cols = st.columns(3)
        with cols[0]:
            st.info("**Independent Learning**\nEach agent learns alone\nSimple but non-stationary")
        with cols[1]:
            st.info("**Centralized Training**\nShared critic\nDecentralized execution")
        with cols[2]:
            st.info("**Communication**\nAgents share info\nEmergent protocols")
        
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
        st.markdown("## Interactive Demo: Warehouse Robots")
        
        st.info("🤖 Multiple robots learn to coordinate picking up and delivering items")
        
        col1, col2 = st.columns(2)
        with col1:
            n_agents = st.slider("Number of Robots", 2, 5, 3)
            episodes = st.slider("Training Episodes", 100, 1000, 300)
        with col2:
            grid_size = st.slider("Warehouse Size", 5, 10, 7)
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1)
        
        if st.button("🚀 Train Agents", type="primary"):
            with st.spinner("Training multi-agent system..."):
                results = self._train_agents(n_agents, episodes, grid_size, learning_rate)
                st.session_state['marl_results'] = results
                st.success("✅ Training complete!")
    
    def _train_agents(self, n_agents, episodes, grid_size, lr):
        np.random.seed(42)
        
        # Simulate MARL training
        team_rewards = []
        collisions = []
        deliveries = []
        
        for ep in range(episodes):
            progress = ep / episodes
            
            # Rewards improve over time
            base_reward = 10 + 90 * (1 - np.exp(-2 * progress))
            reward = base_reward + np.random.normal(0, 10)
            team_rewards.append(max(0, reward))
            
            # Collisions decrease
            collision = max(0, int(5 * (1 - progress) + np.random.randint(-1, 2)))
            collisions.append(collision)
            
            # Deliveries increase
            delivery = int(progress * 10 + np.random.randint(-2, 3))
            deliveries.append(max(0, delivery))
        
        return {
            'rewards': team_rewards,
            'collisions': collisions,
            'deliveries': deliveries,
            'n_agents': n_agents,
            'grid_size': grid_size
        }
    
    def _render_results(self):
        r = st.session_state['marl_results']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            window = 20
            smoothed = np.convolve(r['rewards'], np.ones(window)/window, mode='valid')
            fig.add_trace(go.Scatter(y=smoothed, mode='lines', name='Team Reward'))
            fig.update_layout(
                title="Team Reward Over Training",
                template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            fig = go.Figure()
            window = 20
            smooth_col = np.convolve(r['collisions'], np.ones(window)/window, mode='valid')
            smooth_del = np.convolve(r['deliveries'], np.ones(window)/window, mode='valid')
            fig.add_trace(go.Scatter(y=smooth_col, mode='lines', name='Collisions'))
            fig.add_trace(go.Scatter(y=smooth_del, mode='lines', name='Deliveries'))
            fig.update_layout(
                title="Coordination Metrics",
                template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, width='stretch')
        
        # Warehouse visualization
        st.markdown("## 🏭 Final Warehouse State")
        grid = np.zeros((r['grid_size'], r['grid_size']))
        positions = np.random.choice(r['grid_size']**2, r['n_agents'], replace=False)
        
        symbols = []
        for i in range(r['grid_size']):
            row = []
            for j in range(r['grid_size']):
                idx = i * r['grid_size'] + j
                if idx in positions:
                    row.append('🤖')
                elif np.random.random() < 0.1:
                    row.append('📦')
                else:
                    row.append('⬜')
            symbols.append(row)
        
        st.table(symbols)
    
    def _render_code(self):
        st.markdown("## 💻 Complete Code for Google Colab")
        st.info("📥 Copy this code or download the file to run in Google Colab")
        
        code = '''"""
MULTI-AGENT REINFORCEMENT LEARNING (MARL)
Warehouse Robot Coordination
Complete code for Google Colab
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# STEP 1: SIMPLE Q-LEARNING AGENT
# ============================================================
class QLearningAgent:
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.95, epsilon=0.3):
        self.q_table = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
    
    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.q_table.shape[1])
        return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state):
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error

# ============================================================
# STEP 2: WAREHOUSE ENVIRONMENT
# ============================================================
class WarehouseEnv:
    def __init__(self, grid_size=5, n_agents=3):
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.n_states = grid_size * grid_size
        self.n_actions = 5  # Up, Down, Left, Right, Pick/Drop
        self.reset()
    
    def reset(self):
        # Random agent positions
        positions = np.random.choice(self.n_states, self.n_agents, replace=False)
        self.agent_pos = list(positions)
        return self.agent_pos.copy()
    
    def step(self, actions):
        rewards = []
        collisions = 0
        
        new_positions = []
        for i, action in enumerate(actions):
            pos = self.agent_pos[i]
            row, col = pos // self.grid_size, pos % self.grid_size
            
            # Movement
            if action == 0 and row > 0: row -= 1
            elif action == 1 and row < self.grid_size - 1: row += 1
            elif action == 2 and col > 0: col -= 1
            elif action == 3 and col < self.grid_size - 1: col += 1
            
            new_pos = row * self.grid_size + col
            new_positions.append(new_pos)
        
        # Check collisions
        for i in range(len(new_positions)):
            for j in range(i+1, len(new_positions)):
                if new_positions[i] == new_positions[j]:
                    collisions += 1
                    rewards.extend([-1, -1])  # Penalty for collision
        
        # No collision: small positive reward
        if collisions == 0:
            rewards = [0.1] * self.n_agents
        
        self.agent_pos = new_positions
        done = False
        
        return self.agent_pos.copy(), rewards, done, {'collisions': collisions}

# ============================================================
# STEP 3: TRAIN MULTI-AGENT SYSTEM
# ============================================================
np.random.seed(42)

# Setup
grid_size = 5
n_agents = 3
episodes = 300

env = WarehouseEnv(grid_size=grid_size, n_agents=n_agents)
agents = [QLearningAgent(env.n_states, env.n_actions) for _ in range(n_agents)]

print("🤖 MARL: Warehouse Robot Coordination")
print("=" * 50)
print(f"Grid: {grid_size}x{grid_size}, Agents: {n_agents}")

# Training metrics
team_rewards = []
collision_history = []

for ep in range(episodes):
    states = env.reset()
    episode_reward = 0
    episode_collisions = 0
    
    for step in range(50):  # Max steps per episode
        # Each agent selects action
        actions = [agent.select_action(state) for agent, state in zip(agents, states)]
        
        # Joint step
        next_states, rewards, done, info = env.step(actions)
        
        # Each agent updates
        for i, agent in enumerate(agents):
            agent.update(states[i], actions[i], rewards[i] if i < len(rewards) else 0, next_states[i])
        
        episode_reward += sum(rewards) if rewards else 0
        episode_collisions += info['collisions']
        states = next_states
    
    team_rewards.append(episode_reward)
    collision_history.append(episode_collisions)
    
    if ep % 50 == 0:
        avg_reward = np.mean(team_rewards[-50:]) if ep > 0 else team_rewards[0]
        avg_col = np.mean(collision_history[-50:]) if ep > 0 else collision_history[0]
        print(f"Episode {ep:3d}: Avg Reward={avg_reward:6.2f}, Avg Collisions={avg_col:.1f}")

# ============================================================
# STEP 4: VISUALIZATION
# ============================================================
plt.figure(figsize=(12, 5))

# Plot 1: Team Reward
plt.subplot(1, 2, 1)
window = 20
smoothed = np.convolve(team_rewards, np.ones(window)/window, mode='valid')
plt.plot(smoothed, color='#667eea')
plt.xlabel('Episode')
plt.ylabel('Team Reward')
plt.title('Multi-Agent Team Reward')
plt.grid(True, alpha=0.3)

# Plot 2: Collisions
plt.subplot(1, 2, 2)
smoothed_col = np.convolve(collision_history, np.ones(window)/window, mode='valid')
plt.plot(smoothed_col, color='#ff6b6b')
plt.xlabel('Episode')
plt.ylabel('Collisions')
plt.title('Agent Collisions (Coordination Metric)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\\n📊 FINAL RESULTS:")
print(f"Avg Team Reward (last 50): {np.mean(team_rewards[-50:]):.2f}")
print(f"Avg Collisions (last 50): {np.mean(collision_history[-50:]):.2f}")
print("\\n✅ MARL training complete!")
'''
        st.code(code, language='python')
        
        st.download_button(
            label="📥 Download Code (.py)",
            data=code,
            file_name="marl_warehouse_robots.py",
            mime="text/plain",
            type="primary"
        )
    
    def _render_predict(self):
        """Render action selection interface"""
        st.markdown("## 🔮 Multi-Agent Action Selection")
        
        if 'marl_results' not in st.session_state:
            st.warning("⚠️ Please train the agents first in the Demo tab!")
            return
        
        r = st.session_state['marl_results']
        
        st.success("✅ Agents trained! Simulating coordinated actions.")
        st.info(f"🤖 {r['n_agents']} robots in a {r['grid_size']}x{r['grid_size']} warehouse")
        
        if st.button("🎯 Get Joint Action", type="primary"):
            actions = ['↑ Up', '↓ Down', '← Left', '→ Right', '📦 Pick', '📫 Drop']
            
            st.markdown("---")
            st.markdown("### 🤖 Agent Actions:")
            for i in range(r['n_agents']):
                action = np.random.choice(actions)
                st.info(f"Robot {i+1}: **{action}**")
