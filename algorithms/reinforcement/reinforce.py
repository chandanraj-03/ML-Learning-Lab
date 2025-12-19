"""
REINFORCE Demo - Policy Gradient for Recommendations
"""
import numpy as np
import streamlit as st
import plotly.graph_objects as go

import sys
sys.path.append('..')
from utils.explanations import get_explanation


class REINFORCEDemo:
    def __init__(self):
        self.explanation = get_explanation('reinforce')
        
    def render(self):
        st.markdown(f"# 📰 {self.explanation['name']}")
        st.markdown(f"**Project:** {self.explanation['project']}")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Learn", "🎮 Demo", "🔮 Predict", "💻 Code", "📊 Results"])
        
        with tab1:
            self._render_theory()
            st.markdown("📖 **For more details:** [Click here for geeksforgeeks](https://www.geeksforgeeks.org/machine-learning/policy-gradient-methods-in-reinforcement-learning/)")
        with tab2:
            self._render_demo()
        with tab3:
            self._render_predict()
        with tab4:
            self._render_code()
        with tab5:
            if 'reinforce_results' in st.session_state:
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
        st.markdown("## Interactive Demo: Article Recommendation")
        
        st.info("📰 Policy learns to recommend articles that maximize user engagement")
        
        col1, col2 = st.columns(2)
        with col1:
            episodes = st.slider("Episodes", 100, 1000, 500)
            learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01)
        with col2:
            n_articles = st.slider("Number of Articles", 5, 20, 10)
            gamma = st.slider("Discount Factor (γ)", 0.9, 1.0, 0.99)
        
        if st.button("🚀 Train Policy", type="primary"):
            results = self._train_policy(episodes, learning_rate, n_articles, gamma)
            st.session_state['reinforce_results'] = results
            st.success("✅ Training complete!")
    
    def _train_policy(self, episodes, lr, n_articles, gamma):
        np.random.seed(42)
        
        # Simple softmax policy
        theta = np.zeros(n_articles)
        
        # Hidden user preferences (to simulate)
        user_prefs = np.random.randn(n_articles)
        user_prefs = user_prefs / np.sum(np.abs(user_prefs))
        
        rewards_history = []
        policy_history = []
        
        for ep in range(episodes):
            # Softmax policy
            exp_theta = np.exp(theta - np.max(theta))
            policy = exp_theta / np.sum(exp_theta)
            
            # Sample article
            action = np.random.choice(n_articles, p=policy)
            
            # Reward based on user preferences
            reward = user_prefs[action] + np.random.normal(0, 0.1)
            
            # REINFORCE update
            grad = -policy.copy()
            grad[action] += 1
            theta += lr * reward * grad
            
            rewards_history.append(reward)
            policy_history.append(policy.copy())
        
        final_policy = np.exp(theta - np.max(theta))
        final_policy = final_policy / np.sum(final_policy)
        
        st.session_state['reinforce_policy'] = final_policy
        
        return {
            'rewards': rewards_history,
            'final_policy': final_policy,
            'user_prefs': user_prefs,
            'n_articles': n_articles
        }
    
    def _render_results(self):
        r = st.session_state['reinforce_results']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            window = 20
            smoothed = np.convolve(r['rewards'], np.ones(window)/window, mode='valid')
            fig.add_trace(go.Scatter(y=smoothed, mode='lines'))
            fig.update_layout(
                title="Reward Over Training", xaxis_title="Episode",
                template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            articles = [f'Article {i+1}' for i in range(r['n_articles'])]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=articles, y=r['final_policy'], name='Learned Policy'))
            fig.add_trace(go.Bar(x=articles, y=r['user_prefs'], name='True Preference'))
            fig.update_layout(
                title="Learned Policy vs True Preferences", barmode='group',
                template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, width='stretch')
    
    def _render_code(self):
        st.markdown("## 💻 Complete Code for Google Colab")
        st.info("📥 Copy this code or download the file to run in Google Colab")
        
        code = '''"""
REINFORCE - POLICY GRADIENT FOR RECOMMENDATIONS
Complete code for Google Colab
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# STEP 1: DEFINE SOFTMAX POLICY
# ============================================================
def softmax(theta):
    """Compute softmax probabilities"""
    exp_theta = np.exp(theta - np.max(theta))
    return exp_theta / np.sum(exp_theta)

# ============================================================
# STEP 2: REINFORCE ALGORITHM
# ============================================================
class REINFORCEAgent:
    def __init__(self, n_actions, learning_rate=0.01):
        self.n_actions = n_actions
        self.theta = np.zeros(n_actions)  # Policy parameters
        self.lr = learning_rate
    
    def get_policy(self):
        return softmax(self.theta)
    
    def select_action(self):
        policy = self.get_policy()
        return np.random.choice(self.n_actions, p=policy)
    
    def update(self, action, reward):
        policy = self.get_policy()
        
        # Policy gradient: ∇log(π) * R
        grad = -policy.copy()
        grad[action] += 1  # ∇log(π) = 1 - π for selected, -π for others
        
        self.theta += self.lr * reward * grad

# ============================================================
# STEP 3: SIMULATE ARTICLE RECOMMENDATION
# ============================================================
np.random.seed(42)

# Environment settings
n_articles = 10
episodes = 500

# Hidden user preferences (agent doesn't know these)
user_prefs = np.random.randn(n_articles)
user_prefs = user_prefs / np.sum(np.abs(user_prefs))  # Normalize

print("📰 REINFORCE: Article Recommendation")
print("=" * 50)
print(f"Hidden user preferences: {np.round(user_prefs, 2)}")

# Train agent
agent = REINFORCEAgent(n_actions=n_articles, learning_rate=0.05)
rewards_history = []
policy_history = []

for ep in range(episodes):
    # Select article
    article = agent.select_action()
    
    # Reward based on user preference (with noise)
    reward = user_prefs[article] + np.random.normal(0, 0.1)
    
    # Update policy
    agent.update(article, reward)
    
    rewards_history.append(reward)
    policy_history.append(agent.get_policy().copy())
    
    if ep % 100 == 0:
        avg_reward = np.mean(rewards_history[-100:]) if ep > 0 else rewards_history[0]
        print(f"Episode {ep:3d}: Avg Reward={avg_reward:.4f}")

# ============================================================
# STEP 4: ANALYZE RESULTS
# ============================================================
final_policy = agent.get_policy()

print("\\n📊 RESULTS:")
print("-" * 40)
print(f"{'Article':<10} {'Preference':>12} {'Policy':>10}")
print("-" * 40)
for i in range(n_articles):
    print(f"Article {i+1:<3} {user_prefs[i]:>12.3f} {final_policy[i]:>10.3f}")

# Correlation between learned policy and true preferences
correlation = np.corrcoef(user_prefs, final_policy)[0, 1]
print(f"\\nCorrelation with true preferences: {correlation:.4f}")

# ============================================================
# STEP 5: VISUALIZATION
# ============================================================
plt.figure(figsize=(14, 5))

# Plot 1: Reward over training
plt.subplot(1, 3, 1)
window = 20
smoothed = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
plt.plot(smoothed, color='#667eea')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('REINFORCE Training Progress')
plt.grid(True, alpha=0.3)

# Plot 2: Policy evolution
plt.subplot(1, 3, 2)
policy_array = np.array(policy_history)
for i in range(n_articles):
    plt.plot(policy_array[:, i], alpha=0.7, label=f'Art. {i+1}')
plt.xlabel('Episode')
plt.ylabel('Selection Probability')
plt.title('Policy Evolution')
plt.legend(fontsize=6, loc='upper right')
plt.grid(True, alpha=0.3)

# Plot 3: Final policy vs preferences
plt.subplot(1, 3, 3)
x = np.arange(n_articles)
width = 0.35
plt.bar(x - width/2, user_prefs, width, label='True Prefs', color='#4ecdc4')
plt.bar(x + width/2, final_policy, width, label='Learned Policy', color='#667eea')
plt.xlabel('Article')
plt.ylabel('Value')
plt.title('Learned Policy vs True Preferences')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\\n✅ REINFORCE training complete!")
'''
        st.code(code, language='python')
        
        st.download_button(
            label="📥 Download Code (.py)",
            data=code,
            file_name="reinforce_recommendations.py",
            mime="text/plain",
            type="primary"
        )
    
    def _render_predict(self):
        """Render action selection interface"""
        st.markdown("## 🔮 Sample Article Recommendation")
        
        if 'reinforce_policy' not in st.session_state:
            st.warning("⚠️ Please train the policy first in the Demo tab!")
            return
        
        policy = st.session_state['reinforce_policy']
        n_articles = len(policy)
        
        st.success("✅ Policy trained! Click to sample an article recommendation.")
        
        if st.button("🎯 Get Recommendation", type="primary"):
            article = np.random.choice(n_articles, p=policy)
            
            st.markdown("---")
            st.markdown(f"### 📰 Recommended: **Article {article + 1}**")
            st.markdown(f"Selection probability: {policy[article]*100:.1f}%")
            
            st.markdown("#### Full Policy Distribution:")
            for i, prob in enumerate(policy):
                st.progress(prob, text=f"Article {i+1}: {prob*100:.1f}%")
