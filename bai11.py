import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# CÂU 11.3.1: CÀI ĐẶT MÔI TRƯỜNG GYMNASIUM
# ==========================================
class VietnamEconomyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(5)
        # 4 state variables (GDP growth, Digital index, AI capacity, Unemployment risk), each has 3 levels
        self.observation_space = spaces.MultiDiscrete([3, 3, 3, 3])
        self.T = 10
        
        # 5 actions (Budget allocation percentages)
        self.allocation = {
            0: np.array([0.70, 0.10, 0.10, 0.10]), # Truyền thống
            1: np.array([0.40, 0.25, 0.15, 0.20]), # Cân bằng
            2: np.array([0.25, 0.45, 0.15, 0.15]), # Số hóa nhanh
            3: np.array([0.20, 0.20, 0.45, 0.15]), # AI dẫn dắt
            4: np.array([0.30, 0.20, 0.10, 0.40])  # Bao trùm
        }
        self.w = np.array([0.40, 0.25, 0.20, 0.15])
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Khởi tạo trạng thái ban đầu: VN 2026 (1=medium, 0=low)
        self.state = np.array([1, 1, 0, 1]) 
        self.t = 0
        
        # Giá trị khởi tạo từ Bài 1
        self.K = 27500.0
        self.D = 20.3
        self.AI = 86.0
        self.H = 30.0
        self.Y = self.K**0.33 * (54.0)**0.42 * self.D**0.10 * self.AI**0.08 * self.H**0.07
        
        return self.state, {}

    def step(self, action):
        a = self.allocation[action]
        budget = 1000 # Ngân sách giả định 1000 nghìn tỷ
        
        prev_Y = self.Y
        
        # Cập nhật vốn tích lũy
        self.K += a[0] * budget
        self.D += a[1] * budget / 100  # Hệ số quy đổi giả định
        self.AI += a[2] * budget / 20  
        self.H += a[3] * budget / 200  
        
        # Tính Y mới
        self.Y = self.K**0.33 * (54.0)**0.42 * self.D**0.10 * self.AI**0.08 * self.H**0.07
        
        # Tính các chỉ số cho phần thưởng (Mô phỏng đơn giản hóa)
        delta_Y = self.Y - prev_Y
        unemploy_risk = max(0, 40 - self.H) # H càng cao, U càng thấp
        cyber_risk = self.AI * 0.1
        emission = self.K * 0.05
        
        # Hàm phần thưởng (Cần Normalize trong dự án thực tế)
        reward = self.w[0]*delta_Y - self.w[1]*unemploy_risk - self.w[2]*cyber_risk - self.w[3]*emission
        
        self.t += 1
        done = self.t >= self.T
        
        # Rời rạc hóa State (Discretization) để Q-table hiểu được
        state_0 = min(2, max(0, int(delta_Y / 50)))       # GDP Growth
        state_1 = min(2, max(0, int(self.D / 30)))        # Digital Index
        state_2 = min(2, max(0, int(self.AI / 50)))       # AI Capacity
        state_3 = min(2, max(0, int(unemploy_risk / 10))) # U Risk
        
        self.state = np.array([state_0, state_1, state_2, state_3])
        
        return self.state, reward, done, False, {}

# ==========================================
# CÂU 11.3.2 & 11.3.3: HUẤN LUYỆN Q-LEARNING
# ==========================================
env = VietnamEconomyEnv()

# Khởi tạo Q-table (81 states x 5 actions)
Q = np.zeros((3, 3, 3, 3, 5))

alpha = 0.1
gamma = 0.95
episodes = 10000
rewards_history = []

print("Bắt đầu huấn luyện Q-learning...")
for ep in range(episodes):
    s, _ = env.reset()
    total_reward = 0
    
    while True:
        # Khối lệnh này đã được lùi đúng 4 khoảng trắng so với while
        eps = max(0.05, 1.0 - ep/5000)
        
        if np.random.rand() < eps:
            a = env.action_space.sample() # Khám phá (Explore)
        else:
            a = int(np.argmax(Q[tuple(s)])) # Khai thác (Exploit)
            
        s2, r, done, _, _ = env.step(a)
        
        # Cập nhật Q-value theo công thức Bellman
        Q[tuple(s) + (a,)] += alpha * (r + gamma * np.max(Q[tuple(s2)]) - Q[tuple(s) + (a,)])
        
        s = s2
        total_reward += r
        
        if done:
            break
            
    rewards_history.append(total_reward)

print("Huấn luyện hoàn tất!")

# ==========================================
# CÂU 11.3.4: ĐÁNH GIÁ CHÍNH SÁCH SO VỚI RULE-BASED
# ==========================================
def evaluate_policy(policy_type, episodes=10):
    total_rewards = []
    for _ in range(episodes):
        s, _ = env.reset()
        ep_reward = 0
        while True:
            # Khối lệnh này đã được lùi đúng 4 khoảng trắng so với while
            if policy_type == 'learned':
                a = int(np.argmax(Q[tuple(s)]))
            elif policy_type == 'always_a1':
                a = 1
            elif policy_type == 'always_a3':
                a = 3
            elif policy_type == 'random':
                a = env.action_space.sample()
                
            s, r, done, _, _ = env.step(a)
            ep_reward += r
            if done: 
                break
                
        total_rewards.append(ep_reward)
    return np.mean(total_rewards)

print("\n--- KẾT QUẢ ĐÁNH GIÁ (Phần thưởng tích lũy trung bình) ---")
print(f"Chính sách Q-learning (Learned): {evaluate_policy('learned'):.2f}")
print(f"Chính sách Cân bằng (Always a1): {evaluate_policy('always_a1'):.2f}")
print(f"Chính sách AI dẫn dắt (Always a3): {evaluate_policy('always_a3'):.2f}")
print(f"Chính sách Ngẫu nhiên (Random): {evaluate_policy('random'):.2f}")