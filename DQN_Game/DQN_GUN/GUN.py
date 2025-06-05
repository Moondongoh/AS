import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import gymnasium as gym
from gymnasium import spaces


class ShootingGameEnv(gym.Env):
    def __init__(self):
        super(ShootingGameEnv, self).__init__()
        self.map_width = 8
        self.map_height = 20
        self.max_steps = 200

        self.action_space = spaces.Discrete(4)  # 정지, 좌, 우, 발사
        self.observation_space = spaces.Box(low=0, high=20, shape=(3,), dtype=np.int32)

        self.gun_x = 0
        self.target_x = 0
        self.bullet_pos = None

        self.step_count = 0
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.gun_x = self.map_width // 2
        self.target_x = random.randint(0, self.map_width - 1)
        self.bullet_pos = None
        self.step_count = 0
        self.done = False
        return self._get_obs(), {}

    def _get_obs(self):
        bullet_y = self.bullet_pos[1] if self.bullet_pos else -1
        return np.array([self.gun_x, self.target_x, bullet_y], dtype=np.int32)

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True, False, {}

        if action == 1 and self.gun_x > 0:
            self.gun_x -= 1
        elif action == 2 and self.gun_x < self.map_width - 1:
            self.gun_x += 1
        elif action == 3 and self.bullet_pos is None:
            self.bullet_pos = [self.gun_x, self.map_height - 1]

        reward = 0

        if self.bullet_pos:
            self.bullet_pos[1] -= 1
            if self.bullet_pos[1] == 0:
                if self.bullet_pos[0] == self.target_x:
                    reward = 100
                    self.done = True
                else:
                    reward = -10
                self.bullet_pos = None
            elif self.bullet_pos[1] < 0:
                self.bullet_pos = None
                reward = -5

        if self.step_count % 5 == 0:
            self.target_x += random.choice([-1, 0, 1])
            self.target_x = np.clip(self.target_x, 0, self.map_width - 1)

        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.done = True

        return self._get_obs(), reward, self.done, False, {}

    def close(self):
        pass


class DQN(nn.Module):
    def __init__(self, input_dim=3, output_dim=4):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self):
        self.state_dim = 3
        self.action_dim = 4
        self.gamma = 0.99
        self.lr = 1e-3
        self.batch_size = 64
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay = 5000

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = deque(maxlen=10000)
        self.steps_done = 0

    def select_action(self, state):
        eps = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -1.0 * self.steps_done / self.epsilon_decay
        )
        self.steps_done += 1
        if random.random() < eps:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                return self.policy_net(state_tensor).argmax(1).item()

    def store(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s_, d = zip(*batch)
        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        s_ = torch.FloatTensor(s_).to(self.device)
        d = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        q = self.policy_net(s).gather(1, a)
        with torch.no_grad():
            q_target = (
                r + (1 - d) * self.gamma * self.target_net(s_).max(1, keepdim=True)[0]
            )

        loss = nn.MSELoss()(q, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


if __name__ == "__main__":
    env = ShootingGameEnv()
    agent = DQNAgent()
    for ep in range(1000):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.store(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            total_reward += reward
        agent.update_target()
        print(f"Episode {ep+1} Reward: {total_reward}")

    torch.save(agent.policy_net.state_dict(), "shooting_dqn.pt")
    print("Model saved as shooting_dqn.pt")
