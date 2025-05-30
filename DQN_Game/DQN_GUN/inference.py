import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class ShootingGameEnv(gym.Env):
    def __init__(self):
        super(ShootingGameEnv, self).__init__()
        self.map_width = 8
        self.map_height = 20
        self.max_steps = 200

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=20, shape=(3,), dtype=np.int32)

        self.gun_x = 0
        self.target_x = 0
        self.bullet_pos = None

        self.step_count = 0
        self.done = False
        self.fig = None
        self.ax = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.gun_x = self.map_width // 2
        self.target_x = np.random.randint(0, self.map_width)
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
            self.target_x += np.random.choice([-1, 0, 1])
            self.target_x = np.clip(self.target_x, 0, self.map_width - 1)

        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.done = True

        return self._get_obs(), reward, self.done, False, {}

    def render(self, mode="human"):
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(4, 10))
            plt.ion()

        self.ax.clear()
        self.ax.set_xlim(0, self.map_width)
        self.ax.set_ylim(0, self.map_height)
        self.ax.set_aspect("equal")
        self.ax.set_facecolor("black")

        # Target
        self.ax.add_patch(patches.Rectangle((self.target_x, 0), 1, 1, color="red"))
        # Gun
        self.ax.add_patch(
            patches.Rectangle((self.gun_x, self.map_height - 1), 1, 1, color="blue")
        )
        # Bullet
        if self.bullet_pos:
            x, y = self.bullet_pos
            if 0 <= y < self.map_height:
                self.ax.add_patch(
                    patches.Circle((x + 0.5, y + 0.5), 0.2, color="white")
                )

        self.ax.set_xticks([])
        self.ax.set_yticks([])
        plt.pause(0.05)
        self.fig.canvas.draw()


class DQN(nn.Module):
    def __init__(self, input_dim=3, output_dim=4):
        super().__init__()
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(3, 4).to(self.device)
        self.policy_net.eval()

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.policy_net(state_tensor).argmax(1).item()


if __name__ == "__main__":
    agent = DQNAgent()
    agent.policy_net.load_state_dict(
        torch.load("shooting_dqn.pt", map_location=agent.device)
    )
    print("모델 로드 완료: shooting_dqn.pt")

    env = ShootingGameEnv()
    for ep in range(3):
        print(f"\n[Inference Episode {ep+1}]")
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            total_reward += reward
            env.render()
        print(f"Total Reward: {total_reward}")

    plt.ioff()
    plt.show()
