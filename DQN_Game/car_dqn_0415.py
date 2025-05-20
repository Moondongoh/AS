import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

# 사용할 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 기본 고정 위험 구역
DEFAULT_DANGER_ZONES = [
    {"top_left": (1, 2), "size": (3, 3)},
    {"top_left": (5, 5), "size": (2, 2)},
    {"top_left": (9, 6), "size": (2, 2)},
]


class GridEnvironment(gym.Env):
    """
    15x15 격자환경 (GridEnvironment)
      - 에이전트는 격자 내부를 이동하며, 위험 구역으로부터 멀어지거나 새로 방문하는 곳에 대해 보상을 받습니다.
      - 행동: 0~4 (각각 회전각 -90°, -45°, 0°, 45°, 90°)
      - 관측(state): 벽까지 거리 5 + 위험 구역까지 거리 5 + 시작점까지 거리 + 진행 방향 + 스텝 비율 = 13차원
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, grid_height=15, grid_width=15, max_steps=200):
        super().__init__()
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.max_steps = max_steps

        # 행동 및 관측 공간 정의
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(13,), dtype=np.float32
        )

        # 위험 구역 및 시작 위치
        self.danger_zones = DEFAULT_DANGER_ZONES
        self.start = (2, 1)

        # 렌더링용 변수
        self.viewer = None

        # 초기화
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.position = self.start
        self.direction = 0
        self.current_steps = 0
        self.total_reward = 0
        self.visited_positions = {}
        self.direction_score = 0
        obs = self._get_state()
        return obs, {}

    # 주어진 position이 danger_zones 중 하나의 사각형 내부에 속하는지 확인
    # 반환값: 내부이면 True, 아니면 False
    def _is_in_square(self, position):
        x, y = position
        for zone in self.danger_zones:
            x0, y0 = zone["top_left"]
            h, w = zone["size"]
            if x0 <= x < x0 + h and y0 <= y < y0 + w:
                return True
        return False

    # 지정된 방향(angle)으로 위험 구역까지 직선 탐색 후 거리(격자 단위) 반환
    # 반환값: 가장 가까운 위험 구역까지의 거리, 없으면 max(grid_dim)
    def _distance_to_danger_zone(self, position, angle):
        dx, dy = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        x, y = position
        max_d = max(self.grid_width, self.grid_height)
        for d in range(1, max_d):
            cx = int(round(x + dx * d))
            cy = int(round(y + dy * d))
            if 0 <= cx < self.grid_height and 0 <= cy < self.grid_width:
                if self._is_in_square((cx, cy)):
                    return d
        return max_d

    # 지정된 방향(angle)으로 벽(격자 경계)까지 직선 거리(실수) 계산
    # 반환값: 음수 제외 후 x, y 방향 중 최소 이동거리
    def _distance_in_direction(self, angle):
        rad = np.radians(angle)
        x, y = self.position
        dx, dy = np.cos(rad), np.sin(rad)
        tx = (
            float("inf")
            if dx == 0
            else ((0 - x) / dx if dx < 0 else (self.grid_height - 1 - x) / dx)
        )
        ty = (
            float("inf")
            if dy == 0
            else ((0 - y) / dy if dy < 0 else (self.grid_width - 1 - y) / dy)
        )
        return min(max(tx, 0), max(ty, 0))

    def _get_state(self):
        # 상태 벡터를 구성할 5개 방향(−90°, −45°, 0°, +45°, +90°) 정의
        dirs = [-90, -45, 0, 45, 90]

        # 각 방향으로 벽까지 남은 거리(격자 단위)를 그리드 너비로 정규화
        wall = [
            self._distance_in_direction(self.direction + a) / self.grid_width
            for a in dirs
        ]

        # 각 방향으로 위험 구역까지 남은 거리(격자 단위)를 그리드 너비로 정규화
        danger = [
            self._distance_to_danger_zone(self.position, self.direction + a)
            / self.grid_width
            for a in dirs
        ]

        # 시작점으로부터의 유클리드 거리(격자 단위)를 그리드 너비로 정규화
        dx = self.position[0] - self.start[0]
        dy = self.position[1] - self.start[1]
        dist_start = np.hypot(dx, dy) / self.grid_width

        # 현재 진행 방향을 arctan2로 계산 후 π로 나눠 [0,1) 범위로 정규화
        angle_norm = (np.arctan2(dy, dx) / np.pi) % 1.0

        # 현재 스텝(진행 단계) 비율 계산
        step_ratio = self.current_steps / self.max_steps

        # 벽 거리(5) + 위험구역 거리(5) + 시작점 거리 + 진행 방향 + 스텝 비율 = 13차원 상태 반환
        return np.array(
            wall + danger + [dist_start, angle_norm, step_ratio], dtype=np.float32
        )

    def step(self, action):
        # 1) 방향 회전: action(0~4)에 대응하는 회전각(-90, -45, 0, 45, 90)을 더하고 0~359°로 정규화
        rotations = [-90, -45, 0, 45, 90]
        delta = rotations[action]
        self.direction = (self.direction + delta) % 360
        if self.direction == -45:  # -45 → 315° 보정
            self.direction = 315

        # 2) 한 칸 이동: 현재 direction에 맞춰 dx,dy를 적용하고 격자 범위로 클리핑
        move_map = {0: (0, 1), 45: (-1, 1), 315: (1, 1), 90: (-1, 0), 270: (1, 0)}
        prev = self.position
        dx, dy = move_map.get(self.direction, (0, 0))
        nx = int(np.clip(self.position[0] + dx, 0, self.grid_height - 1))
        ny = int(np.clip(self.position[1] + dy, 0, self.grid_width - 1))
        self.position = (nx, ny)
        self.current_steps += 1
        # 방문 횟수 기록
        self.visited_positions[self.position] = (
            self.visited_positions.get(self.position, 0) + 1
        )

        # 3) 종료 판정 및 보상 계산 초기화
        terminated = False
        if self._is_in_square(self.position):
            # 위험 구역 충돌 시 큰 패널티
            reward = -100
            terminated = True

        elif nx in [0, self.grid_height - 1] or ny in [0, self.grid_width - 1]:
            # 벽 충돌 시 패널티
            reward = -20
            terminated = True

        else:
            # 기본 이동 보상
            reward = 1
            # 첫 방문 보너스
            if self.visited_positions[self.position] == 1:
                reward += 2
            # 재방문 페널티(최댓값 제한)
            else:
                reward -= min(self.visited_positions[self.position] * 1.5, 8)
            # 제자리 머무름 페널티
            if self.position == prev:
                reward -= 15
            # 회전 없는 직진 보너스 / 과도 회전 페널티
            if delta == 0:
                self.direction_score = 0
                reward += 0.5
            else:
                self.direction_score += 1
                reward -= 0.1
                if self.direction_score >= 3:
                    reward -= 10
            # 위험 구역과 충분히 거리가 멀면 추가 보너스
            if self._distance_to_danger_zone(self.position, self.direction) >= 2:
                reward += 1

        # 4) 스텝 상한 도달 시 truncated 판정 및 보상 조정
        truncated = False
        if self.current_steps >= self.max_steps:
            truncated = True
            reward += 50 if len(self.visited_positions) >= 30 else -20

        # 5) 누적 보상 업데이트 및 다음 상태 반환
        self.total_reward += reward
        obs = self._get_state()
        return obs, reward, terminated, truncated, {}

    # 시각화
    def render(self, mode="human"):
        if self.viewer is None:
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            self.viewer = True
        self.ax.clear()
        self.ax.set_xlim(0, self.grid_width)
        self.ax.set_ylim(0, self.grid_height)
        self.ax.grid(True)

        for zone in self.danger_zones:
            x0, y0 = zone["top_left"]
            h, w = zone["size"]
            self.ax.add_patch(
                plt.Rectangle(
                    (y0, self.grid_height - x0 - h), w, h, color="red", alpha=0.5
                )
            )

        self.ax.plot(
            self.position[1] + 0.5,
            self.grid_height - self.position[0] - 0.5,
            "bo",
            markersize=10,
        )
        self.ax.set_title(
            f"Step: {self.current_steps}, Total Reward: {self.total_reward:.2f}"
        )
        plt.pause(0.001)

        if mode == "rgb_array":
            self.fig.canvas.draw()
            img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            return img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

    # 렌더링 종료
    def close(self):
        if self.viewer:
            plt.close(self.fig)
            self.viewer = None


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim=5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.fc(x)


def train_dqn(
    env,
    num_episodes=10000,
    target_update=10,
    batch_size=256,
    max_buffer_size=20000,
    gamma=0.99,
    lr=0.0003,
):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = DQN(state_dim, action_dim).to(device)
    target_model = DQN(state_dim, action_dim).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    replay_buffer = []
    epsilon = 1.0
    epsilon_decay = 0.999
    epsilon_min = 0.01

    rewards_history = []
    best_reward = -float("inf")

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    q_vals = model(
                        torch.tensor(
                            state, dtype=torch.float32, device=device
                        ).unsqueeze(0)
                    )
                action = q_vals.argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            replay_buffer.append((state, action, reward, next_state, done))
            if len(replay_buffer) > max_buffer_size:
                replay_buffer.pop(0)
            state = next_state

            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.tensor(states, dtype=torch.float32, device=device)
                next_states = torch.tensor(
                    next_states, dtype=torch.float32, device=device
                )
                actions = torch.tensor(
                    actions, dtype=torch.long, device=device
                ).unsqueeze(1)
                rewards = torch.tensor(
                    rewards, dtype=torch.float32, device=device
                ).unsqueeze(1)
                dones = torch.tensor(
                    dones, dtype=torch.float32, device=device
                ).unsqueeze(1)

                current_q = model(states).gather(1, actions)
                with torch.no_grad():
                    max_next_q = target_model(next_states).max(1)[0].unsqueeze(1)
                    target_q = rewards + (1 - dones) * gamma * max_next_q

                loss = criterion(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        if (episode + 1) % target_update == 0:
            target_model.load_state_dict(model.state_dict())

        rewards_history.append(total_reward)
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(model.state_dict(), f"result/best_model_ep{episode+1}.pth")
            print(
                f"✅ Best model saved at episode {episode+1} with reward {total_reward:.2f}"
            )

        if (episode + 1) % 10 == 0:
            avg_reward = sum(rewards_history[-10:]) / 10
            print(
                f"Episode {episode+1}: Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.2f}"
            )

    return model, rewards_history


if __name__ == "__main__":
    env = GridEnvironment()
    model, history = train_dqn(env)
    torch.save(model.state_dict(), "dqn_model_survival.pth")
