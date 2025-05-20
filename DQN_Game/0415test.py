# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import random
# import matplotlib.pyplot as plt
# from matplotlib import animation
# from PIL import Image


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")


# ### 환경 설정 class
# class GridEnvironment:
#     def __init__(self):
#         self.grid_height = 5
#         self.grid_width = 15
#         self.grid = np.zeros((self.grid_height, self.grid_width))

#         self.start = (2, 1)
#         self.max_steps = 200
#         self.current_steps = 0
#         self.direction = 0
#         self.position = self.start
#         self.last_angle = 0
#         self.total_reward = 0
#         self.flag = False
#         self.prev_dist = 0
#         self.visited_goals = []
#         self.visited_positions = {}
#         self.direction_score = 0

#         self.goals = [(i, 14) for i in range(5)]
#         self.danger_zones = [
#             {"top_left": (1, 5), "size": (1, 1)},
#             {"top_left": (2, 8), "size": (1, 1)},
#             {"top_left": (2, 11), "size": (1, 1)},
#         ]

#         self.reset()

#     def reset(self):
#         self.direction_score = 0
#         self.position = self.start
#         self.direction = 0
#         self.current_steps = 0
#         self.last_angle = np.arctan2(0, 0)
#         self.total_reward = 0
#         self.prev_dist = 0
#         self.flag = False
#         self.visited_goals = []
#         return self._get_state()

#     def _is_in_square(self, position):
#         x, y = position
#         for zone in self.danger_zones:
#             x0, y0 = zone["top_left"]
#             h, w = zone["size"]
#             if x0 <= x < x0 + h and y0 <= y < y0 + w:
#                 return True
#         return False

#     def _distance_to_boundary(self, position, angle):
#         x, y = position
#         dx, dy = np.cos(angle), np.sin(angle)
#         t_x = (
#             (0 - x) / dx
#             if dx < 0
#             else (self.grid_width - 1 - x) / dx if dx > 0 else float("inf")
#         )
#         t_y = (
#             (0 - y) / dy
#             if dy < 0
#             else (self.grid_height - 1 - y) / dy if dy > 0 else float("inf")
#         )
#         return min(max(t_x, 0), max(t_y, 0))

#     def _distance_to_danger_zone(self, position, angle):
#         dx, dy = np.cos(np.radians(angle)), np.sin(np.radians(angle))
#         x, y = position
#         max_distance = max(self.grid_width, self.grid_height)
#         for d in range(1, max_distance):
#             check_x = int(round(x + dx * d))
#             check_y = int(round(y + dy * d))
#             if 0 <= check_x < self.grid_height and 0 <= check_y < self.grid_width:
#                 if self._is_in_square((check_x, check_y)):
#                     return d
#         return max_distance

#     def _distance_in_direction(self, angle):
#         radians = np.radians(angle)
#         return self._distance_to_boundary(self.position, radians)

#     def _get_state(self):
#         directions = [-90, -45, 0, 45, 90]
#         state_info = [
#             self._distance_in_direction(self.direction + angle) / self.grid_width
#             for angle in directions
#         ]
#         danger_info = [
#             self._distance_to_danger_zone(self.position, self.direction + angle)
#             / self.grid_width
#             for angle in directions
#         ]

#         dx = self.position[0] - self.start[0]
#         dy = self.position[1] - self.start[1]
#         dist_to_start = np.sqrt(dx**2 + dy**2) / self.grid_width
#         current_angle = np.arctan2(dy, dx) / np.pi
#         step_info = self.current_steps / self.max_steps

#         return np.array(
#             state_info + danger_info + [dist_to_start, current_angle, step_info]
#         )

#     def step(self, action):
#         rotations = [-90, -45, 0, 45, 90]
#         # directions = [90, 45, 0, 315, 270]  # ↑ ↗ → ↘ ↓
#         angle_change = rotations[action]
#         self.direction = (self.direction + angle_change) % 360
#         if self.direction == -45:
#             self.direction = 315

#         move_map = {
#             0: (0, 1),  # →
#             45: (-1, 1),  # ↗
#             315: (1, 1),  # ↘
#             90: (-1, 0),  # ↑
#             270: (1, 0),  # ↓
#         }
#         prev_position = self.position
#         dx, dy = move_map.get(self.direction, (0, 0))
#         new_x = int(np.clip(self.position[0] + dx, 0, self.grid_height - 1))
#         new_y = int(np.clip(self.position[1] + dy, 0, self.grid_width - 1))
#         self.position = (new_x, new_y)
#         self.current_steps += 1

#         if self.position == prev_position:
#             reward = -10
#             self.total_reward += reward
#             return self._get_state(), reward, True

#         reward = 1

#         if angle_change == 0:
#             self.direction_score = 0
#         else:
#             self.direction_score += int(np.sign(angle_change))
#             reward += 1

#         if any(
#             np.linalg.norm(np.array(self.position) - np.array(goal)) < 0.5
#             for goal in self.goals
#         ):
#             reward += 10
#             self.total_reward += reward
#             return self._get_state(), reward, True

#         if abs(self.direction_score) >= 3:
#             self.total_reward += reward
#             return self._get_state(), reward, True

#         if self._is_in_square(self.position):
#             reward -= 20
#             self.total_reward += reward
#             return self._get_state(), reward, True

#         if self.position[0] in [0, self.grid_height - 1] or self.position[1] in [
#             0,
#             self.grid_width - 1,
#         ]:
#             self.total_reward += reward
#             return self._get_state(), reward, True

#         dx_start = self.position[0] - self.start[0]
#         dy_start = self.position[1] - self.start[1]
#         dist_to_start = np.sqrt(dx_start**2 + dy_start**2)

#         if dist_to_start > self.prev_dist:
#             reward += dist_to_start
#         else:
#             reward -= 5

#         self.prev_dist = dist_to_start
#         self.total_reward += reward

#         if self.total_reward <= -50:
#             return self._get_state(), reward, True

#         done = self.current_steps >= self.max_steps
#         return self._get_state(), reward, done


# class DQN(nn.Module):
#     def __init__(self, state_dim, action_dim=5):
#         super(DQN, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(state_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, action_dim),
#         )

#     def forward(self, x):
#         return self.fc(x)


# def test_model(env, model_path, state_dim=13, action_dim=5, gif_path="test_result.gif"):
#     model = DQN(state_dim, action_dim).to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()

#     state = env.reset()
#     done = False
#     positions = [env.position]
#     rewards = []
#     frames = []

#     fig, ax = plt.subplots(figsize=(8, 5))
#     ax.set_xlim(0, env.grid_width)
#     ax.set_ylim(0, env.grid_height)
#     ax.set_xticks(range(env.grid_width))
#     ax.set_yticks(range(env.grid_height))
#     ax.invert_yaxis()
#     ax.grid(True)

#     while not done:
#         state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
#         with torch.no_grad():
#             q_values = model(state_tensor)
#         action = q_values.argmax().item()
#         next_state, reward, done = env.step(action)
#         positions.append(env.position)
#         rewards.append(reward)
#         state = next_state

#     for step in range(len(positions)):
#         ax.clear()
#         ax.set_xlim(0, env.grid_width)
#         ax.set_ylim(0, env.grid_height)
#         ax.set_xticks(range(env.grid_width))
#         ax.set_yticks(range(env.grid_height))
#         ax.invert_yaxis()
#         ax.grid(True)

#         for zone in env.danger_zones:
#             x0, y0 = zone["top_left"]
#             h, w = zone["size"]
#             for i in range(h):
#                 for j in range(w):
#                     ax.add_patch(
#                         plt.Rectangle((y0 + j, x0 + i), 1, 1, color="gray", alpha=0.4)
#                     )

#         path = np.array(positions[: step + 1])
#         ax.plot(path[:, 1], path[:, 0], "b-", alpha=0.5)
#         ax.plot(env.start[1], env.start[0], "go", markersize=15, label="Start")
#         for goal in env.goals:
#             ax.plot(goal[1], goal[0], "yx", markersize=12, label="Goal")
#         ax.plot(path[-1, 1], path[-1, 0], "r*", markersize=15, label="Current")

#         cumulative_reward = sum(rewards[: step + 1])
#         ax.set_title(
#             f"Step {step}/{len(positions)-1} | Reward: {cumulative_reward:.1f}"
#         )
#         ax.legend()

#         fig.canvas.draw()
#         image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
#         image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#         frames.append(Image.fromarray(image))

#     frames[0].save(
#         gif_path, save_all=True, append_images=frames[1:], duration=300, loop=0
#     )
#     print(f"✅ GIF saved to: {gif_path}")
#     plt.close(fig)


# if __name__ == "__main__":
#     env = GridEnvironment()
#     test_model(
#         env,
#         model_path="result/best_model_ep7082.pth",
#         gif_path="result/test/episoderesult.gif",
#     )

# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from PIL import Image
# from car_dqn_0415 import GridEnvironment, DQN


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def test_model(env, model_path, state_dim=13, action_dim=5, gif_path="test_result.gif"):
#     model = DQN(state_dim, action_dim).to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()

#     state = env.reset()
#     done = False
#     positions = [env.position]
#     rewards = []
#     frames = []

#     fig, ax = plt.subplots(figsize=(8, 5))
#     ax.set_xlim(0, env.grid_width)
#     ax.set_ylim(0, env.grid_height)
#     ax.set_xticks(range(env.grid_width))
#     ax.set_yticks(range(env.grid_height))
#     ax.invert_yaxis()
#     ax.grid(True)

#     step_count = 0
#     while not done:
#         state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
#         with torch.no_grad():
#             q_values = model(state_tensor)
#         action = q_values.argmax().item()
#         next_state, reward, done = env.step(action)

#         positions.append(env.position)
#         rewards.append(reward)
#         state = next_state
#         step_count += 1

#         # 최대 200 스텝까지 실행
#         if step_count >= env.max_steps:
#             break

#     # 시각화 (GIF 저장)
#     for step in range(len(positions)):
#         ax.clear()
#         ax.set_xlim(0, env.grid_width)
#         ax.set_ylim(0, env.grid_height)
#         ax.set_xticks(range(env.grid_width))
#         ax.set_yticks(range(env.grid_height))
#         ax.invert_yaxis()
#         ax.grid(True)

#         # 위험구역 시각화
#         for zone in env.danger_zones:
#             x0, y0 = zone["top_left"]
#             h, w = zone["size"]
#             for i in range(h):
#                 for j in range(w):
#                     ax.add_patch(
#                         plt.Rectangle((y0 + j, x0 + i), 1, 1, color="gray", alpha=0.5)
#                     )

#         # 경로
#         path = np.array(positions[: step + 1])
#         ax.plot(path[:, 1], path[:, 0], "b-", alpha=0.6)
#         ax.plot(env.start[1], env.start[0], "go", markersize=10, label="Start")
#         # for goal in env.goals:
#         #     ax.plot(goal[1], goal[0], "yx", markersize=10)
#         ax.plot(path[-1, 1], path[-1, 0], "r*", markersize=12, label="Current")

#         cumulative_reward = sum(rewards[: step + 1])
#         ax.set_title(
#             f"Step {step}/{len(positions)-1} | Cumulative Reward: {cumulative_reward:.1f}"
#         )
#         ax.legend()

#         fig.canvas.draw()
#         image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
#         image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#         frames.append(Image.fromarray(image))

#     frames[0].save(
#         gif_path, save_all=True, append_images=frames[1:], duration=300, loop=0
#     )
#     print(f"✅ Test 완료! GIF 저장됨: {gif_path}")
#     plt.close(fig)


# if __name__ == "__main__":
#     env = GridEnvironment()
#     test_model(
#         env,
#         model_path="result/best_model_ep5004.pth",  # 저장된 모델 경로로 바꿔줘
#         gif_path="result/test/episoderesult.gif",
#     )

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import time
import random


# ====================================
# GridEnvironment (앞에서 수정한 코드와 동일)
# ====================================
class GridEnvironment(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, grid_height=30, grid_width=30, max_steps=200):
        super(GridEnvironment, self).__init__()
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(
            5
        )  # 행동: 0~4 (회전각: -90, -45, 0, 45, 90)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(13,), dtype=np.float32
        )

        # 기본 위험 구역 (테스트 시 나중에 바꿀 수 있음)
        self.danger_zones = [
            {"top_left": (1, 5), "size": (2, 2)},
            {"top_left": (2, 8), "size": (2, 2)},
            {"top_left": (2, 11), "size": (2, 2)},
        ]
        self.start = (2, 1)
        self.viewer = None
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
        info = {}
        return obs, info

    def _is_in_square(self, position):
        x, y = position
        for zone in self.danger_zones:
            x0, y0 = zone["top_left"]
            h, w = zone["size"]
            if x0 <= x < x0 + h and y0 <= y < y0 + w:
                return True
        return False

    def _distance_to_danger_zone(self, position, angle):
        dx, dy = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        x, y = position
        max_distance = max(self.grid_width, self.grid_height)
        for d in range(1, max_distance):
            check_x = int(round(x + dx * d))
            check_y = int(round(y + dy * d))
            if 0 <= check_x < self.grid_height and 0 <= check_y < self.grid_width:
                if self._is_in_square((check_x, check_y)):
                    return d
        return max_distance

    def _distance_in_direction(self, angle):
        radians = np.radians(angle)
        x, y = self.position
        dx, dy = np.cos(radians), np.sin(radians)
        t_x = (
            float("inf")
            if dx == 0
            else ((0 - x) / dx if dx < 0 else (self.grid_height - 1 - x) / dx)
        )
        t_y = (
            float("inf")
            if dy == 0
            else ((0 - y) / dy if dy < 0 else (self.grid_width - 1 - y) / dy)
        )
        return min(max(t_x, 0), max(t_y, 0))

    def _get_state(self):
        directions = [-90, -45, 0, 45, 90]
        state_info = [
            self._distance_in_direction(self.direction + angle) / self.grid_width
            for angle in directions
        ]
        danger_info = [
            self._distance_to_danger_zone(self.position, self.direction + angle)
            / self.grid_width
            for angle in directions
        ]
        dx = self.position[0] - self.start[0]
        dy = self.position[1] - self.start[1]
        dist_to_start = np.sqrt(dx**2 + dy**2) / self.grid_width
        current_angle = (np.arctan2(dy, dx) / np.pi) % 1.0
        step_info = self.current_steps / self.max_steps
        return np.array(
            state_info + danger_info + [dist_to_start, current_angle, step_info],
            dtype=np.float32,
        )

    def step(self, action):
        rotations = [-90, -45, 0, 45, 90]
        angle_change = rotations[action]
        self.direction = (self.direction + angle_change) % 360
        if self.direction == -45:
            self.direction = 315

        move_map = {0: (0, 1), 45: (-1, 1), 315: (1, 1), 90: (-1, 0), 270: (1, 0)}
        prev_position = self.position
        dx, dy = move_map.get(self.direction, (0, 0))
        new_x = int(np.clip(self.position[0] + dx, 0, self.grid_height - 1))
        new_y = int(np.clip(self.position[1] + dy, 0, self.grid_width - 1))
        self.position = (new_x, new_y)
        self.current_steps += 1

        if self.position in self.visited_positions:
            self.visited_positions[self.position] += 1
        else:
            self.visited_positions[self.position] = 1

        terminated = False
        if self._is_in_square(self.position):
            reward = -100
            terminated = True
        elif self.position[0] in [0, self.grid_height - 1] or self.position[1] in [
            0,
            self.grid_width - 1,
        ]:
            reward = -20
            terminated = True
        else:
            reward = 1
            if self.visited_positions[self.position] == 1:
                reward += 2
            elif self.visited_positions[self.position] > 1:
                penalty = min(self.visited_positions[self.position] * 1.5, 8)
                reward -= penalty

            if self.position == prev_position:
                reward -= 15

            if angle_change == 0:
                self.direction_score = 0
                reward += 0.5
            else:
                self.direction_score += 1
                reward -= 0.1
            if self.direction_score >= 3:
                reward -= 10

            if self._distance_to_danger_zone(self.position, self.direction) >= 2:
                reward += 1

        truncated = False
        if self.current_steps >= self.max_steps:
            truncated = True
            if len(self.visited_positions) >= 30:
                reward += 50
            else:
                reward -= 20

        self.total_reward += reward
        obs = self._get_state()
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        if self.viewer is None:
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            self.viewer = True
        self.ax.clear()
        self.ax.set_xlim(0, self.grid_width)
        self.ax.set_ylim(0, self.grid_height)
        self.ax.set_xticks(range(self.grid_width + 1))
        self.ax.set_yticks(range(self.grid_height + 1))
        self.ax.grid(True)

        for zone in self.danger_zones:
            x0, y0 = zone["top_left"]
            h, w = zone["size"]
            rect = plt.Rectangle(
                (y0, self.grid_height - x0 - h), w, h, color="red", alpha=0.5
            )
            self.ax.add_patch(rect)

        agent_x = self.position[1] + 0.5
        agent_y = self.grid_height - self.position[0] - 0.5
        self.ax.plot(agent_x, agent_y, "bo", markersize=10)
        self.ax.set_title(
            f"Step: {self.current_steps}, Total Reward: {self.total_reward:.2f}"
        )
        plt.pause(0.001)
        if mode == "rgb_array":
            self.fig.canvas.draw()
            image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return image

    def close(self):
        if self.viewer:
            plt.close(self.fig)
            self.viewer = None


# ====================================
# DQN 모델 정의 (이전과 동일한 구조)
# ====================================
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim=5):
        super(DQN, self).__init__()
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


# ====================================
# 테스트 코드: 저장된 모델 불러와서 평가 진행
# ====================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 환경 생성 및 위험 구역을 30x30 격자 내에 여러 개 임의 생성 (예시로 10개)
    env = GridEnvironment(grid_height=30, grid_width=30, max_steps=200)
    num_danger_zones = 10
    danger_zones = []
    for _ in range(num_danger_zones):
        w = random.randint(1, 4)
        h = random.randint(1, 4)
        x0 = random.randint(0, env.grid_height - h)
        y0 = random.randint(0, env.grid_width - w)
        danger_zones.append({"top_left": (x0, y0), "size": (h, w)})
    env.danger_zones = danger_zones

    print("생성된 위험 구역:")
    for zone in env.danger_zones:
        print(zone)

    # DQN 모델 생성 (상태 차원 13, 행동 차원 5)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = DQN(state_dim, action_dim).to(device)

    # 저장된 모델 로드 (모델 파일 경로에 주의)
    model_path = "result/best_model_ep6172.pth"  # 저장된 파일명
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"모델을 {model_path} 에서 불러왔습니다.")
    except Exception as e:
        print(f"모델 불러오기 실패: {e}")
        exit()

    model.eval()  # 평가 모드로 전환

    # 평가(테스트) 에피소드 실행 (한 에피소드 예시)
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        # 모델을 통해 행동 선택 (greedy policy)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state_tensor)
        action = q_values.argmax().item()

        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        env.render()
        time.sleep(0.1)

    print("테스트 완료.")
    print("최종 Total Reward:", env.total_reward)
    env.close()
    plt.show()
