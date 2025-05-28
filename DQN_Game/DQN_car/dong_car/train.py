import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches  # 장애물 그리기를 위해 추가
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import gymnasium as gym
from gymnasium import spaces


# -----------------------------
# 1. 사용자 정의 환경 정의 (수정됨)
# -----------------------------
class ObstacleAvoidanceEnv(gym.Env):
    """
    직사각형 격자 맵 환경:
     - 에이전트는 맵 경계와 내부에 랜덤하게 배치된 직사각형 장애물을 피해 움직임.
     - 상태: 에이전트 위치를 기준으로 5방향의 장애물 또는 경계까지 거리 (센서 값)
     - 행동: 0 - 전진, 1 - 좌측 45도 회전 후 전진, 2 - 우측 45도 회전 후 전진
     - 목표: 충돌 없이 최대한 오래 이동
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        map_width=20,
        map_height=10,
        num_obstacles=3,
        min_obstacle_size=1.0,
        max_obstacle_size=2.0,
        max_steps=100,
    ):  # max_steps 증가 고려
        super(ObstacleAvoidanceEnv, self).__init__()

        self.map_width = map_width
        self.map_height = map_height
        self.num_obstacles = num_obstacles
        self.min_obstacle_size = min_obstacle_size
        self.max_obstacle_size = max_obstacle_size
        self.max_steps = max_steps

        # 센서 설정: 상대각도(도 단위) 0, +45, -45, +90, -90
        self.sensor_angles = [0, 45, -45, 90, -90]
        self.sensor_max_range = math.sqrt(
            map_width**2 + map_height**2
        )  # 맵 대각선 길이로 최대 범위 설정
        self.sensor_ray_step = 0.2  # 센서 감지 정밀도

        # 에이전트가 한 번에 이동하는 거리
        self.step_size = 1.0

        # 행동 공간: 0 - 전진, 1 - 좌측 45도 회전, 2 - 우측 45도 회전
        self.action_space = spaces.Discrete(3)
        # 관측 공간: 5개 센서의 거리값 (0 ~ sensor_max_range)
        self.observation_space = spaces.Box(
            low=0.0, high=self.sensor_max_range, shape=(5,), dtype=np.float32
        )

        self.obstacles = []  # 장애물 정보를 담을 리스트 (x, y, width, height)
        self.agent_pos = np.zeros(2, dtype=np.float32)
        self.agent_heading = 0.0  # 도 단위
        self.num_steps = 0
        self.done = False

        # 시각화를 위한 변수
        self.fig = None
        self.ax = None

        # 초기화 시 reset 호출 보장 (gym.Env 권장사항)
        # observation, info = self.reset() # reset()이 observation, info 튜플을 반환하도록 수정 필요
        # 여기서는 일단 reset 내부에서 observation 계산만 하도록 유지

    def _generate_obstacles(self):
        """맵 내부에 겹치지 않도록 랜덤 장애물 생성"""
        self.obstacles = []
        attempts = 0
        max_attempts = self.num_obstacles * 20  # 장애물 생성 시도 횟수 제한

        while len(self.obstacles) < self.num_obstacles and attempts < max_attempts:
            attempts += 1
            # 랜덤 크기
            obs_w = random.uniform(self.min_obstacle_size, self.max_obstacle_size)
            obs_h = random.uniform(self.min_obstacle_size, self.max_obstacle_size)
            # 랜덤 위치 (맵 내부에 완전히 들어오도록)
            obs_x = random.uniform(0, self.map_width - obs_w)
            obs_y = random.uniform(0, self.map_height - obs_h)

            new_obstacle = (obs_x, obs_y, obs_w, obs_h)

            # 다른 장애물과 겹치는지 확인 (단순 AABB 충돌 검사)
            collision = False
            for ox, oy, ow, oh in self.obstacles:
                if (
                    obs_x < ox + ow
                    and obs_x + obs_w > ox
                    and obs_y < oy + oh
                    and obs_y + obs_h > oy
                ):
                    collision = True
                    break

            if not collision:
                self.obstacles.append(new_obstacle)

        if len(self.obstacles) < self.num_obstacles:
            print(
                f"Warning: Could only generate {len(self.obstacles)} non-overlapping obstacles."
            )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Gymnasium API 준수

        # 장애물 재생성
        self._generate_obstacles()

        # 에이전트를 빈 공간에 배치 (장애물 및 경계와 겹치지 않도록)
        valid_position = False
        while not valid_position:
            self.agent_pos = np.array(
                [
                    random.uniform(0.5, self.map_width - 0.5),  # 경계에서 약간 안쪽
                    random.uniform(0.5, self.map_height - 0.5),
                ],
                dtype=np.float32,
            )
            if not self._is_collision(self.agent_pos):
                valid_position = True

        # 에이전트의 초기 heading은 무작위 선택 (단위: 도)
        self.agent_heading = random.uniform(
            0, 360
        )  # 더 부드러운 시작을 위해 연속적인 각도 사용 가능

        self.num_steps = 0
        self.done = False

        observation = self._get_observation()
        info = {}  # 추가 정보 (필요시)

        # 시각화 초기화 (필요시)
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

        return observation, info  # Gymnasium API 준수

    def step(self, action):
        if self.done:
            # Gymnasium 최신 버전에서는 에피소드 종료 후 step 호출 시 경고 발생
            # return self._get_observation(), 0, self.done, False, {} # 이전 방식
            # 환경 문서를 참조하여 적절한 반환값 결정 필요 (보통 마지막 상태 반환)
            # 여기서는 간단히 마지막 상태와 0 보상 반환 가정
            obs = self._get_observation()
            return obs, 0.0, self.done, False, {}  # truncated=False

        # 행동에 따른 회전 및 보상/페널티 설정
        turn_penalty = -3.0  # 회전 시 페널티 (조정 가능)
        straight_bonus = 2.0  # 직진 시 보너스 (조정 가능)
        action_reward = 0.0

        if action == 1:  # 좌회전
            self.agent_heading = (self.agent_heading + 45) % 360
            action_reward = turn_penalty
        elif action == 2:  # 우회전
            self.agent_heading = (self.agent_heading - 45) % 360
            action_reward = turn_penalty
        else:  # 직진 (action == 0)
            action_reward = straight_bonus

        # 에이전트 이동 (step_size 만큼)
        rad = math.radians(self.agent_heading)
        delta = np.array([math.cos(rad), math.sin(rad)]) * self.step_size
        new_pos = self.agent_pos + delta

        # 이동 후 충돌 체크
        collision = self._is_collision(new_pos)
        truncated = False  # 시간 초과 외의 이유로 종료되지 않음

        if collision:
            reward = -50.0  # 충돌 시 큰 페널티
            self.done = True
        else:
            self.agent_pos = new_pos
            # 생존 보상 + 행동 보상 (직진 보너스 또는 회전 페널티)
            reward = 1.0 + action_reward

        self.num_steps += 1
        if self.num_steps >= self.max_steps:
            self.done = True
            truncated = True  # 시간 초과로 인한 종료 표시
            if (
                not collision
            ):  # 시간 초과 시 충돌이 아니었다면, 마지막 스텝 보상에서 페널티/보너스 제외 가능 (선택 사항)
                reward = 1.0  # 예: 시간 초과 자체는 페널티가 아님

        observation = self._get_observation()
        info = {}

        return (
            observation,
            reward,
            self.done,
            truncated,
            info,
        )  # Gymnasium API 준수 (obs, rew, terminated, truncated, info)

    def _is_collision(self, pos):
        # 맵 경계 체크
        if not (0 <= pos[0] < self.map_width and 0 <= pos[1] < self.map_height):
            return True

        # 각 장애물과 충돌 체크 (AABB 충돌 검사)
        for obs_x, obs_y, obs_w, obs_h in self.obstacles:
            if obs_x <= pos[0] < obs_x + obs_w and obs_y <= pos[1] < obs_y + obs_h:
                return True
        return False

    def _get_observation(self):
        # 에이전트 위치 기준, 각 센서 방향으로 경계 또는 장애물까지의 거리 계산
        sensor_readings = []
        for angle_offset in self.sensor_angles:
            sensor_angle_deg = (self.agent_heading + angle_offset) % 360
            sensor_angle_rad = math.radians(sensor_angle_deg)
            direction = np.array(
                [math.cos(sensor_angle_rad), math.sin(sensor_angle_rad)]
            )

            distance = 0.0
            hit = False
            # 센서 최대 범위까지 조금씩 전진하며 충돌 검사
            while distance < self.sensor_max_range:
                test_pos = self.agent_pos + direction * distance
                if self._is_collision(test_pos):
                    hit = True
                    break
                distance += self.sensor_ray_step  # 정밀도에 따라 조절

            # 실제 충돌 거리 또는 최대 범위 중 작은 값 사용
            sensor_readings.append(min(distance, self.sensor_max_range))

        return np.array(sensor_readings, dtype=np.float32)

    def render(self, mode="human"):
        # 시각화 설정 (최초 호출 시)
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))  # 크기 조절 가능

        self.ax.clear()  # 이전 프레임 지우기

        # 맵 경계 설정 및 배경
        self.ax.set_xlim(0, self.map_width)
        self.ax.set_ylim(0, self.map_height)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_facecolor("lightgray")  # 배경색

        # 장애물 그리기
        for x, y, w, h in self.obstacles:
            rect = patches.Rectangle(
                (x, y), w, h, linewidth=1, edgecolor="black", facecolor="dimgray"
            )
            self.ax.add_patch(rect)

        # 에이전트 그리기 (파란색 원)
        self.ax.plot(
            self.agent_pos[0], self.agent_pos[1], "bo", markersize=8, label="Agent"
        )

        # # 센서 광선 그리기 (빨간 점선)
        # obs_for_render = self._get_observation()  # 현재 센서 값 가져오기
        # for i, angle_offset in enumerate(self.sensor_angles):
        #     sensor_angle_deg = (self.agent_heading + angle_offset) % 360
        #     sensor_angle_rad = math.radians(sensor_angle_deg)
        #     distance = obs_for_render[i]  # 계산된 거리 사용
        #     end_point = (
        #         self.agent_pos
        #         + np.array([math.cos(sensor_angle_rad), math.sin(sensor_angle_rad)])
        #         * distance
        #     )
        #     self.ax.plot(
        #         [self.agent_pos[0], end_point[0]],
        #         [self.agent_pos[1], end_point[1]],
        #         "r--",
        #         linewidth=1,
        #         label="Sensor Ray" if i == 0 else "",
        #     )  # 범례 중복 방지

        # 제목 및 레이아웃
        self.ax.set_title(f"Obstacle Avoidance | Step: {self.num_steps}")
        # self.ax.legend() # 범례가 너무 많아질 수 있으므로 주석 처리
        plt.tight_layout()

        if mode == "human":
            plt.pause(0.01)  # 잠시 멈춰서 보여줌
            self.fig.canvas.draw()  # 캔버스 업데이트
        elif mode == "rgb_array":
            self.fig.canvas.draw()
            renderer = self.fig.canvas.get_renderer()
            img = np.frombuffer(renderer.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            # plt.close(self.fig) # rgb_array 모드에서는 계속 사용해야 할 수 있으므로 닫지 않음 (필요시 관리)
            return img

    def close(self):
        # 시각화 창 닫기
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


# -----------------------------
# 2. DQN 에이전트 구현 (변경 없음)
# -----------------------------
class DQN(nn.Module):
    def __init__(self, input_dim=5, output_dim=3):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        state_dim=5,
        action_dim=3,
        lr=1e-4,
        gamma=0.99,  # 학습률 조정 가능성
        batch_size=64,
        buffer_size=50000,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=20000,
    ):  # 버퍼 크기 증가 고려, epsilon 파라미터 추가
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(input_dim=state_dim, output_dim=action_dim).to(
            self.device
        )
        self.target_net = DQN(input_dim=state_dim, output_dim=action_dim).to(
            self.device
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=buffer_size)
        self.steps_done = 0

        # Epsilon 값 조정 (더 긴 탐험 기간 또는 다른 스케줄 고려)
        self.epsilon_start = epsilon_start  # 파라미터 사용
        self.epsilon_end = epsilon_end  # 파라미터 사용
        self.epsilon_decay = epsilon_decay  # 파라미터 사용

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        # Epsilon 계산 시 self.steps_done 사용 (에피소드 단위 대신 스텝 단위)
        eps_threshold = self.epsilon_end + (
            self.epsilon_start - self.epsilon_end
        ) * math.exp(-1.0 * self.steps_done / self.epsilon_decay)
        self.steps_done += 1  # 스텝마다 증가

        if random.random() < eps_threshold:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()

    def store_transition(
        self, state, action, reward, next_state, terminated, truncated
    ):
        # Gymnasium API 변경에 맞춰 done 대신 terminated와 truncated 사용 고려
        # 여기서는 done 플래그 하나로 합쳐서 저장 (terminated or truncated)
        done = terminated or truncated
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return 0.0  # Loss 반환 시 초기값

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        # dones를 bool 대신 float으로 사용 (계산을 위해)
        dones_mask = (
            torch.FloatTensor(np.array(dones).astype(float))
            .unsqueeze(1)
            .to(self.device)
        )

        # 현재 상태에서 실제 취한 행동의 Q 값
        curr_Q = self.policy_net(states).gather(1, actions)

        # 다음 상태의 최대 Q 값 (타겟 네트워크 사용)
        # detach() 하여 그래디언트 흐름 차단
        with torch.no_grad():
            next_max_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            # 다음 상태가 종료 상태(done=True)이면 next_Q는 0이 되어야 함
            expected_Q = rewards + (1 - dones_mask) * self.gamma * next_max_q

        # Loss 계산 (MSE)
        loss = nn.MSELoss()(curr_Q, expected_Q)  # expected_Q는 detach 되어 있음

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # 그래디언트 클리핑 (옵션)
        # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()  # loss 값 반환 (모니터링용)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


# ---------
# 3. 학습 (변경 없음, 단 파라미터 조정 가능)
# ---------
def train_dqn(
    num_episodes=8000,
    max_steps=100,
    target_update_interval=10,
    save_model=True,
    render_interval=100,
):  # 렌더링 간격 추가
    # 환경 생성 시 파라미터 전달 가능
    env = ObstacleAvoidanceEnv(
        map_width=20, map_height=10, num_obstacles=3, max_steps=max_steps
    )
    # DQNAgent 생성 시 epsilon_decay 전달
    agent = DQNAgent(
        buffer_size=100000, epsilon_decay=30000
    )  # 환경 복잡도에 맞춰 파라미터 조정

    episode_rewards = []
    episode_losses = []  # 에피소드 평균 로스 추적
    total_steps = 0

    # tqdm progress bar를 에피소드 루프에 적용
    with tqdm(total=num_episodes, desc="Training DQN Episodes") as pbar:
        for episode in range(num_episodes):
            state, _ = env.reset()  # info 무시
            episode_reward = 0
            step_losses = []
            terminated = False
            truncated = False

            while not (terminated or truncated):
                # 렌더링 (지정된 간격마다)
                # if (episode + 1) % render_interval == 0:
                #     env.render(mode='human') # 이 줄을 주석 처리하여 렌더링 비활성화

                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(
                    action
                )  # info 무시
                agent.store_transition(
                    state, action, reward, next_state, terminated, truncated
                )

                loss = agent.update()  # update는 batch size 이상 쌓여야 실행됨
                if loss > 0:  # update가 실행되었을 때만 loss 기록
                    step_losses.append(loss)

                state = next_state
                episode_reward += reward
                total_steps += 1

                if terminated or truncated:
                    break

            episode_rewards.append(episode_reward)
            avg_loss = np.mean(step_losses) if step_losses else 0
            episode_losses.append(avg_loss)

            # 타겟 네트워크 업데이트 (에피소드 기준)
            if (episode + 1) % target_update_interval == 0:
                agent.update_target()
                # print(f"\nTarget network updated at episode {episode+1}")

            # 진행 상황 업데이트
            pbar.set_description(
                f"Ep {episode+1}/{num_episodes} | Steps: {env.num_steps} | Reward: {episode_reward:.1f} | Avg Loss: {avg_loss:.4f} | Epsilon: {agent.epsilon_end + (agent.epsilon_start - agent.epsilon_end) * math.exp(-1. * agent.steps_done / agent.epsilon_decay):.3f}"
            )
            pbar.update(1)

    env.close()  # 학습 완료 후 환경 닫기

    if save_model:
        torch.save(agent.policy_net.state_dict(), "dqn_obstacle_avoidance_weights.pth")
        print("\n모델 가중치를 dqn_obstacle_avoidance_weights.pth 에 저장하였습니다.")

    # 학습 결과 시각화 (옵션)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(episode_losses)
    plt.title("Average Episode Loss")
    plt.xlabel("Episode")
    plt.ylabel("Average Loss")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return agent, episode_rewards


if __name__ == "__main__":
    # 학습 실행 (에피소드 수, 렌더링 간격 등 조절 가능)
    agent, rewards = train_dqn(num_episodes=8000, max_steps=100, render_interval=50)
