import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches  # 장애물 그리기를 위해 추가
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces


# -----------------------------
# 1. 환경 정의 (train.py와 동일하게 수정)
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
        num_obstacles=5,
        min_obstacle_size=1.0,
        max_obstacle_size=2.0,
        max_steps=200,
    ):  # max_steps 증가 고려
        super(ObstacleAvoidanceEnv, self).__init__()
        self.goal_pos = np.array([19.0, 9.0], dtype=np.float32)

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
        # self.done = False # Gymnasium API에서는 terminated/truncated 사용
        self.terminated = False
        self.truncated = False

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

        # 장애물 재생성 (num_obstacles=0 이면 빈 리스트)
        self._generate_obstacles()

        # 시작점 고정 (1,1), 초기 방향 0°
        self.agent_pos = np.array([1.0, 1.0], dtype=np.float32)
        self.agent_heading = 0.0

        self.num_steps = 0
        # self.done = False
        self.terminated = False
        self.truncated = False

        observation = self._get_observation()
        info = {}  # 추가 정보 (필요시)

        # 시각화 초기화 (필요시)
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

        return observation, info  # Gymnasium API 준수

    def step(self, action):
        # if self.done: # 이전 방식
        if self.terminated or self.truncated:
            # Gymnasium 최신 버전에서는 에피소드 종료 후 step 호출 시 경고 발생
            # return self._get_observation(), 0, self.done, False, {} # 이전 방식
            # 환경 문서를 참조하여 적절한 반환값 결정 필요 (보통 마지막 상태 반환)
            # 여기서는 간단히 마지막 상태와 0 보상 반환 가정
            obs = self._get_observation()
            # return obs, 0.0, self.done, False, {} # truncated=False # 이전 방식
            return obs, 0.0, self.terminated, self.truncated, {}

        # 행동에 따른 회전 및 보상/페널티 설정
        turn_penalty = -0.1  # 회전 시 페널티 (조정 가능)
        straight_bonus = 0.2  # 직진 시 보너스 (조정 가능)
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
        # truncated = False # 시간 초과 외의 이유로 종료되지 않음 # step 시작 시 초기화

        if collision:
            reward = -30.0  # 충돌 시 큰 페널티
            # self.done = True
            self.terminated = True
        else:
            self.agent_pos = new_pos
            # 생존 보상 + 행동 보상 (직진 보너스 또는 회전 페널티)
            reward = 1.0 + action_reward
            # 목표 지점 도달 체크
            if np.allclose(self.agent_pos, self.goal_pos, atol=self.step_size * 0.1):
                reward = 100.0
                self.terminated = True

        # 2) 목표점까지의 거리 계산
        dist_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)

        # 4) 정확히 목표점에 도달했을 때 최종 보상 및 종료
        goal_threshold = self.step_size * 0.3  # (≈0.1 셀 이내)
        if dist_to_goal <= goal_threshold:
            reward = 300.0  # goal bonus
            # self.done = True
            self.terminated = True
            print("도~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~착!")

        self.num_steps += 1
        if self.num_steps >= self.max_steps:
            # self.done = True
            self.truncated = True  # 시간 초과로 인한 종료 표시
            if (
                not collision
            ):  # 시간 초과 시 충돌이 아니었다면, 마지막 스텝 보상에서 페널티/보너스 제외 가능 (선택 사항)
                reward = 1.0  # 예: 시간 초과 자체는 페널티가 아님

        observation = self._get_observation()
        info = {}

        # return observation, reward, self.done, truncated, info # Gymnasium API 준수 (obs, rew, terminated, truncated, info)
        # return observation, reward, self.terminated, self.truncated, info
        return observation, reward, self.terminated, self.truncated, info

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
        # ── 통합된 render() ──
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            if mode == "human":
                plt.ion()

        # 1) 프레임 초기화
        self.ax.clear()

        # 2) 그리드
        self.ax.set_xlim(0, self.map_width)
        self.ax.set_ylim(0, self.map_height)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_facecolor("lightgray")
        self.ax.set_xticks(np.arange(0, self.map_width + 1, 1))
        self.ax.set_yticks(np.arange(0, self.map_height + 1, 1))
        self.ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.7)

        # 3) 장애물
        for x, y, w, h in self.obstacles:
            self.ax.add_patch(
                patches.Rectangle((x, y), w, h, edgecolor="black", facecolor="dimgray")
            )

        # 4) 에이전트 (예: 빨간 사각형)
        self.ax.plot(
            self.agent_pos[0], self.agent_pos[1], "rs", markersize=8, label="Agent"
        )

        # 센서 광선 그리기 (빨간 점선)
        obs_for_render = self._get_observation()  # 현재 센서 값 가져오기
        for i, angle_offset in enumerate(self.sensor_angles):
            sensor_angle_deg = (self.agent_heading + angle_offset) % 360
            sensor_angle_rad = math.radians(sensor_angle_deg)
            distance = obs_for_render[i]  # 계산된 거리 사용
            end_point = (
                self.agent_pos
                + np.array([math.cos(sensor_angle_rad), math.sin(sensor_angle_rad)])
                * distance
            )
            self.ax.plot(
                [self.agent_pos[0], end_point[0]],
                [self.agent_pos[1], end_point[1]],
                "r--",
                linewidth=1,
                label="Sensor Ray" if i == 0 else "",
            )  # 범례 중복 방지

        # 5) ★ 목표 지점 그리기
        gx, gy = self.goal_pos
        self.ax.scatter(
            gx,
            gy,
            marker="*",
            s=200,
            c="gold",
            edgecolors="black",
            linewidths=1,
            zorder=10,
            label="Goal",
        )

        # 6) 타이틀 및 화면 갱신
        self.ax.set_title(f"Step: {self.num_steps}")
        plt.tight_layout()
        if mode == "human":
            plt.pause(0.01)
            self.fig.canvas.draw()
        elif mode == "rgb_array":
            self.fig.canvas.draw()
            w, h = self.fig.canvas.get_width_height()
            buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            return buf.reshape((h, w, 3))

    def close(self):
        # 시각화 창 닫기
        if self.fig is not None:
            plt.ioff()  # 인터랙티브 모드 비활성화
            plt.close(self.fig)
            self.fig = None
            self.ax = None


# -----------------------------
# 2. DQN 모델 및 에이전트 정의 (train.py와 동일한 구조 사용)
# -----------------------------
class DQN(nn.Module):
    def __init__(self, input_dim=5, output_dim=3):
        super(DQN, self).__init__()
        # train.py의 네트워크 구조와 동일하게 맞춰야 함
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),  # train.py와 동일하게 128
            nn.ReLU(),
            nn.Linear(128, 128),  # train.py와 동일하게 128
            nn.ReLU(),
            nn.Linear(128, 64),  # train.py와 동일하게 64
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim=5, action_dim=3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # DQN 클래스를 사용하여 policy_net 초기화
        self.policy_net = DQN(input_dim=state_dim, output_dim=action_dim).to(
            self.device
        )
        # 추론 시에는 eval 모드로 설정
        self.policy_net.eval()

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            # 가장 높은 Q 값을 가진 행동 선택
            return q_values.max(1)[1].item()


# -----------------------------
# 3. 애니메이션 방식 시뮬레이션 (simulate_live) - 수정됨
# -----------------------------
def simulate_live(agent, max_steps=500):
    # train.py와 동일한 환경 파라미터 사용
    # 장애물 랜덤 대신 사용자 직접 입력
    env = ObstacleAvoidanceEnv(
        map_width=20, map_height=10, num_obstacles=5, max_steps=max_steps
    )

    # 2) 초기화 → 빈 obstacles 리스트 확보
    _, _ = env.reset()
    env.obstacles.clear()

    # # 3) 사용자 장애물 입력
    # num_obs = int(input("장애물 개수를 입력하세요: "))
    # for i in range(num_obs):
    #     x, y, w, h = map(float, input(f"장애물 {i+1} (x y width height): ").split())
    #     env.obstacles.append((x, y, w, h))

    # 3) 하드코딩된 장애물 설정
    #    (x, y, width, height) 형태로 원하는 개수만큼 추가하세요.
    hardcoded_obstacles = [
        (3.0, 4.0, 2.0, 1.0),
        (8.0, 2.0, 2.0, 2.0),
        (5.0, 2.0, 1.0, 1.0),
        (12.0, 6.0, 2.0, 2.0),
        (15.0, 6.0, 1.0, 1.0),
        (13.0, 6.0, 1.0, 1.0),
        (10.0, 6.0, 1.0, 1.0),
        # (17.0, 8.0, 1.0, 1.0),
        # … 추가 가능 …
    ]
    env.obstacles.extend(hardcoded_obstacles)

    # 3) 시작점(1,1) 고정, 방향 초기화
    env.agent_pos = np.array([1.0, 1.0], dtype=np.float32)
    env.agent_heading = 0.0

    # 4) 초기 관측값 재계산
    state = env._get_observation()
    # 초기 화면 한 번 띄우기
    env.render(mode="human")
    total_reward = 0.0
    terminated = False
    truncated = False

    # plt.ion() 등은 env.render() 내부에서 처리됨

    # while not (terminated or truncated):
    #     action = agent.select_action(state)
    #     # Gymnasium API에 맞게 반환값 받기
    #     next_state, reward, terminated, truncated, _ = env.step(
    #         action
    #     )  # info는 사용하지 않음
    #     total_reward += reward

    #     # 환경의 render 메서드 사용
    #     env.render(mode="human")

    #     state = next_state
    #     # done 플래그 대신 terminated와 truncated 사용
    #     if terminated or truncated:
    #         break

    # # plt.ioff() 등은 env.close() 내부에서 처리됨
    # env.close()  # 시뮬레이션 종료 후 환경 자원 해제
    # print(f"시뮬레이션 종료 - 총 보상: {total_reward}")

    # ── 주 루프 ──
    while not (terminated or truncated):
        # 1) 행동 선택
        action = agent.select_action(state)

        # 2) 환경 한 단계 진행
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        # 3) 화면 갱신
        env.render(mode="human")

    # ── 종료 처리 ──
    env.close()
    print(f"시뮬레이션 종료 - 총 보상: {total_reward}")


# -----------------------------
# 4. 메인 실행부 (로컬 환경에서 실행)
# -----------------------------
if __name__ == "__main__":
    # DQNAgent 초기화 (state_dim, action_dim 확인)
    agent = DQNAgent(state_dim=5, action_dim=3)
    # train.py에서 저장한 가중치 파일명 사용
    weights_path = "dqn_obstacle_avoidance_weights_skrr.pth"
    try:
        # map_location을 사용하여 CPU/GPU 간 호환성 확보
        agent.policy_net.load_state_dict(
            torch.load(weights_path, map_location=agent.device)
        )
        print(f"가중치 파일 '{weights_path}'을(를) 성공적으로 로드하였습니다.")
    except FileNotFoundError:
        print(f"오류: 가중치 파일 '{weights_path}'을(를) 찾을 수 없습니다.")
        exit()
    except Exception as e:
        print(f"가중치 파일 로드 중 오류 발생: {e}")
        exit()

    # 시뮬레이션 실행 (max_steps는 환경 설정과 맞춤)
    simulate_live(agent, max_steps=200)
