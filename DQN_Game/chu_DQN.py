import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import time

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

# -------------------------------
# [전역 변수 설정]
# -------------------------------
# 격자 크기
n = 5

# 시작점 좌표 (row, col)
START_ROW = 4
START_COL = 0

# 목표점(끝점) 좌표 (row, col)
GOAL_ROW = 0
GOAL_COL = 4

# 최대 스텝 수
MAX_STEPS_PER_EPISODE = 80

# -------------------------------
# 1. Gymnasium 커스텀 환경 정의
# -------------------------------
class GridWorldEnv(gym.Env):
    """
    Gymnasium 방식의 커스텀 환경.
    - 상태: 0 ~ (n*n - 1) 범위의 정수 (row * n + col)
    - 행동: 0=상, 1=우, 2=하, 3=좌
    - 보상:
      * 목표점에 도달하면 +1
      * 그 외에는 0
    - 에피소드 종료:
      * 목표점 도달
      * 최대 스텝 도달
    - 렌더링:
      * Pygame을 이용해 격자, 시작점, 목표점, 에이전트 위치를 그립니다.
    """

    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(self, render_mode=None):
        super().__init__()

        self.n = n
        self.start = (START_ROW, START_COL)
        self.goal = (GOAL_ROW, GOAL_COL)
        self.max_steps = MAX_STEPS_PER_EPISODE

        # 상태공간: Discrete(n*n)
        self.observation_space = spaces.Discrete(self.n * self.n)
        # 행동공간: Discrete(4) = 상/우/하/좌
        self.action_space = spaces.Discrete(4)

        # 에피소드 내 상태
        self.current_pos = None
        self.steps_done = 0

        # 렌더링 모드 / Pygame 관련
        self.render_mode = render_mode
        self.screen = None
        self.cell_size = 60

    def _pos_to_state(self, row, col):
        return row * self.n + col

    def _state_to_pos(self, s):
        return divmod(s, self.n)

    def reset(self, seed=None, options=None):
        """
        에피소드 시작 시 호출
        - 초기 상태(= 시작점)로 이동
        - 반환: (관측값, info)
        """
        super().reset(seed=seed)
        self.current_pos = self.start
        self.steps_done = 0

        state = self._pos_to_state(*self.current_pos)
        return state, {}

    def step(self, action):
        """
        한 스텝 진행
        - 입력: action(0=상,1=우,2=하,3=좌)
        - 출력: (다음 상태, 보상, 종료여부, 트렁케이션여부, info)
        """
        r, c = self.current_pos

        if action == 0:   # 상
            nr, nc = r - 1, c
        elif action == 1: # 우
            nr, nc = r, c + 1
        elif action == 2: # 하
            nr, nc = r + 1, c
        elif action == 3: # 좌
            nr, nc = r, c - 1
        else:
            nr, nc = r, c

        # 격자 범위 검사
        if 0 <= nr < self.n and 0 <= nc < self.n:
            self.current_pos = (nr, nc)

        self.steps_done += 1

        reward = 0.0
        done = False
        truncated = False

        if self.current_pos == self.goal:
            reward = 1.0
            done = True
        elif self.steps_done >= self.max_steps:
            done = True

        next_state = self._pos_to_state(*self.current_pos)

        return next_state, reward, done, truncated, {}

    def render(self):
        """
        render_mode='human'일 때만 Pygame으로 화면 표시.
        """
        if self.render_mode != "human":
            return

        if self.screen is None:
            pygame.init()
            screen_size = (self.n * self.cell_size, self.n * self.cell_size)
            self.screen = pygame.display.set_mode(screen_size)
            pygame.display.set_caption("GridWorld - DQN (Gymnasium)")

        # 이벤트 처리(창 닫기 등)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        RED   = (255, 0, 0)
        BLUE  = (0, 0, 255)
        GREEN = (0, 200, 0)

        self.screen.fill(WHITE)

        # 격자 선
        for i in range(self.n + 1):
            pygame.draw.line(
                self.screen, BLACK,
                (i * self.cell_size, 0),
                (i * self.cell_size, self.n * self.cell_size)
            )
            pygame.draw.line(
                self.screen, BLACK,
                (0, i * self.cell_size),
                (self.n * self.cell_size, i * self.cell_size)
            )

        # 좌표
        sr, sc = self.start
        gr, gc = self.goal
        ar, ac = self.current_pos

        # 시작점(파랑 사각형)
        start_rect = pygame.Rect(
            sc*self.cell_size+5, sr*self.cell_size+5,
            self.cell_size-10, self.cell_size-10
        )
        pygame.draw.rect(self.screen, BLUE, start_rect)

        # 목표점(초록 사각형)
        goal_rect = pygame.Rect(
            gc*self.cell_size+5, gr*self.cell_size+5,
            self.cell_size-10, self.cell_size-10
        )
        pygame.draw.rect(self.screen, GREEN, goal_rect)

        # 에이전트(빨강 원)
        agent_rect = pygame.Rect(
            ac*self.cell_size+10, ar*self.cell_size+10,
            self.cell_size-20, self.cell_size-20
        )
        pygame.draw.ellipse(self.screen, RED, agent_rect)

        pygame.display.flip()

# -------------------------------
# 2. 학습(Train) - 화면 표시 없음
# -------------------------------
def train_dqn(total_timesteps=20000):
    """
    DQN 모델을 학습하는 함수
    - total_timesteps: 학습할 전체 스텝 수
    (학습 시에는 render_mode=None → 화면 표시 안 함)
    """
    env = GridWorldEnv(render_mode=None)  # 화면 OFF
    vec_env = DummyVecEnv([lambda: env])

    # DQN 모델
    model = DQN(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=5000,
        exploration_fraction=0.3,  # epsilon 스케줄 비슷
        gamma=0.80,
        target_update_interval=500,
        device="cuda"
    )

    model.learn(total_timesteps=total_timesteps)
    model.save("dqn_gridworld")

    print("학습 완료, 모델 저장: dqn_gridworld.zip")


# -------------------------------
# 3. 테스트(Test) - 화면 표시, 움직임 느리게
# -------------------------------
def test_dqn(episodes=5, delay=0.2):
    """
    학습된 모델로 에피소드 테스트
    - render_mode='human' → Pygame 화면 표시
    - delay: 에이전트 한 스텝마다 대기 시간(초)
    """
    model = DQN.load("dqn_gridworld")

    env = GridWorldEnv(render_mode="human")

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # 렌더링
            env.render()

            # 모델에 의해 행동 선택(탐욕적)
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            # 속도 조절
            time.sleep(delay)

        print(f"[에피소드 {ep+1}] 보상 합계: {total_reward:.2f}")

    print("테스트 완료. 3초 후 종료됩니다.")
    time.sleep(3)
    pygame.quit()


# -------------------------------
# 메인 실행부
# -------------------------------
if __name__ == "__main__":
    # 1) 학습 (화면 표시 X)
    train_dqn(total_timesteps=10000)  # 예) 10,000 스텝 학습

    # 2) 학습된 모델 테스트 (화면 표시 ON, 0.2초 지연)
    test_dqn(episodes=5, delay=0.2)
