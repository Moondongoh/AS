import pygame
import random
import numpy as np

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

# 학습 파라미터
alpha = 0.1  # 학습률
gamma = 0.80  # 감가율
epsilon = 0.5  # 탐험 확률

# 에피소드 관련
episode_count = 15  # 학습할 에피소드 수
steps_each_ep = 80  # 한 에피소드 최대 스텝


class GridWorld:
    """
    전역 변수로 정해진 n, (START_ROW, START_COL), (GOAL_ROW, GOAL_COL)를 사용해
    n x n 격자 환경을 구성.

    - 상태(state): (row, col)
    - 행동(action): 0=상, 1=우, 2=하, 3=좌
    - 보상(reward): 목표점 도달 시 +1, 그 외 0
    - 에피소드 종료 조건: 목표점 도달 or 스텝 초과
    """

    def __init__(self, max_steps=50):
        global n, START_ROW, START_COL, GOAL_ROW, GOAL_COL
        self.n = n
        self.max_steps = max_steps

        # 전역에서 설정된 시작점과 목표점
        self.start = (START_ROW, START_COL)
        self.goal = (GOAL_ROW, GOAL_COL)

        # 현재 상태 (매 에피소드 시작 때 reset으로 갱신)
        self.current_pos = None
        self.steps_done = 0

        print(f"[환경 초기화] n={self.n}, start={self.start}, goal={self.goal}")

    def reset(self):
        """에피소드 시작 시 에이전트의 위치를 start로 초기화"""
        self.current_pos = self.start
        self.steps_done = 0
        return self._pos_to_state(*self.current_pos)

    def step(self, action):
        """action에 따라 에이전트 이동 후 (next_state, reward, done) 반환"""
        r, c = self.current_pos
        if action == 0:  # 상
            nr, nc = r - 1, c
        elif action == 1:  # 우
            nr, nc = r, c + 1
        elif action == 2:  # 하
            nr, nc = r + 1, c
        elif action == 3:  # 좌
            nr, nc = r, c - 1
        else:
            nr, nc = r, c

        # 격자 범위를 벗어나면 이동 무시
        if 0 <= nr < self.n and 0 <= nc < self.n:
            self.current_pos = (nr, nc)

        self.steps_done += 1

        # 보상/종료 판단
        reward = 0.0
        done = False
        if self.current_pos == self.goal:
            reward = 1.0
            done = True
        elif self.steps_done >= self.max_steps:
            done = True

        next_state = self._pos_to_state(*self.current_pos)
        return next_state, reward, done

    def _pos_to_state(self, row, col):
        return row * self.n + col


# -------------------------------
# 1. 전역 변수 기반 환경/학습 세팅
# -------------------------------
env = GridWorld(max_steps=steps_each_ep)

num_states = n * n
num_actions = 4  # 상,우,하,좌
Q = np.zeros((num_states, num_actions))

# -------------------------------
# 2. Pygame 초기화
# -------------------------------
pygame.init()

# 한 칸 크기
cell_size = 60
screen_size = (n * cell_size, n * cell_size)
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption("GridWorld with Q-Learning (Pygame, Global Variables)")

clock = pygame.time.Clock()

# 색상 정의
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 200, 0)

# -------------------------------
# 3. 메인 루프 (여기서 학습 & 시각화)
# -------------------------------
for ep in range(episode_count):
    state = env.reset()
    done = False
    step_count = 0

    while not done and step_count < steps_each_ep:
        step_count += 1

        # 이벤트 처리(창 닫기 등)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # -----------------------
        # (1) 행동 선택(ε-탐욕)
        # -----------------------
        if random.random() < epsilon:
            action = random.randint(0, num_actions - 1)
        else:
            action = np.argmax(Q[state])

        # -----------------------
        # (2) 환경에서 step
        # -----------------------
        next_state, reward, done = env.step(action)

        # -----------------------
        # (3) Q-Learning 업데이트
        # -----------------------
        best_next = np.argmax(Q[next_state])
        td_target = reward + gamma * Q[next_state, best_next]
        Q[state, action] += alpha * (td_target - Q[state, action])

        # 상태 갱신
        state = next_state

        # -----------------------
        # (4) 화면 그리기
        # -----------------------
        screen.fill(WHITE)

        # 격자선 그리기
        for i in range(n + 1):
            pygame.draw.line(screen, BLACK, (i * cell_size, 0), (i * cell_size, n * cell_size))
            pygame.draw.line(screen, BLACK, (0, i * cell_size), (n * cell_size, i * cell_size))

        # 좌표 변환( row→y, col→x )
        sr, sc = env.start
        gr, gc = env.goal
        ar, ac = env.current_pos

        # 시작점(파랑 사각형)
        start_rect = pygame.Rect(sc * cell_size + 5, sr * cell_size + 5, cell_size - 10, cell_size - 10)
        pygame.draw.rect(screen, BLUE, start_rect)

        # 목표점(초록 사각형)
        goal_rect = pygame.Rect(gc * cell_size + 5, gr * cell_size + 5, cell_size - 10, cell_size - 10)
        pygame.draw.rect(screen, GREEN, goal_rect)

        # 에이전트(빨간 원)
        agent_rect = pygame.Rect(ac * cell_size + 10, ar * cell_size + 10, cell_size - 20, cell_size - 20)
        pygame.draw.ellipse(screen, RED, agent_rect)

        # 텍스트 표시
        font = pygame.font.SysFont(None, 24)
        msg = f"Episode {ep + 1}/{episode_count}, Step {step_count}, Reward={reward:.2f}"
        text = font.render(msg, True, BLACK)
        screen.blit(text, (10, 10))

        pygame.display.flip()
        clock.tick(5)  # 초당 5프레임 (느리게 보이려면 더 낮추고, 빠르게는 높이세요)

print("학습 완료!")
pygame.quit()
