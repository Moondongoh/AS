import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


### 환경 설정 class
class GridEnvironment:
    ### 환경설정
    def __init__(self):
        self.grid_size = 9                                      # 격자 크기
        self.grid = np.zeros((self.grid_size, self.grid_size))  # 9x9 크기의 0으로 채워진 배열 생성
        self.start = (1, 1)                                     # 시작 지점
        #self.danger_zone = {'top_left': (0, 4), 'size': 5}      # 3,3에서 시작하는 3x3 정사각형
        self.danger_zone = {'top_left': (3, 3), 'size': 3}      # 3,3에서 시작하는 3x3 정사각형
        self.max_steps = 200                                    # 최대 이동 횟수(같은 방향으로 너무 많이 진행할 경우 방지지)
        self.current_steps = 0                                  # 현재 이동 횟수 
        self.direction = 0                                      # 초기 방향 (0도는 오른쪽, 90도는 위쪽)    
        self.position = self.start                              # 현재 위치   
        self.last_angle = 0                                     # 마지막 각도 (이전 이동 방향)
        self.total_reward = 0                                   # 누적 보상 추적을 위해 추가
        self.flag = False                                       # 위험 지역 진입 여부 플래그    
        self.prev_dist = 0                                      # 이전 위치와의 거리
        self.visited_goals = []                                 # 방문한 목표 지점 리스트 초기화
        self.visited_positions = {}                             # 중복 방문 체크용 딕셔너리

        # prev_dx, prev_dy 초기화
        self.prev_dx = 0  
        self.prev_dy = 0 

        ### 환경 설정 후 초기화 호출
        self.reset()

    def reset(self):
        self.position = self.start              # 시작 위치로 초기화
        self.direction = 0                      # 방향 초기화
        self.current_steps = 0                  # 이동 횟수 초기화
        self.last_angle = np.arctan2(0, 0)      # 마지막 각도 초기화
        self.total_reward = 0                   # 리셋 시 누적 보상도 초기화
        self.prev_dist = 0                      # 이전 위치와의 거리 초기화                                                    
        self.flag = False                       # 위험 지역 진입 여부 플래그 초기화
        self.visited_goals = []                 # 방문한 목표 지점 리스트 초기화
        return self._get_state()                # 초기 상태 반환
    
    ### _is_in_square() (위치가 위험 지역 내에 있는지 확인)
    ### 현재 위치가 위험 지역 내에 있는지 확인하는 함수
    ### (3,3)을 시작으로 3x3 크기의 정사각형 내에 있으면 True 반환 
    ### True (위험 지역 안에 있음) / False (위험 지역 밖에 있음) << x0 <= x < x0 + size && y0 <= y < y0 + size → 위험 지역 내부
    def _is_in_square(self, position):
        x, y = position
        x0, y0 = self.danger_zone['top_left']
        size = self.danger_zone['size']
        return x0 <= x < x0 + size and y0 <= y < y0 + size
    
    ### _distance_to_boundary() (경계까지의 거리 계산)
    
    # 입력: 현재 위치 (x, y), 이동 방향 angle
    # 출력: 현재 방향으로 직진했을 때 경계(boundary)까지의 거리
    # 가장 작은 값을 반환 → 가장 먼저 부딪히는 벽까지 거리
    
    ### 현재 방향(angle)에서 가장자리까지의 거리 계산
    ### dx, dy = np.cos(angle), np.sin(angle)을 이용해 이동 방향 결정
    ### x축과 y축 각각에 대해 벽까지의 거리 계산 후 최솟값을 반환
    def _distance_to_boundary(self, position, angle):
        x, y = position
        dx, dy = np.cos(angle), np.sin(angle)

        t_x = (0 - x) / dx if dx < 0 else (self.grid_size - 1 - x) / dx if dx > 0 else float('inf')
        t_y = (0 - y) / dy if dy < 0 else (self.grid_size - 1 - y) / dy if dy > 0 else float('inf')

        distance_to_boundary = min(max(t_x, 0), max(t_y, 0))
        return distance_to_boundary
    
    ### _distance_to_square() (위험 지역까지의 거리 계산)
    
    # 입력: 현재 위치 (x, y), 이동 방향 angle
    # 출력: 현재 방향으로 직진했을 때 위험 지역까지의 거리
    
    ### 현재 방향에서 위험 지역(정사각형)까지의 거리 계산
    ### 에이전트가 바라보는 방향을 기준으로 직선 이동했을 때 정사각형의 어느 변과 만나는지 계산 후 최소값을 반환
    def _distance_to_square(self, position, angle):
        x, y = position
        x0, y0 = self.danger_zone['top_left']
        size = self.danger_zone['size']

        dx, dy = np.cos(angle), np.sin(angle)

        t_vals = []
        for edge_x in [x0, x0 + size]:
            if dx != 0:
                t_vals.append((edge_x - x) / dx)

        for edge_y in [y0, y0 + size]:
            if dy != 0:
                t_vals.append((edge_y - y) / dy)

        t_vals = [t for t in t_vals if t > 0]
        return min(t_vals) if t_vals else float('inf')
    
    ### _distance_in_direction() (특정 방향에서 경계 또는 위험 지역까지의 거리)
    # 작은값 반환
    def _distance_in_direction(self, angle):
        radians = np.radians(angle)
        dist_to_boundary = self._distance_to_boundary(self.position, radians)
        dist_to_danger = self._distance_to_square(self.position, radians)
        return min(dist_to_boundary, dist_to_danger)

    ### _get_state() (현재 상태 반환)
    def _get_state(self):
        directions = [-90, -45, 0, 45, 90]
        state_info = [
            self._distance_in_direction(self.direction + angle) / self.grid_size
            for angle in directions
        ]

        dx = self.position[0] - self.start[0]
        dy = self.position[1] - self.start[1]
        dist_to_start = np.sqrt(dx**2 + dy**2) / self.grid_size
        current_angle = np.arctan2(dy, dx) / np.pi
        step_info = self.current_steps / self.max_steps

        return np.array(state_info + [dist_to_start, current_angle, step_info])

    def step(self, action):
        # 이동방향 5개중 1개의 값 선택
        rotations = [-90, -45, 0, 45, 90]
        self.direction = (self.direction + rotations[action]) % 360
        radians = np.radians(self.direction)
        dx, dy = np.cos(radians), np.sin(radians)  # 이동 벡터
        reward = 0.17   # 기본 보상
        # 새로운 위치 업데이트 x,y값을 더해줌
        new_x = np.clip(self.position[0] + dx, 0, self.grid_size - 1)
        new_y = np.clip(self.position[1] + dy, 0, self.grid_size - 1)
        self.position = (new_x, new_y)
        self.current_steps += 1

        # 종료 조건 체크
        if self._is_in_square(self.position):
            return self._get_state(), reward,True  # 위험 지역 진입 시 종료
    
        if self.position[0] in [0, self.grid_size - 1] or self.position[1] in [0, self.grid_size - 1]:
            return self._get_state(), reward, True  # 가장자리 충돌 시 종료

        # 현재 위치와 시작 위치 사이의 유클리드 거리 계산.
        dx_start = self.position[0] - self.start[0]
        dy_start = self.position[1] - self.start[1]
        dist_to_start = np.sqrt(dx_start**2 + dy_start**2)

        # 보상 계산
        
    
        # 목표 지점 추가 보상
        # goal_positions = [(1, 8), (8, 8), (8, 1)]
        # for goal in goal_positions:
        #     if (round(self.position[0]), round(self.position[1])) == goal and goal not in self.visited_goals:
        #         reward += 20.0
        #         self.visited_goals.append(goal)

        # 현재거리가 이전 거리 보다 크면 보상을 추가 하고, 작으면 패널티
        if dist_to_start > self.prev_dist:
            reward += 1 * dist_to_start
        else:
            reward -= 5

        # 회전 보상 (이전 이동 방향 기준)
        if self.prev_dx is not None and self.prev_dy is not None:
            prev_angle = np.arctan2(self.prev_dy, self.prev_dx)
            current_angle = np.arctan2(dy, dx)
            angle_diff = abs(current_angle - prev_angle) % (2 * np.pi)
            if angle_diff > 0:
                reward += 0.2

        # # **중복 이동 패널티 추가**
        # rounded_pos = (round(self.position[0]), round(self.position[1]))
        # if rounded_pos in self.visited_positions:
        #     self.visited_positions[rounded_pos] += 1
        #     if self.visited_positions[rounded_pos] >= 3:  # 같은 위치 3번 이상 방문하면 패널티
        #         reward -= 10
        # else:
        #     self.visited_positions[rounded_pos] = 1
         
        # # 일정 횟수 이상 같은 위치를 방문하면 강제 종료
        # if self.visited_positions[rounded_pos] >= 5:
        #     return self._get_state(), reward - 20, True  # -20 추가 패널티 후 종료

        # # 이전 이동 방향 저장
        # self.prev_dx = dx
        # self.prev_dy = dy
        # self.prev_dist = dist_to_start
        # self.total_reward += reward

        # 보상이 -50 이하일 경우 종료
        if self.total_reward <= -50:
            return self._get_state(), reward, True

        # 최대 스텝 초과 시 종료
        done = self.current_steps >= self.max_steps
        return self._get_state(), reward, done

class DQN(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        # state_dim: 8 (5개 방향 + 시작점 거리 + 각도 + 스텝) action_dim: 5 (5개 방향)
        # 신경망 구조 정의
        # 8개 입력 → 256개 출력 →256개 출력 →128개출력 → action dim출력, ReLU 활성화 함수 
        self.fc = nn.Sequential( 
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

def train_dqn(env):
    state_dim = 8  # 5개 방향 + 시작점 거리 + 각도 + 스텝
    action_dim = 5
    
    model = DQN(state_dim, action_dim).to(device)
    target_model = DQN(state_dim, action_dim).to(device)
    target_model.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    criterion = nn.MSELoss()

    replay_buffer = []
    max_buffer_size = 20000
    batch_size = 256
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.999
    epsilon_min = 0.01
    episodes = 2000
    target_update = 10

    rewards_history = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = model(state_tensor)
                action = q_values.argmax().item()

            next_state, reward, done = env.step(action)
            total_reward += reward

            replay_buffer.append((state, action, reward, next_state, done))
            if len(replay_buffer) > max_buffer_size:
                replay_buffer.pop(0)

            state = next_state

            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.FloatTensor(np.array(states)).to(device)
                next_states = torch.FloatTensor(np.array(next_states)).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                dones = torch.FloatTensor(dones).to(device)

                current_q_values = model(states).gather(1, actions).squeeze()

                with torch.no_grad():
                    next_q_values = target_model(next_states).max(1)[0]
                    target_q_values = rewards + (1 - dones) * gamma * next_q_values

                loss = criterion(current_q_values, target_q_values)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        
        if episode % target_update == 0:
            target_model.load_state_dict(model.state_dict())

        rewards_history.append(total_reward)
        
        if (episode + 1) % 10 == 0:
            avg_reward = sum(rewards_history[-10:]) / 10
            print(f"Episode {episode + 1}: Average Reward: {avg_reward:.2f}, Epsilon: {epsilon:.2f}")

    return model, rewards_history


### 시각화 함수들

### 에이전트가 특정 에피소드에서 수행한 경로를 시각적으로 보임.
### 에이전트의 이동 경로, 현재 위치, 위험 지역 등을 시각화
def visualize_episode_steps(env, model, episode_num, fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    state = env.reset()
    positions = [env.position]
    rewards = []
    states = [state]
    done = False

    # 축 고정
    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)
    ax.invert_yaxis()  # Y축을 위에서 아래로 표시
    ax.set_xticks(range(env.grid_size))
    ax.set_yticks(range(env.grid_size))
    ax.grid(True)

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state_tensor)
        action = q_values.argmax().item()
        
        next_state, reward, done = env.step(action)
        positions.append(env.position)
        rewards.append(reward)
        states.append(next_state)
        state = next_state

    for step in range(len(positions)):
        ax.clear()
        # 위험 영역(정사각형) 표시
        x0, y0 = env.danger_zone['top_left']
        size = env.danger_zone['size']
        square = plt.Rectangle((y0, x0), size, size, edgecolor='red', facecolor='red', alpha=0.9)
        ax.add_patch(square)
        
        # 고정된 축 설정
        ax.set_xlim(0, env.grid_size)
        ax.set_ylim(0, env.grid_size)
        ax.invert_yaxis()
        ax.set_xticks(range(env.grid_size))
        ax.set_yticks(range(env.grid_size))
        ax.grid(True)

        # 경로 그리기
        current_positions = positions[:step+1]
        if current_positions:
            path_coords = np.array(current_positions)
            ax.plot(path_coords[:, 1], path_coords[:, 0], 'b-', alpha=0.5)
            for idx, (pos_x, pos_y) in enumerate(current_positions[:-1]):
                ax.plot(pos_y, pos_x, 'bo', alpha=0.3, markersize=8)

        # 시작점과 현재 위치
        ax.plot(env.start[1], env.start[0], 'go', markersize=15, label='Start')
        current_x, current_y = positions[step]
        ax.plot(current_y, current_x, 'r*', markersize=15, label='Current')

        # 누적 보상
        cumulative_reward = sum(rewards[:step]) if step > 0 else 0
        ax.set_title(f'Episode {episode_num+1}, Step {step}/{len(positions)-1}\n'
                     f'Position: {positions[step]}, Reward: {cumulative_reward:.1f}\n ' f'state: {states[step]}')
        ax.legend()
        
        plt.draw()
        plt.pause(0.2)

### DQN 알고리즘을 사용하여 에이전트를 학습시키고, 학습 과정에서 보상과 스텝 수를 시각화
def train_dqn_with_visualization(env):                              
    state_dim = 8
    action_dim = 5  
    
    model = DQN(state_dim, action_dim).to(device)
    target_model = DQN(state_dim, action_dim).to(device)
    target_model.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    criterion = nn.MSELoss()

    replay_buffer = []
    max_buffer_size = 20000
    batch_size = 256
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.999
    epsilon_min = 0.01
    episodes = 2000
    target_update = 10

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))  # ax2 지움 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    rewards_history = []
    steps_history = []
    plt.ion()

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        episode_terminated = False
        
        while not done and not episode_terminated:
            ### epsilon 확률로 랜덤 행동을 수행 (탐험) >> 무작위로 움직인다.
            ### 그 외에는 현재 신경망이 예측한 Q-value 중 가장 높은 행동 선택 (활용)
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = model(state_tensor)
                action = q_values.argmax().item()
            
            next_state, reward, done = env.step(action)
            total_reward += reward
            step_count += 1
            
            # 누적 리워드가 -50 이하면 에피소드 종료
            if total_reward <= -50:
                episode_terminated = True
                done = True
            
            # Store in replay buffer
            replay_buffer.append((state, action, reward, next_state, done))
            if len(replay_buffer) > max_buffer_size:
                replay_buffer.pop(0)

            state = next_state

            # Train model if buffer has enough samples
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.FloatTensor(np.array(states)).to(device)
                next_states = torch.FloatTensor(np.array(next_states)).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                dones = torch.FloatTensor(dones).to(device)

                current_q_values = model(states).gather(1, actions).squeeze()
                with torch.no_grad():
                    next_q_values = target_model(next_states).max(1)[0]
                    target_q_values = rewards + (1 - dones) * gamma * next_q_values

                loss = criterion(current_q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        if episode % target_update == 0:
            target_model.load_state_dict(model.state_dict())

        rewards_history.append(total_reward)
        steps_history.append(step_count)

        # Update plots
        ax1.clear()
        
        ax1.plot(rewards_history, label='Rewards', color='blue')
        ax1.set_title("Episode Rewards")
        ax1.set_xlabel("Episodes")
        ax1.set_ylabel("Total Reward")
        ax1.legend()

        plt.draw()
        plt.pause(0.01)

        if (episode + 1) % 10 == 0:
            if not episode_terminated:  # Only visualize if episode wasn't terminated early
                visualize_episode_steps(env, model, episode, fig, ax1)
            avg_reward = sum(rewards_history[-10:]) / 10
            print(f"Episode {episode + 1}: Average Reward: {avg_reward:.2f}, Epsilon: {epsilon:.2f}")
            if episode_terminated:
                print(f"Episode terminated early due to low reward: {total_reward}")

    plt.ioff()
    plt.show()

    return model, rewards_history

# 순서 
# 환경 초기화
# Epsilon-Greedy로 행동 선택
# 환경에서 행동 수행 후 보상 획득
# 경험을 Replay Buffer에 저장
# 미니배치를 샘플링하여 학습 진행
# 벨만 방정식 기반으로 Q값 업데이트 ->> 미래의 예상 보상을 기반으로 업데이트
# 여러 번 반복하여 최적의 정책 학습

# 실행
env = GridEnvironment()
trained_model = train_dqn_with_visualization(env)