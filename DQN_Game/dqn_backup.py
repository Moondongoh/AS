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
        self.direction_score = 0  # 방향 점수 초기화
        self.goal = (7, 7)  # 목표 지점
        
        self.recent_moves = []                                  # 최근 3번의 회전 기록



        self.reset()
        
    def _get_wall_distances(self):
        x, y = self.position
        top_dist = x
        bottom_dist = self.grid_size - 1 - x
        left_dist = y
        right_dist = self.grid_size - 1 - y
        return top_dist, bottom_dist, left_dist, right_dist

    
    def reset(self):
        self.direction_score = 0  # 방향 점수 초기화
        self.position = self.start              # 시작 위치로 초기화
        self.direction = 0                      # 방향 초기화
        self.current_steps = 0                  # 이동 횟수 초기화
        self.last_angle = np.arctan2(0, 0)      # 마지막 각도 초기화
        self.total_reward = 0                   # 리셋 시 누적 보상도 초기화
        self.prev_dist = 0                      # 이전 위치와의 거리 초기화                                                    
        self.flag = False                       # 위험 지역 진입 여부 플래그 초기화
        self.visited_goals = []                 # 방문한 목표 지점 리스트 초기화
        
        self.recent_moves = []                  # 최근 3번의 회전 기록 초기화
        
        return self._get_state()                # 초기 상태 반환

    def _is_in_square(self, position):
        x, y = position
        x0, y0 = self.danger_zone['top_left']
        size = self.danger_zone['size']
        return x0 <= x < x0 + size and y0 <= y < y0 + size
    
    def _distance_to_boundary(self, position, angle):
        x, y = position
        dx, dy = np.cos(angle), np.sin(angle)

        t_x = (0 - x) / dx if dx < 0 else (self.grid_size - 1 - x) / dx if dx > 0 else float('inf')
        t_y = (0 - y) / dy if dy < 0 else (self.grid_size - 1 - y) / dy if dy > 0 else float('inf')

        distance_to_boundary = min(max(t_x, 0), max(t_y, 0))
        return distance_to_boundary
    
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

    def _distance_in_direction(self, angle):
        radians = np.radians(angle)
        dist_to_boundary = self._distance_to_boundary(self.position, radians)
        dist_to_danger = self._distance_to_square(self.position, radians)
        return min(dist_to_boundary, dist_to_danger)

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
        
        # 벽까지 거리 정보 추가
        top_dist, bottom_dist, left_dist, right_dist = self._get_wall_distances()
        wall_info = [top_dist / self.grid_size, bottom_dist / self.grid_size,
                 left_dist / self.grid_size, right_dist / self.grid_size]

        return np.array(state_info + [dist_to_start, current_angle, step_info] + wall_info)

    def step(self, action):
        rotations = [-45, 0, 45]
        angle_change = rotations[action]

        # 방향 갱신
        self.direction = (self.direction + angle_change) % 360
        if self.direction == -45:
            self.direction = 315
            
        # 🧠 벽 근처 회피: 이동 전 방향 조정
        wall_threshold = 1  # 1칸 이내면 회피
        top_dist, bottom_dist, left_dist, right_dist = self._get_wall_distances()

        if self.direction == 0 and right_dist <= wall_threshold:
            self.direction = 315 if random.random() < 0.5 else 45
        elif self.direction == 45 and (top_dist <= wall_threshold or right_dist <= wall_threshold):
            self.direction = 0
        elif self.direction == 315 and (bottom_dist <= wall_threshold or right_dist <= wall_threshold):
            self.direction = 0

        # 이동 방향 맵핑 (정수 격자 기반으로)
        move_map = {
            0:   (0, 1),    # 오른쪽
            45:  (-1, 1),   # 오른쪽 위
            315: (1, 1),    # 오른쪽 아래
        }
        
        # 이동 전 위치 저장
        prev_position = self.position

        dx, dy = move_map.get(self.direction, (0, 0))  # 안전 처리
        new_x = int(np.clip(self.position[0] + dx, 0, self.grid_size - 1))
        new_y = int(np.clip(self.position[1] + dy, 0, self.grid_size - 1))
        self.position = (new_x, new_y)
        self.current_steps += 1
        
        # ✅ 제자리면 penalty 또는 종료
        if self.position == prev_position:
            reward = -10  # 강한 패널티
            self.total_reward += reward
            return self._get_state(), reward, True  # 즉시 종료 (선택)

        # 기본 보상
        reward = 1

        # # 방향 점수 업데이트
        # if angle_change == 0:
        #     self.direction_score = 0  # ← ❗ 직진도 점수 누적 중
        # else:
        #     self.direction_score += int(np.sign(angle_change))
        #     reward += 1  # ✅ 방향 전환 시 소량 보상
        
        # 방향 점수 업데이트
        if angle_change != 0:
            self.recent_moves.append(int(np.sign(angle_change)))  # -1 or +1
        else:
            self.recent_moves.append(0)  # 직진도 기록
            
        if len(self.recent_moves) > 3:
            self.recent_moves.pop(0)
            
        if len(self.recent_moves) == 3 and sum(self.recent_moves) in [3, -3]:
            print(f"🚫 최근 회전이 한쪽으로 3번 누적됨: {self.recent_moves} → 종료")
            return self._get_state(), reward, True
        # 제자리면 종료
        
        if self.position == prev_position:
            print(f"🛑 종료: 제자리 이동 감지 at {self.position}")
            reward = -10
            self.total_reward += reward
            return self._get_state(), reward, True


        # 목표 도달
        if np.linalg.norm(np.array(self.position) - np.array(self.goal)) < 0.5:
            print(f"🏁 종료: 목표 도달 at {self.position}")
            reward += 10
            self.total_reward += reward
            return self._get_state(), reward, True

        # 방향 점수 누적 (예전 방식 유지 중이면)
        if abs(self.direction_score) >= 3:
            print(f"🛑 종료: 방향 누적 점수 초과: {self.direction_score}")
            self.total_reward += reward
            return self._get_state(), reward, True

        # 위험 지역 도달
        if self._is_in_square(self.position):
            print(f"🛑 종료: 위험 지역 진입 at {self.position}")
            self.total_reward += reward
            return self._get_state(), reward, True

        # 벽 도달
        if self.position[0] in [0, self.grid_size - 1] or self.position[1] in [0, self.grid_size - 1]:
            print(f"🛑 종료: 벽 도달 at {self.position}")
            self.total_reward += reward
            return self._get_state(), reward, True

        # 누적 리워드 너무 낮음
        if self.total_reward <= -100:
            print(f"🛑 종료: 누적 리워드 {self.total_reward} 이하")
            return self._get_state(), reward, True

        # 최대 스텝 초과
        done = self.current_steps >= self.max_steps
        if done:
            print(f"🛑 종료: 최대 스텝 초과 {self.current_steps}")
            return self._get_state(), reward, done

        

        # 목표 도달 확인
        if np.linalg.norm(np.array(self.position) - np.array(self.goal)) < 0.5:
            reward += 10
            self.total_reward += reward
            return self._get_state(), reward, True

        # 방향 제한 조건
        if abs(self.direction_score) >= 3:
            self.total_reward += reward
            return self._get_state(), reward, True

        # 위험지역 도달
        if self._is_in_square(self.position):
            self.total_reward += reward
            return self._get_state(), reward, True

        # 벽(경계) 도달
        if self.position[0] in [0, self.grid_size - 1] or self.position[1] in [0, self.grid_size - 1]:
            self.total_reward += reward
            return self._get_state(), reward, True

        # 시작점과의 거리 기반 보상
        dx_start = self.position[0] - self.start[0]
        dy_start = self.position[1] - self.start[1]
        dist_to_start = np.sqrt(dx_start ** 2 + dy_start ** 2)

        if dist_to_start > self.prev_dist:
            reward += dist_to_start
        else:
            reward -= 5

        self.prev_dist = dist_to_start
        self.total_reward += reward

        # 누적 보상이 너무 낮으면 종료
        if self.total_reward <= -100:
            return self._get_state(), reward, True

        # 최대 스텝 초과
        done = self.current_steps >= self.max_steps
        return self._get_state(), reward, done

class DQN(nn.Module):

    def __init__(self, state_dim, action_dim=3):
        super(DQN, self).__init__()
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
    state_dim = 12
    action_dim = 3
    
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
    episodes = 8000
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
        
        if (episode + 1) % 100 == 0:
            avg_reward = sum(rewards_history[-10:]) / 10
            print(f"Episode {episode + 1}: Average Reward: {avg_reward:.2f}, Epsilon: {epsilon:.2f}")

    return model, rewards_history

def visualize_episode_steps(env, model, episode_num, fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    state = env.reset()
    positions = [env.position]
    rewards = []
    states = [state]
    done = False

    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)
    ax.invert_yaxis() 
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
        x0, y0 = env.danger_zone['top_left']
        size = env.danger_zone['size']
        square = plt.Rectangle((y0, x0), size, size, edgecolor='red', facecolor='red', alpha=0.9)
        ax.add_patch(square)
        
        ax.set_xlim(0, env.grid_size)
        ax.set_ylim(0, env.grid_size)
        ax.invert_yaxis()
        ax.set_xticks(range(env.grid_size))
        ax.set_yticks(range(env.grid_size))
        ax.grid(True)

        current_positions = positions[:step+1]
        if current_positions:
            path_coords = np.array(current_positions)
            ax.plot(path_coords[:, 1], path_coords[:, 0], 'b-', alpha=0.5)
            for idx, (pos_x, pos_y) in enumerate(current_positions[:-1]):
                ax.plot(pos_y, pos_x, 'bo', alpha=0.3, markersize=8)

        ax.plot(env.start[1], env.start[0], 'go', markersize=15, label='Start')
        ax.plot(env.goal[1], env.goal[0], 'yx', markersize=15, label='Goal')

        current_x, current_y = positions[step]
        ax.plot(current_y, current_x, 'r*', markersize=15, label='Current')
        
        # ✅ 목표 도달 시 일시정지
        if np.linalg.norm(np.array(positions[step]) - np.array(env.goal)) < 0.5:
            input(f"✅ Goal reached at step {step}! Press Enter to continue...")
    
        cumulative_reward = sum(rewards[:step]) if step > 0 else 0
        ax.set_title(f'Episode {episode_num+1}, Step {step}/{len(positions)-1}\n'
                     f'Position: {positions[step]}, Reward: {cumulative_reward:.1f}\n ' f'state: {states[step]}')
        ax.legend()
        
        plt.draw()
        plt.pause(0.2)

def train_dqn_with_visualization(env):                              
    state_dim = 12
    action_dim = 3  
    
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
    episodes = 8000
    target_update = 10

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
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
            if total_reward <= -100:
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

        if (episode + 1) % 100 == 0:
            if not episode_terminated:
                visualize_episode_steps(env, model, episode, fig, ax1)
            avg_reward = sum(rewards_history[-10:]) / 10
            print(f"Episode {episode + 1}: Average Reward: {avg_reward:.2f}, Epsilon: {epsilon:.2f}")
            if episode_terminated:
                print(f"Episode terminated early due to low reward: {total_reward}")

    plt.ioff()
    plt.show()

    return model, rewards_history

env = GridEnvironment()
trained_model = train_dqn_with_visualization(env)