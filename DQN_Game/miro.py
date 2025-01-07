import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

# GPU 사용 가능 여부 확인 및 device 설정
print(f"CUDA Available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class GridEnvironment:
    def __init__(self):
        self.grid_size = 5
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.start = (0, 0)
        self.small_goal = (2, 3)
        self.large_goal = (1, 4)
        self.reset()

    def reset(self):
        self.position = self.start
        return self.position

    def step(self, action):
        x, y = self.position
        if action == 0 and x > 0:  # 상
            x -= 1
        elif action == 1 and x < self.grid_size - 1:  # 하
            x += 1
        elif action == 2 and y > 0:  # 좌
            y -= 1
        elif action == 3 and y < self.grid_size - 1:  # 우
            y += 1

        self.position = (x, y)

        if self.position == self.small_goal:
            return self.position, 1, True
        elif self.position == self.large_goal:
            return self.position, 10, True
        else:
            return self.position, 0, False

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

def train_dqn():
    env = GridEnvironment()
    state_dim = 2
    action_dim = 4
    
    # 모델을 GPU로 이동
    model = DQN(state_dim, action_dim).to(device)
    target_model = DQN(state_dim, action_dim).to(device)
    target_model.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    replay_buffer = []
    max_buffer_size = 1000
    batch_size = 64
    gamma = 0.99  # 할인율
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.1
    episodes = 100
    target_update = 10

    for episode in range(episodes):
        state = np.array(env.reset())
        done = False
        total_reward = 0

        while not done:
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                # state를 GPU로 이동
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = model(state_tensor)
                action = q_values.argmax().item()

            next_state, reward, done = env.step(action)
            total_reward += reward

            replay_buffer.append((state, action, reward, next_state, done))
            if len(replay_buffer) > max_buffer_size:
                replay_buffer.pop(0)

            state = np.array(next_state)

            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                # 모든 텐서를 GPU로 이동
                states = torch.FloatTensor(np.array(states)).to(device)
                next_states = torch.FloatTensor(np.array(next_states)).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                dones = torch.FloatTensor(dones).to(device)

                q_values = model(states).gather(1, actions).squeeze()

                # 타겟 Q-값을 계산 (할인된 미래 보상 추가)
                with torch.no_grad():
                    max_next_q_values = target_model(next_states).max(1)[0]  # 타겟 폴리시에서 최대 Q-값
                    target_q_values = rewards + (1 - dones) * gamma * max_next_q_values

                # MSE Loss 계산
                loss = criterion(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Epsilon 감소
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        # 타겟 네트워크 업데이트
        if episode % target_update == 0:
            target_model.load_state_dict(model.state_dict())

        if (episode + 1) % 10 == 0:  # 10 에피소드마다 출력
            print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    return model


def visualize_path_with_animation(env, model):
    state = np.array(env.reset())
    done = False
    path = [env.start]

    fig, ax = plt.subplots(figsize=(8, 8))
    grid = np.zeros((env.grid_size, env.grid_size))

    def render():
        grid[:, :] = 0
        # 시작점 표시
        grid[env.start] = 0.3
        # 목표점들 표시
        grid[env.small_goal] = 0.6
        grid[env.large_goal] = 1.0
        # 경로 표시
        for (x, y) in path:
            grid[x, y] = 0.8
        
        # 현재 위치 강조
        current_pos = path[-1]
        
        ax.clear()
        ax.imshow(grid, cmap="Blues")
        
        # 격자 그리기
        for i in range(env.grid_size):
            for j in range(env.grid_size):
                ax.text(j, i, f'({i},{j})', ha='center', va='center', color='red', fontsize=8)
                ax.axhline(y=i-0.5, color='black', linewidth=0.5)
                ax.axvline(x=j-0.5, color='black', linewidth=0.5)
        
        # 범례 추가
        ax.text(-0.5, -0.5, "S: Start", color='black')
        ax.text(1.5, -0.5, "G1: Small Goal", color='black')
        ax.text(3.5, -0.5, "G2: Large Goal", color='black')
        
        # 현재 위치 표시
        ax.plot(current_pos[1], current_pos[0], 'r*', markersize=15)
        
        ax.set_xticks(range(env.grid_size))
        ax.set_yticks(range(env.grid_size))
        ax.grid(True)
        ax.set_title(f'Current Position: {current_pos}')
        plt.draw()
        plt.pause(0.5)

    while not done:
        render()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state_tensor)
        action = q_values.argmax().item()
        next_state, _, done = env.step(action)
        path.append(env.position)
        state = np.array(next_state)

    render()
    plt.show()

# 실행
env = GridEnvironment()
trained_model = train_dqn()
visualize_path_with_animation(env, trained_model)