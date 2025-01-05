from Game import ShootingGameEnv
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
import pygame

# 환경 생성 및 수동 래핑
env = DummyVecEnv([lambda: ShootingGameEnv()])
env = VecTransposeImage(env)

# DQN 모델 생성 및 학습
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-3,
    buffer_size=5000,
    learning_starts=500,
    batch_size=32,
    target_update_interval=500
)

print("Starting training...")
model.learn(total_timesteps=20000, log_interval=100)  #<<<<<<<<<<<<<<<<<Episode 반복을 위해서 total_timesteps 값을 늘려서 에피소드 반복 횟수를 증가
print("Training complete.")

# 모델 저장
model.save("dqn_shooting_game_model")

# 테스트: 학습된 모델로 게임 실행
env = ShootingGameEnv()  # 원래 환경으로 다시 생성
obs = env.reset()
total_rewards = 0
episode_count = 0

print("Starting testing...")
for step in range(1000):
    env.render()  # 게임 화면을 그리기
    action, _ = model.predict(obs, deterministic=True)  # 모델 예측에 따라 행동
    obs, reward, done, _ = env.step(action)
    total_rewards += reward

    if done:
        episode_count += 1
        print(f"Episode {episode_count} ended with reward {total_rewards}")
        obs = env.reset()
        total_rewards = 0
        pygame.time.wait(500)

env.close()
