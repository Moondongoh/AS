import pygame
import numpy as np
from gym import Env
from gym.spaces import Discrete, Box

class ShootingGameEnv(Env):
    def __init__(self):
        super(ShootingGameEnv, self).__init__()
        pygame.init()
        self.screen_width = 200  # 너비 축소
        self.screen_height = 300  # 높이 축소
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("RL Shooting Game")

        self.player_width = 25
        self.player_height = 5
        self.enemy_width = 15
        self.enemy_height = 15

        self.action_space = Discrete(3)  # 0: 왼쪽, 1: 정지, 2: 오른쪽
        self.observation_space = Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 1), dtype=np.uint8)

        self.reset()

    def reset(self):
        self.player_x = self.screen_width // 2 - self.player_width // 2
        self.player_y = self.screen_height - self.player_height - 10

        self.enemy_x = np.random.randint(0, self.screen_width - self.enemy_width)
        self.enemy_y = 0

        self.score = 0
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        state_surface = pygame.Surface((self.screen_width, self.screen_height))
        state_surface.fill((0, 0, 0))

        pygame.draw.rect(state_surface, (0, 255, 0), (self.player_x, self.player_y, self.player_width, self.player_height))
        pygame.draw.rect(state_surface, (255, 0, 0), (self.enemy_x, self.enemy_y, self.enemy_width, self.enemy_height))

        state = pygame.surfarray.array3d(state_surface)
        state = np.transpose(state, (1, 0, 2))

        state = np.mean(state, axis=2, keepdims=True).astype(np.uint8)

        return state

    def step(self, action):
        if action == 0:
            self.player_x = max(0, self.player_x - 5)
        elif action == 2:
            self.player_x = min(self.screen_width - self.player_width, self.player_x + 5)

        self.enemy_y += 3

        reward = 0
        if self.enemy_y + self.enemy_height >= self.player_y:
            if self.player_x <= self.enemy_x <= self.player_x + self.player_width or \
               self.player_x <= self.enemy_x + self.enemy_width <= self.player_x + self.player_width:
                reward = 1
                self.enemy_x = np.random.randint(0, self.screen_width - self.enemy_width)
                self.enemy_y = 0
                self.score += 1
            else:
                self.done = True

        if self.enemy_y > self.screen_height:
            self.enemy_x = np.random.randint(0, self.screen_width - self.enemy_width)
            self.enemy_y = 0

        return self._get_obs(), reward, self.done, {}

    def render(self, mode='human'):
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (0, 255, 0), (self.player_x, self.player_y, self.player_width, self.player_height))
        pygame.draw.rect(self.screen, (255, 0, 0), (self.enemy_x, self.enemy_y, self.enemy_width, self.enemy_height))
        pygame.display.flip()

    def close(self):
        pygame.quit()
