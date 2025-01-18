import gymnasium
import pygame
import numpy as np

env = gymnasium.make("MountainCar-v0", render_mode="human")
env.reset()

done = False

while not done:
    
    action = 2
    new_state, reward, done, _, _ = env.step(action)
    print(new_state)
    env.render()

env.close()