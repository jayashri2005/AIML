import gymnasium as gym
import time

env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset()

for _ in range(3000):
    x, x_dot, theta, theta_dot = obs

    # [x, x_dot, theta, theta_dot] (cart position, cart velocity, pole angle, pole angular velocity)
    action = int(10.0 * theta + theta_dot > 0)  #
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

    time.sleep(0.02)

env.close()