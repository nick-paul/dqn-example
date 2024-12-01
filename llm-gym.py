import gymnasium as gym
import ale_py
from openai import OpenAI


def main():

    client = OpenAI(
        base_url="http://localhost:8080/v1",
        api_key = "sk-no-key-required"
    )

    gym.register_envs(ale_py)

    # Initialise the environment
    env = gym.make("ALE/Breakout-v5", render_mode="human")

    return env

env = main()



#    # Reset the environment to generate the first observation
#    observation, info = env.reset(seed=42)
#    for _ in range(1000):
#        # this is where you would insert your policy
#        action = env.action_space.sample()
#
#        # step (transition) through the environment with the action
#        # receiving the next observation, reward and if the episode has terminated or truncated
#        observation, reward, terminated, truncated, info = env.step(action)
#
#        # If the episode has ended then we can reset to start a new episode
#        if terminated or truncated:
#            observation, info = env.reset()
#
#    env.close()
