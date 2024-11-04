import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from ale_py import ALEInterface
from ale_py.roms import Breakout

# Initialize the Breakout environment
env = gym.make("ALE/Breakout-v5", render_mode=None)

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the DQN network
class DQN(nn.Module):
    def __init__(self, action_space):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_space)
        )

    def forward(self, x):
        return self.network(x)

# Hyperparameters
batch_size = 32
gamma = 0.99
eps_start = 1.0
eps_end = 0.1
eps_decay = 0.999
target_update = 10
memory_capacity = 10000
learning_rate = 1e-4
num_episodes = 500

# Initialize replay memory and DQN
memory = deque(maxlen=memory_capacity)
policy_net = DQN(env.action_space.n).to(device)
target_net = DQN(env.action_space.n).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
steps_done = 0

# Helper functions
def preprocess(frame):
    frame = frame[35:195:2, ::2]
    frame = np.mean(frame, axis=2).astype(np.float32)
    frame /= 255.0
    return frame

def get_epsilon():
    global steps_done
    eps = eps_end + (eps_start - eps_end) * np.exp(-1. * steps_done * eps_decay)
    return eps

def select_action(state):
    global steps_done
    steps_done += 1
    if random.random() < get_epsilon():
        return torch.tensor([[random.randrange(env.action_space.n)]], device=device, dtype=torch.long)
    else:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)

def optimize_model():
    if len(memory) < batch_size:
        return
    transitions = random.sample(memory, batch_size)
    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

    batch_state = torch.cat(batch_state)
    batch_action = torch.cat(batch_action)
    batch_reward = torch.cat(batch_reward)
    batch_next_state = torch.cat(batch_next_state)
    batch_done = torch.cat(batch_done)

    current_q_values = policy_net(batch_state).gather(1, batch_action)
    next_q_values = target_net(batch_next_state).max(1)[0].detach().unsqueeze(1)
    expected_q_values = batch_reward + (gamma * next_q_values * (1 - batch_done))

    criterion = nn.SmoothL1Loss()
    loss = criterion(current_q_values, expected_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Training loop
for episode in range(num_episodes):
    state, _ = env.reset()
    state = preprocess(state)
    state = np.stack([state] * 4, axis=0)
    state = torch.from_numpy(state).unsqueeze(0).to(device)
    episode_reward = 0

    while True:
        action = select_action(state)
        next_state, reward, done, _, _ = env.step(action.item())
        episode_reward += reward

        next_state = preprocess(next_state)
        next_state = np.append(state.cpu().numpy()[0, 1:], [next_state], axis=0)
        next_state = torch.from_numpy(next_state).unsqueeze(0).to(device)

        reward = torch.tensor([[reward]], device=device)
        done = torch.tensor([[done]], device=device, dtype=torch.float32)

        memory.append((state, action, reward, next_state, done))

        state = next_state

        optimize_model()

        if done:
            print(f"Episode {episode + 1} - Reward: {episode_reward}")
            break

    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

env.close()
