import gym
import imageio
import cv2  # For rendering text on frames
import numpy as np
from collections import deque
from openai import OpenAI

from describe_frame import describe_frame
from send_image_to_llm import send_image_to_llm

class RelPlayer:
    def get_action(self, frame_description: dict) -> int:
        rel = frame_description['ball_relative']
        if 'left' in rel:
            action = ACTION_LEFT
        elif 'right' in rel:
            action = ACTION_RIGHT
        else:
            action = ACTION_NOOP
        print(f'{rel} -> {action_str(action)}')
        return action

class LLMPlayer:
    def __init__(self, prompt: str, local=True):
        self.prompt = prompt
        if local:
            self.client = OpenAI(
                base_url="http://localhost:8080/v1",
                api_key = "sk-no-key-required"
            )
            self.model = 'local'
        else:
            with open('openapi-key.txt', 'r') as f:
                key = f.read().strip()
            self.client = OpenAI(api_key = key)
            self.model = 'gpt-4o-mini'

    def get_action(self, frame_description: dict) -> int:
        prompt = self.prompt.replace('[FRAME_DESC]', frame_description['text'])
        print(prompt)

        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model,
        )

        #print(chat_completion)
        response = chat_completion.choices[0].message.content
        print(response)

        action = ACTION_NOOP
        if response:
            if 'LEFT' in response and 'RIGHT' in response:
                print('BAD RESPONSE')
                print(response)
            elif 'LEFT' in response:
                action = ACTION_LEFT
            elif 'RIGHT' in response:
                action = ACTION_RIGHT
            elif "DON'T MOVE" in response:
                action  = ACTION_NOOP
            else:
                print('BAD RESPONSE')
                print(response)

        return action


class VisualLLMPlayer(LLMPlayer):
    def __init__(self, prompt: str, local=True):
        super().__init__(prompt=prompt, local=local)

    def get_action(self, frame_description: dict) -> int:
        frame = frame_description['frame']
        prompt = self.prompt.replace('[BALL_DIRECTION]', frame_description['ball_direction'])
        response = send_image_to_llm(frame, prompt, self.client)

        print(response)

        action = ACTION_NOOP
        if response:
            if 'LEFT' in response and 'RIGHT' in response:
                print('BAD RESPONSE')
                print(response)
            elif 'LEFT' in response:
                action = ACTION_LEFT
            elif 'RIGHT' in response:
                action = ACTION_RIGHT
            elif "DON'T MOVE" in response:
                action  = ACTION_NOOP
            else:
                print('BAD RESPONSE')
                print(response)

        return action

#player = RelPlayer()
#player = LLMPlayer(prompt="""
#Lets play atari breakout. I will describe what is happening on the screen and you will tell me which direction to the move the paddle or to keep it still. In order to win, you will need to move the paddle so that it is aligned with the ball. You may also choose not to move the paddle. Here is the current frame:
#
#[FRAME_DESC]
#
#Which direction should I move the paddle so it is aligned with the ball? Say LEFT, RIGHT, or STAY
#""", local=False)

player = VisualLLMPlayer(prompt="""
This is a screenshot from atari breakout. The goal is to get the ball to hit the paddle at the bottom of the screen. You can move the paddle left, right, or do nothing. In this frame the ball is [BALL_DIRECTION].

Which direction should I move the paddle so it is aligned with the ball? Say LEFT, RIGHT, or STAY
""", local=True)

ACTION_NOOP = 0
ACTION_FIRE = 1
ACTION_RIGHT = 2
ACTION_LEFT = 3

def action_str(action: int) -> str:
    if action == ACTION_NOOP:  return 'NOOP'
    if action == ACTION_FIRE:  return 'FIRE'
    if action == ACTION_LEFT:  return 'LEFT'
    if action == ACTION_RIGHT: return 'RIGHT'
    raise ValueError(f'Not an action: {action}')

# Initialize the Atari Breakout environment
env = gym.make('ALE/Breakout-v5', render_mode="rgb_array", obs_type='ram')

# Output video file
video_filename = "breakout_gameplay_with_descriptions_upscaled.mp4"
frames = []  # To store rendered frames

# Number of episodes to play
num_episodes = 1  # Adjust as needed

# Screen dimensions and upscale factor
screen_width = 160  # Default width of Breakout (adjust if needed)
upscale_factor = 3  # Upscale factor for text rendering

previous_frame_descriptions = deque(maxlen=3)


for episode in range(num_episodes):
    print(f"Starting episode {episode + 1}")
    state = env.reset()
    done = False
    total_reward = 0

    # Initialize previous positions (set to None initially)
    ball_prev = (0, 0)
    paddle_prev = 0
    desc_data = None

    while not done:
        # Extract RAM for ball and paddle positions
        ram = env.unwrapped.ale.getRAM()
        ball_x = ram[99]    # X-position of the ball
        ball_y = ram[101]   # Y-position of the ball
        paddle_x = ram[72]  # X-position of the paddle

        # Current positions
        ball_curr = (ball_x, ball_y)
        paddle_curr = paddle_x

        # Generate frame description
        description = ""
        if ball_prev == (0, 0):  # Avoid describing the first frame
            previous_frame_descriptions.clear()
            description = 'N/A'
        else:
            desc_data = describe_frame(ball_curr, ball_prev, paddle_curr, paddle_prev, screen_width)
            description = desc_data['text']
            previous_frame_descriptions.append(description)

        # Render frame
        frame = env.render()

        if desc_data:
            desc_data['frame'] = frame

        # Upscale the frame for better text rendering
        frame_upscaled = cv2.resize(frame, (frame.shape[1] * upscale_factor, frame.shape[0] * upscale_factor))

        # Render text on the upscaled frame
        y0, dy = 20, 25  # Starting y-coordinate and line spacing for text
        for i, line in enumerate(description.split('\n')):
            y = y0 + i * dy
            # Overlay the text (use smaller font size for readability)
            cv2.putText(frame_upscaled, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # Store the final upscaled frame for video
        frames.append(frame_upscaled)

        # Take a random action (replace with policy if available)
        #action = env.action_space.sample()
        action = ACTION_FIRE
        if ball_prev[0] == ball_curr[0] and ball_prev[1] == ball_curr[1]:
            action = ACTION_FIRE
        else:
            if desc_data:
                action = player.get_action(desc_data)
            else:
                action = ACTION_NOOP

        state, reward, done, truncated, info = env.step(action)

        total_reward += reward

        # Update previous positions
        ball_prev = ball_curr
        paddle_prev = paddle_curr

    print(f"Episode {episode + 1} finished with a total reward of {total_reward}")

# Save frames as a video
print(f"Saving video to {video_filename}...")
imageio.mimsave(video_filename, frames, fps=30)
print(f"Video saved successfully!")

# Close the environment
env.close()
