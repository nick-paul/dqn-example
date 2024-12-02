import gym
import random
import imageio
import cv2  # For rendering text on frames
from describe_frame import describe_frame
from actions import *
from collections import deque
import json
from prompt import make_prompt, PROMPT_FULL_V1, PROMPT_SIMPLE_V1
from actions import action_str, ACTION_RIGHT, ACTION_LEFT, ACTION_NOOP, ACTION_FIRE
from typing import Optional


from agent_paddleai import PaddleAIAgent
from agent_simple import SimpleAgent
from agent_llm import LLMAgent

#player = RelPlayer()
#player = LLMPlayer(prompt="""
#Lets play atari breakout. I will describe what is happening on the screen and you will tell me which direction to the move the paddle or to keep it still. In order to win, you will need to move the paddle so that it is aligned with the ball. You may also choose not to move the paddle. Here is the current frame:
#
#[FRAME_DESC]
#
#Should I move the paddle LEFT, move the paddle RIGHT, or STAY in place? Say LEFT, RIGHT, or STAY
#""", local=False)

player = SimpleAgent()
#player = PaddleAIAgent()

### Basic LLM

#basic_template = """
#Lets play atari breakout. I will describe what is happening on the screen and you will tell me which direction to the move the paddle or to keep it still. In order to win, you will need to move the paddle so that it is aligned with the ball. You may also choose not to move the paddle. Here is the current frame:
#
#[FRAME_DESC]
#
#Based on where the ball is, where should I move the paddle so it is aligned with the ball? Say LEFT, RIGHT, or STAY
#"""
#player = LLMAgent(
#    local=True,
#    prompt_generator=lambda fd: basic_template.replace('[FRAME_DESC]', fd['text'])
#)
#player.model = 'llama-1b'

### Advanced LLM

player = LLMAgent(
    local=True,
    prompt_generator=lambda fd: make_prompt(
        template=PROMPT_FULL_V1,
        ball_curr=fd['ball_curr'],
        ball_prev=fd['ball_prev'],
        paddle_curr=fd['paddle_curr'],
        paddle_prev=fd['paddle_prev'],
        prev_action={
            ACTION_LEFT: 'left',
            ACTION_RIGHT: 'right',
            ACTION_NOOP: 'stay',
            ACTION_FIRE: 'stay'
        }[fd['last_action']].upper(),
    )
)
player.model = 'llama-3b'

def save_log(agent, run_id, total_reward=None) -> Optional[str]:
    logfile = None
    if len(agent.log) > 0:
        log = {'log': agent.log}

        if total_reward is None:
            reward_str = 'running'
        else:
            reward_str = str(int(total_reward))
            # Add reward to log
            log['total_reward'] = total_reward

        logfile = f'runs/log-{run_id}-{reward_str}.json'

        with open(logfile, 'w') as f:
            json.dump(log, f, indent=4)

    return logfile

def save_video(frames, run_id, framecount, total_reward):
    # Save frames as a video
    video_filename = f"runs/breakout_{run_id}_{int(total_reward)}_checkpoint{framecount}.mp4"
    print(f"Saving video to {video_filename}...")
    imageio.mimsave(video_filename, frames, fps=30)
    print(f"Video saved successfully!")


def run():
    run_id = ''.join([chr(random.randint(97, 122)) for _ in range(8)])
    run_id += f'_{player.agent_type}'
    if player.agent_type == 'llm':
        run_id += '-' + player.model

    # Initialize the Atari Breakout environment
    env = gym.make('ALE/Breakout-v5', render_mode="rgb_array", obs_type='ram')

    # Output video file
    frames = []  # To store rendered frames

    # Number of episodes to play
    num_episodes = 1  # Adjust as needed

    # Screen dimensions and upscale factor
    screen_width = 160  # Default width of Breakout (adjust if needed)
    upscale_factor = 3  # Upscale factor for text rendering
    total_reward = 0

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
        framecount = 0
        last_action = ACTION_NOOP

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

            # Upscale the frame for better text rendering
            frame_upscaled = cv2.resize(frame, (frame.shape[1] * upscale_factor, frame.shape[0] * upscale_factor))

            # Render text on the upscaled frame
            #y0, dy = 20, 25  # Starting y-coordinate and line spacing for text
            #for i, line in enumerate(description.split('\n')):
            #    y = y0 + i * dy
            #    # Overlay the text (use smaller font size for readability)
            #    cv2.putText(frame_upscaled, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

            # Store the final upscaled frame for video
            frames.append(frame_upscaled)

            # Take a random action (replace with policy if available)
            #action = env.action_space.sample()
            action = ACTION_FIRE
            if ball_prev[0] == ball_curr[0] and ball_prev[1] == ball_curr[1]:
                action = ACTION_FIRE
            else:
                if desc_data:
                    desc_data['last_action'] = last_action
                    action = player.get_action(desc_data)
                    #print(f'Selected action {action_str(action)}')
                else:
                    action = ACTION_NOOP

            last_action = action

            state, reward, done, truncated, info = env.step(action)

            total_reward += reward

            # Update previous positions
            ball_prev = ball_curr
            paddle_prev = paddle_curr

            if player.agent_type == 'llm':
                # Save in progress logs
                if len(player.log) > 0 and framecount % 10 == 0:
                    save_log(player, run_id+f'-f{framecount}', total_reward=None)

                if framecount % 30 == 1:
                    save_video(frames, run_id, framecount, total_reward)


            framecount += 1

        print(f"Episode {episode + 1} finished with a total reward of {total_reward}")

    # Save frames as a video
    save_video(frames, run_id, framecount, total_reward)

    print('Writing log...')
    logfile = save_log(player, run_id, total_reward)
    print(f'Saved log to {logfile}')

    # Close the environment
    env.close()


try:
    import os
    os.mkdir('runs')
except:
    pass

run()
run()
run()
#for _ in range(100): run()


