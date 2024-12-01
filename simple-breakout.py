import gym
import imageio  # For saving video
import numpy as np

PROMPT = """
Lets play atari breakout. I will describe what is happening on the screen and you will tell me which direction to the move the paddle or to keep it still. Here is the current frame:

FRAME_DESCRIPTION

Which direction should I move the paddle? Say LEFT, RIGHT, or DON'T MOVE
"""

def describe_frame(ball_curr, ball_prev, paddle_curr, screen_width):
    """
    Generates a text description of the game state for a single frame.
    
    Args:
        ball_curr (tuple): Current ball position (x, y).
        ball_prev (tuple): Previous ball position (x, y).
        paddle_curr (int): Current paddle x-position.
        paddle_prev (int): Previous paddle x-position.
        screen_width (int): Width of the screen.
    
    Returns:
        str: Description of the frame.
    """
    # Determine paddle position (left, center, or right)
    if paddle_curr < screen_width * 0.33:
        paddle_position = "Left side of the screen"
    elif paddle_curr > screen_width * 0.66:
        paddle_position = "Right side of the screen"
    else:
        paddle_position = "Center of the screen"

    # Determine ball position relative to paddle
    ball_x, ball_y = ball_curr
    ball_relative_x = ball_x - paddle_curr

    if ball_relative_x < -20:
        ball_relative = "Far to the left of the paddle"
    elif ball_relative_x < 0:
        ball_relative = "Slightly to the left of the paddle"
    elif ball_relative_x < 20:
        ball_relative = "Above and aligned with the paddle"
    else:
        ball_relative = "To the right of the paddle"

    # Determine ball direction (from previous position)
    dx = ball_curr[0] - ball_prev[0]
    dy = ball_curr[1] - ball_prev[1]
    if dx > 0 and dy > 0:
        ball_direction = "Moving down and to the right"
    elif dx > 0 and dy < 0:
        ball_direction = "Moving up and to the right"
    elif dx < 0 and dy > 0:
        ball_direction = "Moving down and to the left"
    elif dx < 0 and dy < 0:
        ball_direction = "Moving up and to the left"
    elif dx == 0 and dy > 0:
        ball_direction = "Moving straight down"
    elif dx == 0 and dy < 0:
        ball_direction = "Moving straight up"
    elif dx > 0 and dy == 0:
        ball_direction = "Moving straight to the right"
    elif dx < 0 and dy == 0:
        ball_direction = "Moving straight to the left"
    else:
        ball_direction = "Stationary"

    # Determine distance from paddle
    distance = abs(ball_y - paddle_curr)
    if distance < 30:
        ball_distance = "Very close"
    elif distance < 70:
        ball_distance = "Mid-range"
    else:
        ball_distance = "Far away"

    # Create the description
    description = (
        f"Frame Update:\n"
        f"    Paddle Position: {paddle_position}\n"
        f"    Ball Position: {ball_relative}\n"
        f"    Ball Direction: {ball_direction}\n"
        f"    Distance from Paddle: {ball_distance}\n"
    )

    return description

# Initialize the Atari Breakout environment
env = gym.make('ALE/Breakout-v5', render_mode="rgb_array", obs_type='ram')

# Output video file
video_filename = "breakout_gameplay.mp4"
frames = []  # To store rendered frames

# Number of episodes to play
num_episodes = 1  # Adjust as needed

frame_count = 0
for episode in range(num_episodes):
    print(f"Starting episode {episode + 1}")
    state = env.reset()
    done = False
    total_reward = 0
    ball_prev = (0, 0)

    while not done:
        # Extract RAM for ball and paddle positions
        ram = env.unwrapped.ale.getRAM()
        ball_x = ram[99]    # X-position of the ball
        ball_y = ram[101]   # Y-position of the ball
        paddle_x = ram[72]  # X-position of the paddle

        prompt = PROMPT.replace('FRAME_DESCRIPTION', describe_frame(
            ball_curr=(ball_x, ball_y),
            ball_prev=ball_prev,
            paddle_curr=paddle_x,
            screen_width=200,
        ))

        print(prompt)

        # Render frame and store it
        frame = env.render()
        frames.append(frame)

        # Take a random action (replace with policy if available)
        action = env.action_space.sample()
        state, reward, done, truncated, info = env.step(action)

        total_reward += reward

        ball_prev = (ball_x, ball_y)

        frame_count += 1
        if frame_count > 100:
            done = True


    print(f"Episode {episode + 1} finished with a total reward of {total_reward}")

# Save frames as a video
print(f"Saving video to {video_filename}...")
imageio.mimsave(video_filename, frames, fps=30)
print(f"Video saved successfully!")

# Close the environment
env.close()
