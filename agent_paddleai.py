from actions import ACTION_RIGHT, ACTION_LEFT, ACTION_NOOP, ACTION_FIRE

def breakout_ai(ball_curr, ball_prev, paddle_curr, paddle_prev, screen_width, prev_action):
    # Ball velocity
    ball_velocity_x = ball_curr[0] - ball_prev[0]
    ball_velocity_y = ball_curr[1] - ball_prev[1]

    # Paddle speed
    paddle_speed = abs(paddle_curr - paddle_prev)
    #paddle_speed = 10

    # Predict ball x-position when it reaches y=180
    if ball_velocity_y != 0:  # Avoid division by zero
        steps_to_paddle = (180 - ball_curr[1]) / ball_velocity_y
        predicted_x = ball_curr[0] + ball_velocity_x * steps_to_paddle
    else:
        predicted_x = ball_curr[0]

    # Reflect predicted_x if it goes out of bounds
    while predicted_x < 50 or predicted_x > 200:
        if predicted_x < 50:
            predicted_x = 50 + (50 - predicted_x)
        elif predicted_x > 200:
            predicted_x = 200 - (predicted_x - 200)

    # Define tolerance and the 8-pixel range for the paddle
    tolerance = 5  # Allow small tolerance to prevent jitter
    target_range_left = paddle_curr - 8  # 8 pixels to the left of the paddle center
    target_range_right = paddle_curr + 8  # 8 pixels to the right of the paddle center

    # Predict paddle's future position
    if prev_action == "RIGHT":
        future_paddle_pos = paddle_curr + paddle_speed
    elif prev_action == "LEFT":
        future_paddle_pos = paddle_curr - paddle_speed
    else:
        future_paddle_pos = paddle_curr

    # Define braking distance (paddle speed)
    braking_distance = paddle_speed

    # Decide paddle action based on predicted ball position and target range
    if target_range_left <= predicted_x <= target_range_right:
        action = "STAY"  # Ball is within the target range, no movement needed
    elif predicted_x < target_range_left:
        # Ball is to the left of the target range
        if future_paddle_pos > target_range_left - braking_distance:
            action = "LEFT"  # Paddle is far enough away to keep moving left
        else:
            action = "STAY"  # Prevent overshooting by stopping
    elif predicted_x > target_range_right:
        # Ball is to the right of the target range
        if future_paddle_pos < target_range_right + braking_distance:
            action = "RIGHT"  # Paddle is far enough away to keep moving right
        else:
            action = "STAY"  # Prevent overshooting by stopping
    else:
        # Fallback; shouldn't get here
        action = "STAY"

    return action

class PaddleAIAgent:
    def __init__(self):
        self.log = [] # unused
        self.agent_type = 'paddleai'

    def get_action(self, frame_description: dict) -> int:
        output = breakout_ai(
            ball_curr=frame_description['ball_curr'],
            ball_prev=frame_description['ball_prev'],
            paddle_curr=frame_description['paddle_curr'],
            paddle_prev=frame_description['paddle_prev'],
            screen_width=frame_description['screen_width'],
            prev_action={
                ACTION_LEFT: 'left',
                ACTION_RIGHT: 'right',
                ACTION_NOOP: 'stay',
                ACTION_FIRE: 'stay'
            }[frame_description['last_action']].upper(),
        )

        return {
            'left': ACTION_LEFT,
            'right': ACTION_RIGHT,
            'stay': ACTION_NOOP,
        }[output.lower()]
