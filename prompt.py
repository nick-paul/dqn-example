PROMPT_FULL_V1 = """
Instructions: We are playing a game of Atari Breakout. The task is to decide how to move the paddle based on the current and previous ball and paddle positions, as well as the speed of the paddle and ball. The goal is to prevent the ball from missing the paddle and achieve the highest score.

Inputs (provided for each frame):

    ball_curr (tuple): The current position of the ball (x, y).
    ball_prev (tuple): The previous position of the ball (x, y).
    paddle_curr (int): The current x-position of the paddle.
    paddle_prev (int): The previous x-position of the paddle.
    prev_action (str): The action taken in the previous frame.

Important Considerations:

    Paddle Width: The paddle is 16 pixels wide. The ball should ideally land within this range, not necessarily in the center of the paddle. We need to aim for a target range that spans ±8 pixels from the paddle's center.

    Braking Distance: The paddle moves quickly, and the AI must prevent it from overshooting the predicted ball position. This is the braking distance. The paddle will stop moving if it’s within a reasonable distance from the target area (defined by the predicted ball position and the 16-pixel target range).

    Ball Prediction:
        Use the ball’s velocity (based on the difference between ball_curr and ball_prev) to predict where the ball will be when it reaches y=180 (the y-position of the paddle).
        If the ball is moving towards the paddle, predict the x position at which the ball will land on the paddle.
        If the ball is moving away from the paddle, anticipate where the ball will move based on its trajectory.

    Paddle Movement Logic:
        Stay if the ball is within ±8 pixels of the paddle's center.
        Move left if the predicted ball position is to the left of the paddle's target range.
        Move right if the predicted ball position is to the right of the paddle's target range.
        The paddle should avoid overshooting. If the ball is within the braking distance, the paddle should stop moving to avoid missing it.

    Screen coordinates:
        The left and right bounds of the screen are from x=50 to x=200 pixels
        The paddle is located at y=180
        As the ball moves down it gets closer to 180, as it moves up it gets closer to 0

Task:

    Based on the above inputs and considerations, decide whether the paddle should move LEFT, RIGHT, or STAY in order to prevent missing the ball and achieve the highest score.

Here are the inputs for the current frame:

    ball_curr: [ball_curr]
    ball_prev: [ball_prev]
    paddle_curr: [paddle_curr]
    paddle_prev: [paddle_prev]
    prev_action: [prev_action]

Which action do I take? LEFT, RIGHT, or STAY?

"""


PROMPT_SIMPLE_V1 = """
Task: Decide whether the paddle should move left, right, or stay in its current position to prevent missing the ball.

Inputs:

    ball_curr (tuple): The current position of the ball (x, y).
    ball_prev (tuple): The previous position of the ball (x, y).
    paddle_curr (int): The current x-position of the paddle.
    paddle_prev (int): The previous x-position of the paddle.
    screen_width (int): The width of the screen, between 50 and 200.

Screen coordinates:
    The left and right bounds of the screen are from x=50 to x=200 pixels
    The paddle is located at y=180
    As the ball moves down it gets closer to 180, as it moves up it gets closer to 0

Instructions:

    Calculate Ball Movement:
        Find the ball’s horizontal speed by comparing ball_curr[0] and ball_prev[0].
        Predict where the ball will land by estimating its position when it reaches the paddle at y=180.

    Set Target Range:
        The paddle is 16 pixels wide, so we want the ball to land within ±8 pixels of the paddle's current position.
        Set the target range from paddle_curr - 8 to paddle_curr + 8.

    Decision Making:
        If the ball is within the target range (±8 pixels of the paddle's center), STAY.
        If the ball is to the left of the target range, MOVE LEFT.
        If the ball is to the right of the target range, MOVE RIGHT.

    Stopping Overshoot:
        If the paddle is already close enough to the target position (within ±5 pixels of the target), STAY to avoid overshooting.

Here are the inputs for the current frame:

    ball_curr: [ball_curr]
    ball_prev: [ball_prev]
    paddle_curr: [paddle_curr]
    paddle_prev: [paddle_prev]
    prev_action: [prev_action]

Which action do I take? LEFT, RIGHT, or STAY?
"""

def make_prompt(template, ball_curr, ball_prev, paddle_curr, paddle_prev, prev_action):
    return (
        template
        .replace('[ball_curr]', ball_curr)
        .replace('[ball_prev]', ball_prev)
        .replace('[paddle_curr]', paddle_curr)
        .replace('[paddle_prev]', paddle_prev)
        .replace('[prev_action]', prev_action)
    )
