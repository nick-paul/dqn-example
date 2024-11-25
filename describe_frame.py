def describe_frame(ball_curr, ball_prev, paddle_curr, _, screen_width) -> dict:
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
    # Convert to normal numbers (we don't want the math to overflow)
    ball_curr = (int(ball_curr[0]), int(ball_curr[1]))
    ball_prev = (int(ball_prev[0]), int(ball_prev[1]))
    paddle_curr = int(paddle_curr)
    screen_width = int(screen_width)

    ppos = f'{paddle_curr}'
    bpos = f'{ball_curr[0], ball_curr[1]}'
    prev_bpos = f'{ball_prev[0], ball_prev[1]}'

    # Determine paddle position (left, center, or right)
    paddle_min = 50
    paddle_max = 200
    paddle_width = paddle_max - paddle_min
    paddle_curr_norm = paddle_curr - paddle_min
    if paddle_curr_norm < paddle_width * 0.33:
        paddle_position = "Left side of the screen"
    elif paddle_curr_norm > paddle_width * 0.66:
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
    dxdy = f'{dx},{dy}'
    if dx > 0 and dy > 0:
        ball_direction = "Moving down and to the right"
    elif dx > 0 and dy < 0:
        ball_direction = "Moving up and to the right"
    elif dx < 0 and dy > 0:
        ball_direction = "Moving down and to the left"
    elif dx < 0 and dy < 0:
        ball_direction = "Moving up and to the left"
    else:
        ball_direction = "Stationary"

    # Determine distance from paddle
    PADDLE_Y = 180
    distance = abs(ball_y - PADDLE_Y)
    if distance < 30:
        ball_distance = "Very close"
    elif distance < 70:
        ball_distance = "Mid-range"
    else:
        ball_distance = "Far away"

    # Create the description
    description = (
        f"Frame Update:\n"
        f"    Ball position (X,Y): {bpos}\n"
        f"    Paddle position (X): {ppos}\n"
        f"    Paddle Position: {paddle_position}\n"
        f"    Ball Position: {ball_relative}\n"
        f"    Ball Direction: {ball_direction}\n"
        f"    Ball height relative to paddle: {ball_distance}\n"
    )

    return {
        'text': description,
        'ball_relative': ball_relative,
        'ball_direction': ball_direction,
    }
