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

