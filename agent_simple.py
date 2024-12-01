from actions import action_str, ACTION_RIGHT, ACTION_LEFT, ACTION_NOOP

class SimpleAgent:
    def __init__(self):
        self.log = [] # unused
        self.agent_type = 'simple'

    def get_action(self, frame_description: dict) -> int:
        rel = frame_description['ball_relative']
        if 'left' in rel:
            action = ACTION_LEFT
        elif 'right' in rel:
            action = ACTION_RIGHT
        else:
            action = ACTION_NOOP
        #print(f'{rel} -> {action_str(action)}')
        return action
