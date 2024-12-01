from openai import OpenAI
from actions import action_str, ACTION_RIGHT, ACTION_LEFT, ACTION_NOOP

class LLMAgent:
    def __init__(self, prompt_generator, local=True):
        self.log = []
        self.agent_type = 'llm'

        # callable(frame_desc: dict) -> prompt: str
        self.prompt_generator = prompt_generator

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
        #if self.use_prompt_template:
        #    prompt = make_prompt(self.prompt, ball_curr, ball_prev, paddle_curr, paddle_prev, prev_action):
        #else:
        #    prompt = self.prompt.replace('[FRAME_DESC]', frame_description['text'])

        prompt = self.prompt_generator(frame_description)
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

            self.log.append({
                'prompt': prompt,
                'response': response,
            })

            str_to_check = response.strip().splitlines()[-1]

            if 'LEFT' in str_to_check and 'RIGHT' in str_to_check:
                print('###\n### ^^ BAD RESPONSE\n###')
                print(response)
            elif 'LEFT' in str_to_check:
                action = ACTION_LEFT
            elif 'RIGHT' in str_to_check:
                action = ACTION_RIGHT
            elif "STAY" in str_to_check:
                action  = ACTION_NOOP
            else:
                print('###\n### ^^ BAD RESPONSE\n###')
                print(response)

        return action



