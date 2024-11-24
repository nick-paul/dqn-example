from openai import OpenAI


def get_llm_client(local=True):
    if local:
        client = OpenAI(
            base_url="http://localhost:8080/v1",
            api_key = "sk-no-key-required"
        )
    else:
        raise ValueError("todo")

    return client


client = get_llm_client()


chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="local",
)

print(chat_completion)
