
import openai
openai.api_key="Your Key"

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Say hello"}]
)

print(response.choices[0].message["content"])