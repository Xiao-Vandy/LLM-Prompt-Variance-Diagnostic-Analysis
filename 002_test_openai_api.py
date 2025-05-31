
import openai
openai.api_key="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"

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Say hello"}]
)

print(response.choices[0].message["content"])