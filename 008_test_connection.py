import requests
# This is designed for testing the Text Generation UI api mode.
API_URL = "http://127.0.0.1:5000/v1/chat/completions"
payload = {
    "prompt": "Say hello to the world.",
    "temperature": 0.7,
    "max_new_tokens": 50
}

r = requests.post(API_URL, json=payload)
print(r.status_code)
print(r.json())
