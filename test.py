import requests
import json

# Set the base URI
base_uri = "https://llm.leibmann.org/v1"

# 1. Get available models
models_endpoint = f"{base_uri}/models"
response = requests.get(models_endpoint)
response.raise_for_status()
models = response.json()

# Use the first returned model
first_model = models['data'][0]['id']

# 2. Create a completion for "hello world" using the first model
completion_endpoint = f"{base_uri}/completions"
headers = {
    "Content-Type": "application/json"
}

payload = {
    "model": first_model,
    "prompt": "hello world",
    "max_tokens": 50
}

response = requests.post(completion_endpoint, headers=headers, data=json.dumps(payload))
response.raise_for_status()
completion = response.json()

# Output the completion
print(completion['choices'][0]['text'])
