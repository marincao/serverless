import requests
import base64

API_KEY = "rpa_47FWOFGYU8L65NGC6GTUJZSFPZ19743USIDPK4PGx9wdcx"
URL = "https://api.runpod.ai/v2/5fa2rv2rr7vua3/run"

# convert image to base64
with open("test.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

payload = {
    "input": {
        "image": img_b64
    }
}

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

res = requests.post(URL, json=payload, headers=headers)

print(res.json())