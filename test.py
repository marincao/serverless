import requests
import base64
import time

API_KEY = "rpa_47FWOFGYU8L65NGC6GTUJZSFPZ19743USIDPK4PGx9wdcx"
ENDPOINT_ID = "5fa2rv2rr7vua3"

RUN_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"
STATUS_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status/"

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

# encode image
with open("test.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

# send request
res = requests.post(RUN_URL, json={"input": {"image": img_b64}}, headers=headers)
job_id = res.json()["id"]

print("job_id:", job_id)

# poll loop
while True:
    r = requests.get(STATUS_URL + job_id, headers=headers).json()

    print("status:", r["status"])

    if r["status"] == "COMPLETED":
        print("RESULT:", r["output"])
        break

    if r["status"] == "FAILED":
        print("ERROR:", r)
        break

    time.sleep(2)