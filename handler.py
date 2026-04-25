import runpod

def handler(event):
    return {"message": "ok"}

runpod.serverless.start({"handler": handler})