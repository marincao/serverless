import runpod
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import base64
import io

# Load model once (important)
model = models.resnet18(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def handler(event):
    try:
        # Get image from request
        image_b64 = event["input"]["image"]

        # Decode
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocess
        x = transform(image).unsqueeze(0)

        # Inference
        with torch.no_grad():
            out = model(x)
            probs = torch.softmax(out, dim=1)
            conf, pred = torch.max(probs, 1)

        return {
            "class_index": int(pred.item()),
            "confidence": float(conf.item())
        }

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})