import runpod
import torch
from PIL import Image
import base64
import io
import torchvision.transforms as transforms
import torchvision.models as models

model = models.resnet18(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def handler(event):
    image_b64 = event["input"]["image"]

    image = Image.open(
        io.BytesIO(base64.b64decode(image_b64))
    ).convert("RGB")

    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1)
        conf, pred = torch.max(prob, 1)

    return {
        "class_index": int(pred.item()),
        "confidence": float(conf.item())
    }

runpod.serverless.start({"handler": handler})