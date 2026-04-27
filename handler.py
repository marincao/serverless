"""
RunPod Serverless Handler — Crystal Microscopy Grading
Accepts image via:
  - base64 encoded string  (field: "image")
  - public URL             (field: "image_url")

Returns JSON:
{
  "prediction": "level_7",
  "confidence": 0.843,
  "scores": {
    "level_4": 0.012,
    "level_5": 0.021,
    "level_6": 0.084,
    "level_7": 0.843,
    "level_8": 0.040
  }
}
"""

import base64
import io
import os
import time
import traceback
from pathlib import Path

import numpy as np
import requests
import runpod
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ── Config ────────────────────────────────────────
LABEL_NAMES      = ["level_4", "level_5", "level_6", "level_7", "level_8"]
CHECKPOINT_DIR   = Path(os.environ.get("CHECKPOINT_DIR", "/app/checkpoints"))
PATCH_SIZES      = (256, 512)
IMG_SIZE         = 224
STRIDE_RATIO     = 0.5
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model definition (must match training exactly) ─
def build_model(num_classes=5):
    model = models.efficientnet_b3(weights=None)
    in_feat = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.35),
        nn.Linear(in_feat, 384),
        nn.SiLU(),
        nn.Dropout(0.15),
        nn.Linear(384, num_classes),
    )
    return model


# ── Load all fold models at cold-start ────────────
def load_fold_models():
    ckpt_files = sorted(CHECKPOINT_DIR.glob("fold_*.pth"))
    if not ckpt_files:
        raise FileNotFoundError(
            f"No fold_*.pth checkpoints found in {CHECKPOINT_DIR}. "
            "Make sure your repo contains the checkpoints/ folder."
        )
    loaded = []
    for ckpt in ckpt_files:
        m = build_model().to(DEVICE)
        m.load_state_dict(
            torch.load(ckpt, map_location=DEVICE, weights_only=True)
        )
        m.eval()
        loaded.append(m)
        print(f"[startup] loaded {ckpt.name}")
    print(f"[startup] {len(loaded)} fold models ready on {DEVICE}")
    return loaded


# Load once at container startup (cold start), reused for all warm requests
MODELS = load_fold_models()

VAL_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


# ── Core inference ─────────────────────────────────
@torch.no_grad()
def predict_pil(img: Image.Image) -> dict:
    """Run ensemble inference on a PIL image, return result dict."""
    img = img.convert("L")
    img = Image.merge("RGB", [img, img, img])
    w, h = img.size

    all_patches = []
    for ps in PATCH_SIZES:
        stride = max(1, int(ps * STRIDE_RATIO))
        cur = img
        cw, ch = w, h
        if cw < ps or ch < ps:
            scale = ps / min(cw, ch) * 1.05
            cur   = img.resize((int(cw * scale), int(ch * scale)), Image.BILINEAR)
            cw, ch = cur.size

        # Fix: pass loop variables as default args to avoid closure capture bug
        def extract(im, _ps=ps, _cw=cw, _ch=ch, _stride=stride):
            for y in range(0, _ch - _ps + 1, _stride):
                for x in range(0, _cw - _ps + 1, _stride):
                    crop = im.crop((x, y, x + _ps, y + _ps))
                    all_patches.append(
                        VAL_TF(crop.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
                    )

        extract(cur)
        extract(cur.transpose(Image.FLIP_LEFT_RIGHT))
        extract(cur.transpose(Image.FLIP_TOP_BOTTOM))

    if not all_patches:
        all_patches = [VAL_TF(img.resize((IMG_SIZE, IMG_SIZE)))]

    batch = torch.stack(all_patches).to(DEVICE)

    fold_probs = []
    for model in MODELS:
        chunk_probs = []
        for i in range(0, len(batch), 64):
            chunk_probs.append(torch.softmax(model(batch[i : i + 64]), dim=1))
        fold_probs.append(torch.cat(chunk_probs).mean(0).cpu().numpy())

    avg_probs  = np.mean(fold_probs, axis=0)
    pred_idx   = int(np.argmax(avg_probs))
    pred_label = LABEL_NAMES[pred_idx]
    confidence = float(avg_probs[pred_idx])
    scores     = {name: round(float(p), 4) for name, p in zip(LABEL_NAMES, avg_probs)}

    return {
        "prediction": pred_label,
        "confidence": round(confidence, 4),
        "scores":     scores,
    }


# ── Image loading helpers ──────────────────────────
def load_image_from_base64(b64_str: str) -> Image.Image:
    # Strip data-URI prefix if present (e.g. "data:image/jpeg;base64,...")
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(data))


def load_image_from_url(url: str) -> Image.Image:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content))


# ── RunPod handler ─────────────────────────────────
def handler(job: dict) -> dict:
    """
    Expected job input (one of):
      { "input": { "image":     "<base64 string>" } }
      { "input": { "image_url": "https://..."     } }
    """
    t0 = time.time()
    try:
        job_input = job.get("input", {})

        # ── resolve image ──────────────────────────
        if "image" in job_input:
            img = load_image_from_base64(job_input["image"])
        elif "image_url" in job_input:
            img = load_image_from_url(job_input["image_url"])
        else:
            return {
                "error": "No image provided. Send 'image' (base64) or 'image_url' (URL)."
            }

        # ── run inference ──────────────────────────
        result = predict_pil(img)
        result["inference_time_ms"] = round((time.time() - t0) * 1000, 1)
        return result

    except Exception as e:
        return {
            "error":   str(e),
            "details": traceback.format_exc(),
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
