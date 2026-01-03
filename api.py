import torch
from torch import nn
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import clip
import io
from torchvision import transforms

# -------------------------------
# App
# -------------------------------
app = FastAPI(title="CLIP Dual-Head API")

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Load CLIP (frozen)
# -------------------------------
clip_model, _ = clip.load("ViT-B/32", device=device)
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

# -------------------------------
# Load checkpoint
# -------------------------------
checkpoint = torch.load("clip_dual_linear.pth", map_location=device)

material_classes = checkpoint["material_classes"]
epoch_classes = checkpoint["epoch_classes"]

# -------------------------------
# Material head
# -------------------------------
material_head = nn.Linear(
    clip_model.visual.output_dim,
    len(material_classes)
).to(device)
material_head.load_state_dict(checkpoint["material_state"])
material_head.eval()

# -------------------------------
# Epoch head
# -------------------------------
epoch_head = nn.Linear(
    clip_model.visual.output_dim,
    len(epoch_classes)
).to(device)
epoch_head.load_state_dict(checkpoint["epoch_state"])
epoch_head.eval()

# -------------------------------
# Transform (CLIP-style)
# -------------------------------
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711)
    )
])

# -------------------------------
# Predict
# -------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = clip_model.encode_image(img)
        features = features / features.norm(dim=-1, keepdim=True)
        features = features.float()

        # material
        mat_logits = material_head(features)
        mat_probs = torch.softmax(mat_logits, dim=1)
        mat_conf, mat_idx = mat_probs.max(dim=1)

        # epoch
        epoch_logits = epoch_head(features)
        epoch_probs = torch.softmax(epoch_logits, dim=1)
        epoch_conf, epoch_idx = epoch_probs.max(dim=1)

    return {
        "material": material_classes[mat_idx.item()],
        "material_confidence": round(mat_conf.item(), 4),
        "epoch": epoch_classes[epoch_idx.item()],
        "epoch_confidence": round(epoch_conf.item(), 4),
    }
