import os
import sys
import torch
import pandas as pd
import requests
import random
import argparse

from pathlib import Path
from torch.utils.data import Dataset
from torchvision.models import resnet18
import torchvision.transforms as transforms

# config
BASE = Path(__file__).parent
PUB_PATH = BASE / "pub.pt"
PRIV_PATH = BASE / "priv.pt"
MODEL_PATH = BASE / "model.pt"
OUTPUT_CSV = BASE / "submission.csv"

BASE_URL = "http://34.63.153.158"   #DONOT CHANGE
API_KEY = "e0c557b2952bebef969bd58692e1205e"
TASK_ID = "01-mia"  #DONOT CHANGE



# dataset classes
class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index):
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)


class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index):
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]


# load datasets
print("Loading datasets...")
pub_ds = torch.load(PUB_PATH, weights_only=False)
priv_ds = torch.load(PRIV_PATH, weights_only=False)


# normalization (same as training)
MEAN = [0.7406, 0.5331, 0.7059]
STD = [0.1491, 0.1864, 0.1301]

base_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Normalize(mean=MEAN, std=STD),
])

aug_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize((32, 32)),
    transforms.Normalize(mean=MEAN, std=STD),
])
pub_ds.transform = base_transform
priv_ds.transform = base_transform


# load model
print("Loading model...")
model = resnet18(weights=None)
model.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
model.maxpool = torch.nn.Identity()
model.fc = torch.nn.Linear(512, 9)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()


# create random submission (remove this later or it will rewrite your actual submission)
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np

print("Running membership inference attack...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def safe_collate(batch):
    batch = [b for b in batch if b is not None]

    if len(batch[0]) == 4:
        ids, imgs, labels, _ = zip(*batch)   
    else:
        ids, imgs, labels = zip(*batch)

    return list(ids), torch.stack(imgs), torch.tensor(labels)

loader = DataLoader(
    priv_ds,
    batch_size=64,
    shuffle=False,
    collate_fn=safe_collate
)
criterion = torch.nn.CrossEntropyLoss(reduction="none")

all_ids = []
all_scores = []

with torch.no_grad():
    for ids_batch, imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        logits = model(imgs)
        probs = F.softmax(logits, dim=1)

        aug_losses = []
        for _ in range(5): 
            aug_imgs = torch.stack([
                aug_transform(img) for img in imgs
            ]).to(device)
            aug_logits = model(aug_imgs)
            aug_losses.append(criterion(aug_logits, labels))
        
        avg_aug_loss = torch.stack(aug_losses).mean(dim=0)
        scores = -avg_aug_loss 
        
        all_ids.extend(ids_batch)
        all_scores.extend(scores.cpu().numpy().tolist())
    
# normalize scores (important)
scores = np.array(all_scores)
scores = torch.tensor(all_scores)
scores = torch.sigmoid(scores)   # keeps [0,1] range naturally

df = pd.DataFrame({
    "id": [str(i) for i in all_ids],
    "score": scores
})

df.to_csv(OUTPUT_CSV, index=False)
print("Saved:", OUTPUT_CSV)

# submit
def die(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)

parser = argparse.ArgumentParser(description="Submit a CSV file to the server.")
args = parser.parse_args()

submit_path = OUTPUT_CSV

if not submit_path.exists():
    die(f"File not found: {submit_path}")

try:
    with open(submit_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/submit/{TASK_ID}",
            headers={"X-API-Key": API_KEY},
            files={"file": (submit_path.name, f, "application/csv")},
            timeout=(10, 600),
        )
    try:
        body = resp.json()
    except Exception:
        body = {"raw_text": resp.text}

    if resp.status_code == 413:
        die("Upload rejected: file too large (HTTP 413).")

    resp.raise_for_status()

    print("Successfully submitted.")
    print("Server response:", body)
    submission_id = body.get("submission_id")
    if submission_id:
        print(f"Submission ID: {submission_id}")

except requests.exceptions.RequestException as e:
    detail = getattr(e, "response", None)
    print(f"Submission error: {e}")
    if detail is not None:
        try:
            print("Server response:", detail.json())
        except Exception:
            print("Server response (text):", detail.text)
    sys.exit(1)