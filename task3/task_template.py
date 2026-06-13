import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.models import resnet18


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "train.npz"
MODEL_SAVE_PATH = BASE_DIR / "model.pt"

data = np.load(DATA_PATH)

images = torch.from_numpy(data["images"]).float() / 255.0
labels = torch.from_numpy(data["labels"]).long()

dataset = TensorDataset(images, labels)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_ds,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=256,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 9)
model = model.to(device)

criterion = nn.CrossEntropyLoss()


epochs = 80

optimizer = optim.SGD(
    model.parameters(),
    lr=0.05,            
    momentum=0.9,
    weight_decay=5e-4
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs
)


eps = 8 / 255
alpha = 2 / 255
steps = 7

def pgd_attack(model, x, y):
    x_up = nn.functional.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
    x_adv = x_up.detach().clone()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = criterion(logits, y)

        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        
        delta = torch.clamp(x_adv - x_up, min=-eps, max=eps)
        x_adv = torch.clamp(x_up + delta, 0, 1).detach()

    return x_adv


best_val_acc = 0.0

print("Launching Refined Robust Optimization Scheme...")
for epoch in range(epochs):
    model.train()
    train_correct = 0
    train_total = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        x_clean_up = nn.functional.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        
        x_adv = pgd_attack(model, x, y)

        logits_clean = model(x_clean_up)
        logits_adv = model(x_adv)

        loss_clean = criterion(logits_clean, y)
        loss_adv = criterion(logits_adv, y)

        loss = 0.5 * loss_clean + 0.5 * loss_adv

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = logits_clean.argmax(dim=1)
        train_total += y.size(0)
        train_correct += (pred == y).sum().item()

    scheduler.step()
    train_acc = train_correct / train_total


    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            x_up = nn.functional.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
            
            logits = model(x_up)
            pred = logits.argmax(dim=1)
            val_total += y.size(0)
            val_correct += (pred == y).sum().item()

    val_acc = val_correct / val_total
    print(f"Epoch {epoch+1:03d}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f" Saved structural checkpoint (val_acc={val_acc:.4f})")

class ServerCompliantWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        x_up = nn.functional.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        return self.base_model(x_up)

final_eval_model = ServerCompliantWrapper(model)
final_eval_model.base_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
final_eval_model.eval()

with torch.no_grad():
    out = final_eval_model(torch.randn(1, 3, 32, 32).to(device))

print("\n Sanity Test Results:")
print("Output tensor footprint:", out.shape)
assert out.shape == (1, 9), "Output dimension validation breach!"
print("Target weights saved successfully.")