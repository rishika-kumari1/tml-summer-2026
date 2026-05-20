import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from safetensors.torch import load_file
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Subset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_MODEL_PATH = "./target_model/weights.safetensors"
TRAIN_INDICES_PATH = "./target_model/train_main_idx.json"
SUSPECT_MODELS_DIR = "./suspect_models"

OUTPUT_CSV = "submission.csv"

BATCH_SIZE = 128
NUM_WORKERS = 2
MAX_IMAGES = 1000


def make_model():

    model = resnet18(weights=None)

    model.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )

    model.maxpool = nn.Identity()

    model.fc = nn.Linear(
        model.fc.in_features,
        100
    )

    return model


def load_model(path):

    model = make_model()

    state_dict = load_file(path, device="cpu")

    model.load_state_dict(state_dict, strict=True)

    model.to(DEVICE)

    model.eval()

    return model


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5071, 0.4867, 0.4408),
        (0.2675, 0.2565, 0.2761),
    ),
])


train_dataset = datasets.CIFAR100(
    root="./cifar100",
    train=True,
    download=True,
    transform=transform,
)

test_dataset = datasets.CIFAR100(
    root="./cifar100",
    train=False,
    download=True,
    transform=transform,
)


with open(TRAIN_INDICES_PATH, "r") as f:
    train_indices = json.load(f)


train_indices = train_indices[:MAX_IMAGES]


train_dataset = Subset(
    train_dataset,
    train_indices
)

test_dataset = Subset(
    test_dataset,
    list(range(MAX_IMAGES))
)


combined_dataset = torch.utils.data.ConcatDataset([
    train_dataset,
    test_dataset
])


loader = DataLoader(
    combined_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)


def get_feature_extractor(model):

    return nn.Sequential(
        *list(model.children())[:-1]
    )


@torch.no_grad()
def extract_features_and_logits(model):

    extractor = get_feature_extractor(model)

    extractor.eval()

    all_features = []
    all_logits = []
    all_preds = []

    for x, _ in loader:

        x = x.to(DEVICE)

        logits = model(x)

        preds = logits.argmax(1)

        features = extractor(x)

        features = features.flatten(1)

        all_logits.append(logits.cpu())

        all_preds.append(preds.cpu())

        all_features.append(features.cpu())

    return (
        torch.cat(all_features),
        torch.cat(all_logits),
        torch.cat(all_preds)
    )


def get_weight_vector(model):

    vecs = []

    for k, v in model.state_dict().items():

        if "weight" in k:
            vecs.append(v.flatten().float())

    return torch.cat(vecs)


def weight_similarity(target, suspect):

    t = get_weight_vector(target)

    s = get_weight_vector(suspect)

    return F.cosine_similarity(
        t.unsqueeze(0),
        s.unsqueeze(0)
    ).item()


def bn_similarity(target, suspect):

    t = []

    s = []

    for (k1, v1), (k2, v2) in zip(
        target.state_dict().items(),
        suspect.state_dict().items()
    ):

        if "running_mean" in k1 or "running_var" in k1:

            t.append(v1.flatten())

            s.append(v2.flatten())

    t = torch.cat(t).float()

    s = torch.cat(s).float()

    return F.cosine_similarity(
        t.unsqueeze(0),
        s.unsqueeze(0)
    ).item()


def prediction_agreement(target_preds, suspect_preds):

    return (
        (target_preds == suspect_preds)
        .float()
        .mean()
        .item()
    )


def kl_similarity(target_logits, suspect_logits):

    kl = F.kl_div(
        suspect_logits.log_softmax(dim=1),
        target_logits.softmax(dim=1),
        reduction="batchmean"
    )

    return float(
        torch.exp(-kl)
    )


def feature_similarity(target_features, suspect_features):

    return F.cosine_similarity(
        target_features,
        suspect_features,
        dim=1
    ).mean().item()


print("\nLoading target model...")

target_model = load_model(
    TARGET_MODEL_PATH
)

print("Extracting target features and logits...")

(
    target_features,
    target_logits,
    target_preds
) = extract_features_and_logits(target_model)

print("Target model ready.\n")


model_paths = sorted(
    Path(SUSPECT_MODELS_DIR).glob("*.safetensors")
)

print(f"Found {len(model_paths)} suspect models.\n")


results = []


for idx, model_path in enumerate(model_paths):

    print(
        f"[{idx + 1}/{len(model_paths)}] "
        f"{model_path.name}"
    )

    suspect_model = load_model(
        str(model_path)
    )

    (
        suspect_features,
        suspect_logits,
        suspect_preds
    ) = extract_features_and_logits(
        suspect_model
    )

    weight_score = weight_similarity(
        target_model,
        suspect_model
    )

    bn_score = bn_similarity(
        target_model,
        suspect_model
    )

    prediction_score = prediction_agreement(
        target_preds,
        suspect_preds
    )

    kl_score = kl_similarity(
        target_logits,
        suspect_logits
    )

    feature_score = feature_similarity(
        target_features,
        suspect_features
    )

    final_score = (
        0.10 * weight_score +
        0.10 * bn_score +
        0.30 * prediction_score +
        0.25 * kl_score +
        0.25 * feature_score
    )

    results.append({
        "id": idx,
        "score": float(final_score)
    })

    del suspect_model
    torch.cuda.empty_cache()


scores = torch.tensor([
    r["score"] for r in results
])

scores = (
    scores - scores.mean()
) / (scores.std() + 1e-8)

scores = torch.sigmoid(4 * scores)


for i in range(len(results)):

    results[i]["score"] = float(
        scores[i].item()
    )


submission_df = pd.DataFrame(results)

submission_df = submission_df.sort_values(
    by="id"
).reset_index(drop=True)


assert len(submission_df) == 360
assert submission_df["id"].min() == 0
assert submission_df["id"].max() == 359
assert submission_df["id"].nunique() == 360


submission_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\nSaved submission.csv\n")

print(submission_df.head())

print("\nSubmission shape:", submission_df.shape)