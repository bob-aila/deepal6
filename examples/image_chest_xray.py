"""
examples/image_chest_xray.py
-----------------------------
Reproduces the thesis Chapter 5 (Healthcare domain) experiment on the NIH
Chest X-ray dataset. Adapt DATA_DIR and LABELS_CSV for your setup.

Requires: pip install deepal6[image]

Usage
-----
    python examples/image_chest_xray.py
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms

from deepal6 import ActiveLearner, ImageDataModule, ALConfig


SEED      = 42
DATA_DIR  = "/path/to/nih-chest-xray/data"         # ← change this
LABELS_CSV = os.path.join(DATA_DIR, "Data_Entry_2017.csv")
MAX_SAMPLES = 1000   # cap for manageable AL runtime
IMG_SIZE    = 256


# ── 1. Build binary DataFrame (Pneumonia vs Normal) ──────────────────────────
image_paths    = glob.glob(DATA_DIR + "/images_*/*/*.png")
img_path_lookup = {os.path.basename(p): p for p in image_paths}

df = pd.read_csv(LABELS_CSV)
df = df[df["Image Index"].isin(img_path_lookup)].copy()

df["label"] = df["Finding Labels"].apply(
    lambda x: 1 if "Pneumonia" in x else (0 if x == "No Finding" else -1)
)
df_binary = df[df["label"] >= 0].copy()
df_binary["filepath"] = df_binary["Image Index"].map(img_path_lookup)

# Stratified cap
df_binary = (
    df_binary.groupby("label", group_keys=False)
    .apply(lambda g: g.sample(min(len(g), MAX_SAMPLES // 2), random_state=SEED))
    .reset_index(drop=True)
)

train_df, test_df = train_test_split(
    df_binary, test_size=0.2, stratify=df_binary["label"], random_state=SEED
)
train_df = train_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

print(f"Train: {len(train_df)} | Test: {len(test_df)}")
print(f"Train Pneumonia rate: {train_df['label'].mean():.3f}")


# ── 2. Custom augmentations (optional) ──────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ── 3. DataModule ────────────────────────────────────────────────────────────
data = ImageDataModule(
    train_df,
    test_df,
    train_transform = train_transform,
    test_transform  = test_transform,
    img_size        = IMG_SIZE,
    pos_label       = 1,   # Pneumonia is the minority / positive class
    num_workers     = 2,
    dropout_rate    = 0.4,
)


# ── 4. Configure experiment ──────────────────────────────────────────────────
config = ALConfig(
    strategy     = ["Random", "Entropy", "Margin", "BALD", "CoreSet", "BADGE"],
    initial_size = 50,
    batch_size   = 20,
    n_rounds     = 20,
    n_seeds      = 5,
    train_epochs = 10,    # fine-tuning: fewer epochs than tabular
    lr           = 1e-4,  # lower LR for pretrained ResNet
    weight_decay = 1e-4,
    dropout_rate = 0.4,
    mc_passes    = 20,
    verbose      = True,
)


# ── 5. Run & visualise ───────────────────────────────────────────────────────
learner = ActiveLearner(data, config)
results = learner.run()

learner.summary_table(results, metric="auc")
learner.plot(results, metric="auc",     save_path="xray_auc.png")
learner.plot(results, metric="bal_acc", save_path="xray_bal_acc.png")
learner.plot(results, metric="recall",  save_path="xray_recall.png")
learner.plot_calibration(results,       save_path="xray_ece.png")
