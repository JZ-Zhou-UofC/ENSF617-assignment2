import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
print(f"Same as tutorial")
# =========================
# CONFIG
# =========================

data_dir = "/work/TALC/ensf617_2026w/garbage_data"

train_dir = os.path.join(data_dir, "CVPR_2024_dataset_Train")
val_dir   = os.path.join(data_dir, "CVPR_2024_dataset_Val")
test_dir  = os.path.join(data_dir, "CVPR_2024_dataset_Test")

BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.0005

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# =========================
# TRANSFORMS
# =========================

transform = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),

    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),

    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
}

# =========================
# DATASETS
# =========================

image_datasets = {
    "train": datasets.ImageFolder(train_dir, transform["train"]),
    "val": datasets.ImageFolder(val_dir, transform["val"]),
    "test": datasets.ImageFolder(test_dir, transform["test"]),
}

dataloaders = {
    x: DataLoader(
        image_datasets[x],
        batch_size=BATCH_SIZE,
        shuffle=(x == "train"),
        num_workers=4,
        pin_memory=True
    )
    for x in ["train", "val", "test"]
}

num_classes = len(image_datasets["train"].classes)

print("Classes:", image_datasets["train"].classes)

# =========================
# MODEL
# =========================

model = models.mobilenet_v2(weights="DEFAULT")

# Freeze feature extractor
for param in model.features.parameters():
    param.requires_grad = False

# Replace classifier with deeper head
in_features = model.classifier[1].in_features

model.classifier = nn.Sequential(

    nn.Linear(in_features, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.5),

    nn.Linear(512, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.3),

    nn.Linear(128, num_classes)
)

model = model.to(device)

# =========================
# LOSS & OPTIMIZER
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.classifier.parameters(),
    lr=LEARNING_RATE
)

# =========================
# TRAIN FUNCTION
# =========================

def train_model():

    best_acc = 0.0

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        for phase in ["train", "val"]:

            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_correct = 0

            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):

                    outputs = model(inputs)

                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_correct += torch.sum(preds == labels)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_correct.double() / len(image_datasets[phase])

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:

                best_acc = epoch_acc

                torch.save(model.state_dict(), "best_model.pth")

                print("Saved best_model.pth")

    print("Best val accuracy:", best_acc)

# =========================
# TEST FUNCTION
# =========================

def test_model():

    model.load_state_dict(torch.load("best_model.pth"))

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for inputs, labels in dataloaders["test"]:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)

            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

    print(f"\nTest Accuracy: {100*correct/total:.2f}%")

# =========================
# RUN
# =========================

train_model()

test_model()