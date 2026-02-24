import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# =========================
# What has been done to improve the accucracy
# 1. Used EfficientNet-B2 as the based model
# 2. Input Augmentation: randomly rotate, horizontal flip, randomly change the color property
# 3. gradual unfreezing: unfreezed the last 4 layers at epoch 5 with a smaller learning rate to learn the output features
# 4. freeze batchnorm. We think the batchnorm for the EfficientNet-B2 will be better than the batchnorm produced by our data set
# 5. replaced the classification layer with our own layers which has dropout to combat overfitting 
# 6. save the best model based on valdiation accuracy. Best model produced at epoch 11.
# =========================

# =========================
# CONFIG
# =========================

data_dir = "/work/TALC/ensf617_2026w/garbage_data"

train_dir = os.path.join(data_dir, "CVPR_2024_dataset_Train")
val_dir = os.path.join(data_dir, "CVPR_2024_dataset_Val")
test_dir = os.path.join(data_dir, "CVPR_2024_dataset_Test")

BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.0005

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# =========================
# FREEZE BATCHNORM FUNCTION
# =========================


def freeze_batchnorm(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False


# =========================
# TRANSFORMS
# =========================

transform = {

    "train": transforms.Compose([

        transforms.RandomResizedCrop(
            288,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1)
        ),

        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomRotation(10),

        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ]),

    "val": transforms.Compose([

        transforms.Resize((288, 288)),
        transforms.CenterCrop(288),
        transforms.ToTensor(),

        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ]),

    "test": transforms.Compose([

        transforms.Resize((288, 288)),
        transforms.CenterCrop(288),
        transforms.ToTensor(),

        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ]),
}

# =========================
# DATALOADER
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
        pin_memory=True,
    )
    for x in ["train", "val", "test"]
}

num_classes = len(image_datasets["train"].classes)

print("Classes:", image_datasets["train"].classes)

# =========================
# MODEL: EfficientNet-B2
# =========================

model = models.efficientnet_b2(weights="DEFAULT")

# Freeze all backbone initially
for param in model.features.parameters():
    param.requires_grad = False

# Replace classifier
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
    nn.Linear(128, num_classes),
)

model = model.to(device)

# Freeze BatchNorm initially
freeze_batchnorm(model)

# =========================
# LOSS & OPTIMIZER
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

# =========================
# TRAIN FUNCTION
# =========================


def train_model(optimizer):

    best_acc = 0.0

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        # =========================
        # Gradual unfreezing
        # Unfreeze last 4 layers at epoch 10, 10% learing rate
        # =========================

        if epoch == 10:

            print("\nUnfreezing last 4 EfficientNet blocks...")

            for param in model.features[-4:].parameters():
                param.requires_grad = True

            freeze_batchnorm(model)

            optimizer = optim.Adam(
                [
                    {"params": model.classifier.parameters(), "lr": LEARNING_RATE},
                    {
                        "params": model.features[-4:].parameters(),
                        "lr": LEARNING_RATE * 0.1,
                    },
                ]
            )

        # =========================

        for phase in ["train", "val"]:

            if phase == "train":
                model.train()
                freeze_batchnorm(model)
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
                torch.save(model, "best_model_full.pth")
                print("Saved best_model.pth")

    print("\nBest val accuracy:", best_acc)


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
train_model(optimizer)

test_model()
