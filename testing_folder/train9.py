import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# =========================
# CONFIG
# =========================
print("Using EfficientNet-B2 with gradual unfreezing added smoothing unfreeze all layers")

DATA_DIR = "garbage_data"
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True


# =========================
# TRANSFORMS
# =========================

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


# =========================
# DATASETS
# =========================

image_datasets = {
    "train": datasets.ImageFolder(os.path.join(DATA_DIR, "train"), train_transform),
    "val": datasets.ImageFolder(os.path.join(DATA_DIR, "val"), val_transform)
}

dataloaders = {
    "train": DataLoader(image_datasets["train"], batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True),
    "val": DataLoader(image_datasets["val"], batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
}

class_names = image_datasets["train"].classes
num_classes = len(class_names)

print("Classes:", class_names)


# =========================
# MODEL
# =========================

model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

model = model.to(device)


# =========================
# FREEZE ALL FEATURES INITIALLY
# =========================

for param in model.features.parameters():
    param.requires_grad = False

for param in model.classifier.parameters():
    param.requires_grad = True


# =========================
# FREEZE BATCHNORM FUNCTION
# =========================

def freeze_batchnorm(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            module.weight.requires_grad = False
            module.bias.requires_grad = False


# =========================
# LOSS
# =========================

criterion = nn.CrossEntropyLoss()


# =========================
# TRAIN FUNCTION
# =========================

def train_model():

    best_acc = 0.0

    # initial optimizer (classifier only)
    optimizer = optim.Adam(
        model.classifier.parameters(),
        lr=LEARNING_RATE
    )


    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch+1}/{EPOCHS}")


        # =========================
        # GRADUAL UNFREEZING
        # =========================

        if epoch == 5:

            print("Stage 1: Unfreeze last 2 blocks")

            for param in model.features[-2:].parameters():
                param.requires_grad = True

            optimizer = optim.Adam([
                {"params": model.classifier.parameters(), "lr": LEARNING_RATE},
                {"params": model.features[-2:].parameters(), "lr": LEARNING_RATE * 0.1}
            ])


        elif epoch == 8:

            print("Stage 2: Unfreeze last 4 blocks")

            for param in model.features[-4:-2].parameters():
                param.requires_grad = True

            optimizer = optim.Adam([
                {"params": model.classifier.parameters(), "lr": LEARNING_RATE},
                {"params": model.features[-4:].parameters(), "lr": LEARNING_RATE * 0.05}
            ])


        elif epoch == 12:

            print("Stage 3: Unfreeze ALL blocks")

            for param in model.features[:-4].parameters():
                param.requires_grad = True

            optimizer = optim.Adam([
                {"params": model.classifier.parameters(), "lr": LEARNING_RATE},
                {"params": model.features.parameters(), "lr": LEARNING_RATE * 0.01}
            ])


        # =========================
        # TRAIN + VALIDATION LOOP
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

                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                        optimizer.step()


                running_loss += loss.item() * inputs.size(0)
                running_correct += (preds == labels).sum().item()


            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_correct / len(image_datasets[phase])

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")


            # =========================
            # SAVE BEST MODEL
            # =========================

            if phase == "val" and epoch_acc > best_acc:

                best_acc = epoch_acc

                torch.save({
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "accuracy": best_acc
                }, "best_model.pth")

                print("Saved best_model.pth")


    print("\nBest validation accuracy:", best_acc)



# =========================
# RUN
# =========================

train_model()

