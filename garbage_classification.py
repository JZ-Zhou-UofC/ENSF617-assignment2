#! /usr/bin/env python
# -------------------------------------------------------

"""
Complete Multimodal Garbage Classification System
Combines DistilBERT (text) and ResNet18 (image) models for robust classification
Based on the provided notebook implementations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ExponentialLR
from transformers import DistilBertModel, DistilBertTokenizer
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import re
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')
#from google.colab import drive
#drive.mount('/content/drive')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Update these paths to your dataset location
#TRAIN_PATH = "/content/drive/MyDrive/CVPR_2024_dataset_Train"
#VAL_PATH = "/content/drive/MyDrive/CVPR_2024_dataset_Val"
#TEST_PATH = "/content/drive/MyDrive/CVPR_2024_dataset_Test"

data_dir = "/work/TALC/ensf617_2026w/garbage_data"
TRAIN_PATH = os.path.join(data_dir, "CVPR_2024_dataset_Train")
VAL_PATH   = os.path.join(data_dir, "CVPR_2024_dataset_Val")
TEST_PATH  = os.path.join(data_dir, "CVPR_2024_dataset_Test")

# Model save paths
TEXT_MODEL_PATH = 'best_text_model.pth'
IMAGE_MODEL_PATH = 'best_image_model.pth'

# Hyperparameters
TEXT_MAX_LEN = 24
TEXT_BATCH_SIZE = 8
TEXT_EPOCHS = 4
TEXT_LR = 2e-5

IMAGE_BATCH_SIZE = 32
IMAGE_EPOCHS = 20
IMAGE_LR = 0.001

NUM_CLASSES = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")

# ============================================================================
# PART 1: TEXT MODEL COMPONENTS
# ============================================================================

def read_text_files_with_labels(path):
    """Extract text from filenames and labels from folder structure"""
    texts = []
    labels = []
    class_folders = sorted([
         d for d in os.listdir(path)
         if os.path.isdir(os.path.join(path, d))
    ])
    label_map = {class_name: idx for idx, class_name in enumerate(class_folders)}

    for class_name in class_folders:
        class_path = os.path.join(path, class_name)
        if os.path.isdir(class_path):
            file_names = sorted(os.listdir(class_path))
            for file_name in file_names:
                file_path = os.path.join(class_path, file_name)
                if os.path.isfile(file_path):
                    file_name_no_ext, _ = os.path.splitext(file_name)
                    text = file_name_no_ext.replace('_', ' ')
                    text_without_digits = re.sub(r'\d+', '', text)
                    texts.append(text_without_digits)
                    labels.append(label_map[class_name])

    return np.array(texts), np.array(labels), label_map


class TextDataset(Dataset):
    """Dataset for text classification"""
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class DistilBERTClassifier(nn.Module):
    """Text classifier using DistilBERT"""
    def __init__(self, num_classes):
        super(DistilBERTClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(self.distilbert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        pooled_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)[0]
        output = self.drop(pooled_output[:, 0])
        return self.out(output)


def train_text_epoch(model, iterator, optimizer, criterion, device):
    """Train text model for one epoch"""
    model.train()
    total_loss = 0
    for batch in iterator:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        output = model(input_ids, attention_mask)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(iterator)


def evaluate_text_model(model, iterator, criterion, device):
    """Evaluate text model"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in iterator:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            output = model(input_ids, attention_mask)
            loss = criterion(output, labels)
            total_loss += loss.item()

    return total_loss / len(iterator)


def predict_text(model, dataloader, device):
    """Get predictions from text model"""
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
    return predictions


# ============================================================================
# PART 2: IMAGE MODEL COMPONENTS
# ============================================================================

class GarbageImageModel(nn.Module):
    """Image classifier using ResNet18 with transfer learning"""
    def __init__(self, num_classes, transfer=True):
        super().__init__()

        # Load pretrained or random
        if transfer:
            self.feature_extractor = models.resnet18(weights='IMAGENET1K_V1')
        else:
            self.feature_extractor = models.resnet18(weights=None)

        # Get feature size
        num_features = self.feature_extractor.fc.in_features

        # Remove original classifier
        self.feature_extractor.fc = nn.Identity()

        # Freeze backbone if transfer learning
        if transfer:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # New classifier
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


def train_image_model(model, trainloader, valloader, criterion, optimizer, 
                      scheduler, device, epochs, save_path):
    """Train image model"""
    best_loss = 1e+20
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / (i + 1)
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.3f}', end=' ')
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, data in enumerate(valloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / (i + 1)
        print(f'Val Loss: {avg_val_loss:.3f}')
        
        # Save best model
        if val_loss < best_loss:
            print("  -> Saving best model")
            torch.save(model.state_dict(), save_path)
            best_loss = val_loss
    
    print('Finished Training Image Model')
    return model


def predict_image(model, dataloader, device):
    """Get predictions from image model"""
    model.eval()
    predictions = []
    with torch.no_grad():
        for data in dataloader:
            images, _ = data
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
    return predictions


# ============================================================================
# PART 3: MULTIMODAL COMPONENTS
# ============================================================================

class MultimodalDataset(Dataset):
    """Dataset that provides both image and text for each sample"""
    def __init__(self, image_folder_dataset, texts, tokenizer, max_len):
        self.image_dataset = image_folder_dataset
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        image, label = self.image_dataset[idx]
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'image': image,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long),
            'text': text
        }


class MultimodalGarbageClassifier:
    """Multimodal classifier combining text and image models"""
    def __init__(self, text_model, image_model, tokenizer, transform, 
                 device, class_names):
        self.text_model = text_model
        self.image_model = image_model
        self.tokenizer = tokenizer
        self.transform = transform
        self.device = device
        self.class_names = class_names
        
        self.text_model.eval()
        self.image_model.eval()
    
    def classify_from_path(self, image_path, text, alpha=0.5, return_details=False):
        """
        Classify a single image with text description
        
        Args:
            image_path: Path to image file
            text: Text description (e.g., from filename)
            alpha: Weight for text model (1-alpha for image model)
            return_details: If True, return detailed probabilities
        
        Returns:
            predicted_class, confidence, (optional) text_probs, image_probs
        """
        # Text prediction
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=TEXT_MAX_LEN,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            text_logits = self.text_model(input_ids, attention_mask)
            text_probs = F.softmax(text_logits, dim=1)
        
        # Image prediction
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_logits = self.image_model(image_tensor)
            image_probs = F.softmax(image_logits, dim=1)
        
        # Combine predictions
        combined_probs = alpha * text_probs + (1 - alpha) * image_probs
        predicted_class = torch.argmax(combined_probs, dim=1).item()
        confidence = combined_probs[0][predicted_class].item()
        
        if return_details:
            return predicted_class, confidence, text_probs[0].cpu().numpy(), image_probs[0].cpu().numpy()
        return predicted_class, confidence
    
    def classify_batch(self, dataloader, alpha=0.5):
        """
        Classify a batch of samples from MultimodalDataset
        
        Args:
            dataloader: DataLoader with MultimodalDataset
            alpha: Weight for text model
            
        Returns:
            predictions, labels, text_probs, image_probs
        """
        all_predictions = []
        all_labels = []
        all_text_probs = []
        all_image_probs = []
        
        self.text_model.eval()
        self.image_model.eval()
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Text predictions
                text_logits = self.text_model(input_ids, attention_mask)
                text_probs = F.softmax(text_logits, dim=1)
                
                # Image predictions
                image_logits = self.image_model(images)
                image_probs = F.softmax(image_logits, dim=1)
                
                # Combine
                combined_probs = alpha * text_probs + (1 - alpha) * image_probs
                predictions = torch.argmax(combined_probs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_text_probs.extend(text_probs.cpu().numpy())
                all_image_probs.extend(image_probs.cpu().numpy())
        
        return (np.array(all_predictions), 
                np.array(all_labels),
                np.array(all_text_probs),
                np.array(all_image_probs))
    
    def optimize_alpha(self, val_dataloader, alphas=None):
        """
        Find optimal alpha weight on validation set
        
        Args:
            val_dataloader: Validation MultimodalDataset dataloader
            alphas: List of alpha values to try
            
        Returns:
            best_alpha, best_accuracy
        """
        if alphas is None:
            alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        print("\nOptimizing alpha weight...")
        print("Alpha | Accuracy")
        print("------|----------")
        
        best_alpha = 0.5
        best_acc = 0
        
        for alpha in alphas:
            predictions, labels, _, _ = self.classify_batch(val_dataloader, alpha=alpha)
            acc = accuracy_score(labels, predictions)
            print(f"{alpha:.1f}   | {acc:.4f}")
            
            if acc > best_acc:
                best_acc = acc
                best_alpha = alpha
        
        print(f"\nBest alpha: {best_alpha} with accuracy: {best_acc:.4f}")
        return best_alpha, best_acc


# ============================================================================
# PART 4: EVALUATION AND VISUALIZATION
# ============================================================================

def plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    plt.savefig(title.replace(' ', '_').lower() + '.png', dpi=150, bbox_inches='tight')
    plt.show()


def evaluate_models(text_model, image_model, multimodal_classifier,
                   test_text_loader, test_image_loader, test_multimodal_loader,
                   labels_test, class_names, alpha=0.5):
    """
    Comprehensive evaluation of all three models
    """
    print("\n" + "="*70)
    print("MODEL EVALUATION ON TEST SET")
    print("="*70)
    
    # Text model evaluation
    print("\n1. TEXT MODEL (DistilBERT)")
    print("-" * 70)
    text_predictions = predict_text(text_model, test_text_loader, DEVICE)
    text_acc = accuracy_score(labels_test, text_predictions)
    print(f"Accuracy: {text_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels_test, text_predictions, 
                                target_names=class_names))
    plot_confusion_matrix(labels_test, text_predictions, class_names, 
                         'Text Model Confusion Matrix')
    
    # Image model evaluation
    print("\n2. IMAGE MODEL (ResNet18)")
    print("-" * 70)
    image_predictions = predict_image(image_model, test_image_loader, DEVICE)
    image_acc = accuracy_score(labels_test, image_predictions)
    print(f"Accuracy: {image_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels_test, image_predictions, 
                                target_names=class_names))
    plot_confusion_matrix(labels_test, image_predictions, class_names,
                         'Image Model Confusion Matrix')
    
    # Multimodal evaluation
    print(f"\n3. MULTIMODAL MODEL (Combined, alpha={alpha})")
    print("-" * 70)
    multi_predictions, multi_labels, text_probs, image_probs = \
        multimodal_classifier.classify_batch(test_multimodal_loader, alpha=alpha)
    multi_acc = accuracy_score(multi_labels, multi_predictions)
    print(f"Accuracy: {multi_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(multi_labels, multi_predictions, 
                                target_names=class_names))
    plot_confusion_matrix(multi_labels, multi_predictions, class_names,
                         'Multimodal Model Confusion Matrix')
    
    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    print(f"Text Model:       {text_acc:.4f}")
    print(f"Image Model:      {image_acc:.4f}")
    print(f"Multimodal Model: {multi_acc:.4f}")
    print(f"Improvement:      {multi_acc - max(text_acc, image_acc):.4f}")
    print("="*70)
    
    return {
        'text_acc': text_acc,
        'image_acc': image_acc,
        'multimodal_acc': multi_acc,
        'text_predictions': text_predictions,
        'image_predictions': image_predictions,
        'multimodal_predictions': multi_predictions
    }


# ============================================================================
# PART 5: MAIN TRAINING AND EVALUATION PIPELINE
# ============================================================================

def main():
    """Main training and evaluation pipeline"""
    
    print("="*70)
    print("MULTIMODAL GARBAGE CLASSIFICATION SYSTEM")
    print("="*70)
    
    # ========================================================================
    # STEP 1: LOAD AND PREPARE TEXT DATA
    # ========================================================================
    print("\n[STEP 1] Loading text data from filenames...")
    text_train, labels_train_text, label_map = read_text_files_with_labels(TRAIN_PATH)
    text_val, labels_val_text, _ = read_text_files_with_labels(VAL_PATH)
    text_test, labels_test_text, _ = read_text_files_with_labels(TEST_PATH)
    
    class_names = sorted(label_map.keys())
    print(f"Classes: {class_names}")
    print(f"Train samples: {len(text_train)}")
    print(f"Val samples: {len(text_val)}")
    print(f"Test samples: {len(text_test)}")
    
    # ========================================================================
    # STEP 2: TRAIN TEXT MODEL
    # ========================================================================
    print("\n[STEP 2] Training text model (DistilBERT)...")
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    train_text_dataset = TextDataset(text_train, labels_train_text, tokenizer, TEXT_MAX_LEN)
    val_text_dataset = TextDataset(text_val, labels_val_text, tokenizer, TEXT_MAX_LEN)
    test_text_dataset = TextDataset(text_test, labels_test_text, tokenizer, TEXT_MAX_LEN)
    
    train_text_loader = DataLoader(train_text_dataset, batch_size=TEXT_BATCH_SIZE, shuffle=True)
    val_text_loader = DataLoader(val_text_dataset, batch_size=TEXT_BATCH_SIZE, shuffle=False)
    test_text_loader = DataLoader(test_text_dataset, batch_size=TEXT_BATCH_SIZE, shuffle=False)
    
    text_model = DistilBERTClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    text_optimizer = optim.Adam(text_model.parameters(), lr=TEXT_LR)
    text_criterion = nn.CrossEntropyLoss()
    
    best_text_loss = 1e+10
    for epoch in range(TEXT_EPOCHS):
        train_loss = train_text_epoch(text_model, train_text_loader, 
                                      text_optimizer, text_criterion, DEVICE)
        val_loss = evaluate_text_model(text_model, val_text_loader, 
                                       text_criterion, DEVICE)
        print(f'Epoch {epoch+1}/{TEXT_EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_text_loss:
            best_text_loss = val_loss
            torch.save(text_model.state_dict(), TEXT_MODEL_PATH)
            print("  -> Saved best text model")
    
    # Load best text model
    text_model.load_state_dict(torch.load(TEXT_MODEL_PATH, weights_only=True))
    print(f"\nText model training complete. Best val loss: {best_text_loss:.4f}")
    
    # ========================================================================
    # STEP 3: TRAIN IMAGE MODEL
    # ========================================================================
    print("\n[STEP 3] Training image model (ResNet18)...")
    
    # Image transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load image datasets
    train_image_dataset = ImageFolder(root=TRAIN_PATH, transform=train_transform)
    val_image_dataset = ImageFolder(root=VAL_PATH, transform=train_transform)
    test_image_dataset = ImageFolder(root=TEST_PATH, transform=test_transform)
    
    train_image_loader = DataLoader(train_image_dataset, batch_size=IMAGE_BATCH_SIZE, 
                                    shuffle=True, num_workers=2)
    val_image_loader = DataLoader(val_image_dataset, batch_size=IMAGE_BATCH_SIZE, 
                                  shuffle=False, num_workers=2)
    test_image_loader = DataLoader(test_image_dataset, batch_size=IMAGE_BATCH_SIZE, 
                                   shuffle=False, num_workers=2)
    
    print(f"Image dataset sizes - Train: {len(train_image_dataset)}, "
          f"Val: {len(val_image_dataset)}, Test: {len(test_image_dataset)}")
    
    # Create and train image model
    image_model = GarbageImageModel(NUM_CLASSES, transfer=True).to(DEVICE)
    image_criterion = nn.CrossEntropyLoss()
    image_optimizer = torch.optim.AdamW(image_model.parameters(), lr=IMAGE_LR)
    image_scheduler = ExponentialLR(image_optimizer, gamma=0.9)
    
    image_model = train_image_model(image_model, train_image_loader, val_image_loader,
                                    image_criterion, image_optimizer, image_scheduler,
                                    DEVICE, IMAGE_EPOCHS, IMAGE_MODEL_PATH)
    
    # Load best image model
    image_model.load_state_dict(torch.load(IMAGE_MODEL_PATH, weights_only=True))
    print("\nImage model training complete.")
    
    # ========================================================================
    # STEP 4: CREATE MULTIMODAL CLASSIFIER
    # ========================================================================
    print("\n[STEP 4] Creating multimodal classifier...")
    
    # Create multimodal test dataset
    test_multimodal_dataset = MultimodalDataset(test_image_dataset, text_test, 
                                                tokenizer, TEXT_MAX_LEN)
    test_multimodal_loader = DataLoader(test_multimodal_dataset, 
                                       batch_size=IMAGE_BATCH_SIZE, 
                                       shuffle=False, num_workers=2)
    
    # Create multimodal validation dataset for alpha optimization
    val_multimodal_dataset = MultimodalDataset(val_image_dataset, text_val, 
                                              tokenizer, TEXT_MAX_LEN)
    val_multimodal_loader = DataLoader(val_multimodal_dataset, 
                                      batch_size=IMAGE_BATCH_SIZE, 
                                      shuffle=False, num_workers=2)
    
    multimodal_classifier = MultimodalGarbageClassifier(
        text_model, image_model, tokenizer, test_transform, DEVICE, class_names
    )
    
    # ========================================================================
    # STEP 5: OPTIMIZE ALPHA WEIGHT
    # ========================================================================
    print("\n[STEP 5] Optimizing alpha weight on validation set...")
    best_alpha, best_val_acc = multimodal_classifier.optimize_alpha(val_multimodal_loader)
    
    # ========================================================================
    # STEP 6: EVALUATE ALL MODELS
    # ========================================================================
    print("\n[STEP 6] Evaluating all models on test set...")
    results = evaluate_models(
        text_model, image_model, multimodal_classifier,
        test_text_loader, test_image_loader, test_multimodal_loader,
        labels_test_text, class_names, alpha=best_alpha
    )
    
    # ========================================================================
    # STEP 7: DEMO - CLASSIFY SINGLE IMAGE
    # ========================================================================
    print("\n[STEP 7] Demo - Single image classification")
    print("-" * 70)
    
    # Get a sample from test set
    sample_idx = 0
    sample_path = test_image_dataset.samples[sample_idx][0]
    sample_text = text_test[sample_idx]
    sample_true_label = test_image_dataset.samples[sample_idx][1]
    
    print(f"\nSample Image Path: {sample_path}")
    print(f"Sample Text: '{sample_text}'")
    print(f"True Label: {class_names[sample_true_label]}")
    
    pred_class, confidence, text_probs, image_probs = \
        multimodal_classifier.classify_from_path(sample_path, sample_text, 
                                                alpha=best_alpha, return_details=True)
    
    print(f"\nPredicted: {class_names[pred_class]} (confidence: {confidence:.3f})")
    print("\nDetailed Probabilities:")
    print("Class        | Text Model | Image Model | Combined")
    print("-" * 60)
    for i, class_name in enumerate(class_names):
        combined_prob = best_alpha * text_probs[i] + (1 - best_alpha) * image_probs[i]
        print(f"{class_name:12} | {text_probs[i]:10.3f} | {image_probs[i]:11.3f} | {combined_prob:8.3f}")
    
    # ========================================================================
    # STEP 8: SAVE RESULTS
    # ========================================================================
    print("\n[STEP 8] Saving results...")
    
    results_summary = {
        'best_alpha': best_alpha,
        'text_accuracy': results['text_acc'],
        'image_accuracy': results['image_acc'],
        'multimodal_accuracy': results['multimodal_acc'],
        'class_names': class_names
    }
    
    torch.save(results_summary, 'multimodal_results.pth')
    print("Results saved to 'multimodal_results.pth'")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nFinal Results:")
    print(f"  Text Model Accuracy:       {results['text_acc']:.4f}")
    print(f"  Image Model Accuracy:      {results['image_acc']:.4f}")
    print(f"  Multimodal Model Accuracy: {results['multimodal_acc']:.4f}")
    print(f"  Optimal Alpha:             {best_alpha:.2f}")
    print(f"\nModels saved:")
    print(f"  - {TEXT_MODEL_PATH}")
    print(f"  - {IMAGE_MODEL_PATH}")
    print("="*70)


# ============================================================================
# BONUS: INFERENCE FUNCTION FOR NEW IMAGES
# ============================================================================

def classify_new_image(image_path, text_description="", 
                      text_model_path=TEXT_MODEL_PATH,
                      image_model_path=IMAGE_MODEL_PATH,
                      alpha=0.5):
    """
    Classify a new garbage image with optional text description
    
    Args:
        image_path: Path to the image
        text_description: Optional text description
        text_model_path: Path to trained text model
        image_model_path: Path to trained image model
        alpha: Weight for text vs image (default 0.5)
    
    Returns:
        predicted_class_name, confidence
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_names = ['Black', 'Blue', 'Green', 'TTR']
    
    # Load models
    text_model = DistilBERTClassifier(num_classes=4).to(device)
    text_model.load_state_dict(torch.load(text_model_path, weights_only=True))
    
    image_model = GarbageImageModel(4, transfer=True).to(device)
    image_model.load_state_dict(torch.load(image_model_path, weights_only=True))
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    classifier = MultimodalGarbageClassifier(
        text_model, image_model, tokenizer, test_transform, device, class_names
    )
    
    pred_class, confidence = classifier.classify_from_path(
        image_path, text_description, alpha=alpha
    )
    
    return class_names[pred_class], confidence


# ============================================================================
# RUN THE PIPELINE
# ============================================================================

if __name__ == "__main__":
    main()
    
    # Example of using the classifier on a new image:
    # predicted_class, confidence = classify_new_image(
    #     "path/to/new/image.jpg", 
    #     "plastic bottle", 
    #     alpha=0.5
    # )
    # print(f"Predicted: {predicted_class} (confidence: {confidence:.3f})")
