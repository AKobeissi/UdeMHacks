# model_training.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Custom dataset for parasite images
class ParasiteDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = self.labels[idx]
        return image, label

# Data preprocessing transformations
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# Get data from database for continual learning
def get_training_data():
    conn = sqlite3.connect('parasite_diagnosis.db')
    
    # Get all verified samples
    df = pd.read_sql_query("""
    SELECT s.image_path, s.doctor_diagnosis
    FROM samples s
    WHERE s.doctor_verified = 1
    """, conn)
    
    conn.close()
    
    # Map class names to indices
    class_names = [
        "Plasmodium falciparum", 
        "Plasmodium vivax", 
        "Plasmodium malariae", 
        "Plasmodium ovale", 
        "Trypanosoma", 
        "Leishmania", 
        "Schistosoma", 
        "Filariasis", 
        "Entamoeba histolytica", 
        "No parasite detected"
    ]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    image_paths = df['image_path'].tolist()
    labels = [class_to_idx[diagnosis] for diagnosis in df['doctor_diagnosis']]
    
    return image_paths, labels, class_names

# Initialize and train the model
def train_model(image_paths, labels, class_names, epochs=10, batch_size=32, learning_rate=0.001):
    # Split data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    # Create datasets
    train_transform, val_transform = get_transforms()
    
    train_dataset = ParasiteDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = ParasiteDataset(val_paths, val_labels, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = models.resnet50(pretrained=True)
    num_classes = len(class_names)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Load existing model if available
    if os.path.exists('parasite_model.pth'):
        model.load_state_dict(torch.load('parasite_model.pth'))
        print("Loaded existing model for fine-tuning")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.1)
    
    # Training loop
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'parasite_model.pth')
            print(f'Model saved with accuracy: {val_acc:.4f}')
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    
    plt.savefig(f'training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    
    return model

# Evaluate model performance and generate confusion matrix
def evaluate_model(model, image_paths, labels, class_names):
    # Create dataset and dataloader
    _, val_transform = get_transforms()
    dataset = ParasiteDataset(image_paths, labels, transform=val_transform)
    dataloader = DataLoader(dataset, batch_size=32)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Evaluate
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    
    # Calculate accuracy
    accuracy = sum(1 for x, y in zip(all_preds, all_labels) if x == y) / len(all_labels)
    print(f'Model accuracy: {accuracy:.4f}')
    
    # Generate confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f'confusion_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    
    # Print classification report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)
    
    # Save report to file
    with open(f'classification_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt', 'w') as f:
        f.write(report)

# Main function to run the training process
def main():
    print("Starting model training process...")
    
    # Get data
    image_paths, labels, class_names = get_training_data()
    
    if len(image_paths) < 10:
        print(f"Not enough verified samples for training: {len(image_paths)} found. Need at least 10.")
        return
    
    print(f"Found {len(image_paths)} verified samples for training")
    
    # Train model
    model = train_model(image_paths, labels, class_names)
    
    # Evaluate model
    evaluate_model(model, image_paths, labels, class_names)
    
    print("Training complete!")

if __name__ == "__main__":
    main()