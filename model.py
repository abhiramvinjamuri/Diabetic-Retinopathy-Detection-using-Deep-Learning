import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import logging

# Set up logging to a file
logging.basicConfig(filename='training_output.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Print to confirm the script is running
logging.info("Model script is running...")
print("Model script is running...")  # Display in terminal

# Dataset class for training and testing
class DiabeticRetinopathyDataset(Dataset):
    def __init__(self, annotations_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.annotations = pd.read_excel(annotations_file)
        self.label_mapping = {
            "No DR signs": 0,
            "Mild (or early) NPDR": 1,
            "Moderate NPDR": 2,
            "Severe NPDR": 3,
            "Very Severe NPDR": 4,
            "PDR": 5,
            "Advanced PDR": 6
        }
        self.annotations['Label'] = self.annotations['Status'].map(self.label_mapping)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        file_name = f"{row['Image']}.{row['Format']}"
        folder_name = row['Status']
        image_path = os.path.join(self.root_dir, folder_name, file_name)

        image = Image.open(image_path).convert("RGB")
        label = row['Label']

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.annotations)

# Define transformations for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Hybrid model definition combining ResNet-152 and DenseNet-121
class HybridResNetDenseNetModel(nn.Module):
    def __init__(self, num_classes=7):
        super(HybridResNetDenseNetModel, self).__init__()
        # Load pre-trained ResNet-152
        self.resnet152 = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
        num_ftrs_resnet = self.resnet152.fc.in_features
        self.resnet152.fc = nn.Identity()  # Remove the final fully connected layer

        # Load pre-trained DenseNet-121
        self.densenet121 = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        num_ftrs_densenet = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Identity()  # Remove the final classifier layer

        # Combine the features from both networks
        self.fc = nn.Linear(num_ftrs_resnet + num_ftrs_densenet, num_classes)

    def forward(self, x):
        resnet_features = self.resnet152(x)
        densenet_features = self.densenet121(x)
        combined_features = torch.cat((resnet_features, densenet_features), dim=1)
        out = self.fc(combined_features)
        return out

# Metrics and confusion matrix function
def evaluate_model(loader):
    model.eval()
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(preds.cpu().numpy())

    # Calculate metrics
    report = classification_report(true_labels, predicted_labels, target_names=train_dataset.label_mapping.keys(), output_dict=True)
    logging.info(f"Classification Report:\n{classification_report(true_labels, predicted_labels, target_names=train_dataset.label_mapping.keys())}")
    print(f"Classification Report:\n{classification_report(true_labels, predicted_labels, target_names=train_dataset.label_mapping.keys())}")

    # Per-class accuracy
    class_accuracies = {label: report[label]['recall'] for label in train_dataset.label_mapping.keys()}
    logging.info(f"Per-Class Accuracy:\n{class_accuracies}")
    print(f"Per-Class Accuracy:\n{class_accuracies}")

    # Confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.label_mapping.keys())
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    return report

# Updated training function to include evaluation
def train_model(num_epochs=15):
    if len(train_loader) == 0:
        logging.warning("Warning: Training DataLoader is empty. Ensure dataset is correctly loaded.")
        print("Warning: Training DataLoader is empty. Ensure dataset is correctly loaded.")  # Display in terminal
        return

    train_losses = []
    train_accuracies = []

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        # Log epoch info
        logging.info(f"Epoch [{epoch + 1}/{num_epochs}] - Training started")
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Training started")  # Display in terminal

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Accuracy calculation
            _, predicted = torch.max(outputs, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            # Log progress for each batch
            batch_accuracy = 100 * correct_predictions / total_predictions
            logging.info(f"Batch [{i + 1}/{len(train_loader)}] - Loss: {loss.item():.4f}, Accuracy: {batch_accuracy:.2f}%")
            print(f"Batch [{i + 1}/{len(train_loader)}] - Loss: {loss.item():.4f}, Accuracy: {batch_accuracy:.2f}%")  # Display in terminal

        # Update the learning rate
        scheduler.step()

        epoch_accuracy = (correct_predictions / total_predictions) * 100
        train_losses.append(epoch_loss / len(train_loader))
        train_accuracies.append(epoch_accuracy)

        # Epoch-level progress display
        logging.info(f"Epoch [{epoch + 1}/{num_epochs}] completed. Loss: {epoch_loss / len(train_loader):.4f}, Accuracy: {epoch_accuracy:.2f}%")
        print(f"Epoch [{epoch + 1}/{num_epochs}] completed. Loss: {epoch_loss / len(train_loader):.4f}, Accuracy: {epoch_accuracy:.2f}%")  # Display in terminal

        # Save model checkpoint after each epoch
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss
        }
        torch.save(checkpoint, f"diabetic_retinopathy_model_epoch_{epoch + 1}.pth")

    # Plotting Loss and Accuracy
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label="Training Accuracy", color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy')

    plt.tight_layout()
    plt.show()

    # Evaluate on training set (or a validation/test set if available)
    evaluate_model(train_loader)

if __name__ == "__main__":
    # Initialize model, loss, and optimizer
    model = HybridResNetDenseNetModel(num_classes=7)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Dataset and DataLoader paths
    annotations_file = "D:\\DATASET\\Annotations of the classifications.xlsx"
    root_dir = "D:\\DATASET"

    train_dataset = DiabeticRetinopathyDataset(annotations_file=annotations_file, root_dir=root_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Check for GPU availability and set up multi-GPU if applicable
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Using DataParallel to utilize multiple GPUs if available
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")  # Display in terminal
        model = nn.DataParallel(model)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Start training
    train_model(num_epochs=15)

    logging.info("Model training completed and saved.")
    print("Model training completed and saved.")  # Display in terminal
