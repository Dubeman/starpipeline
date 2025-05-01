# Supervised Learning Model for Image Classification

from torchvision.models import mobilenet_v3_small
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights
import os
from torchvision import transforms, datasets, models, torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

def visualize_predictions(model, dataloader, class_names, num_images=8):
    model.eval()
    images_shown = 0

    fig = plt.figure(figsize=(15, 6))

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)

            for i in range(images.size(0)):
                if images_shown >= num_images:
                    break

                img = images[i].cpu()
                true_label = class_names[labels[i].item()]
                pred_label = class_names[preds[i].item()]
                correct = pred_label == true_label

                ax = fig.add_subplot(2, num_images // 2, images_shown + 1)
                img_np = img.permute(1, 2, 0) * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
                img_np = img_np.clamp(0, 1).numpy()

                ax.imshow(img_np)
                ax.axis('off')
                ax.set_title(f"Pred: {pred_label}\nTrue: {true_label}", color='green' if correct else 'red')

                images_shown += 1

            if images_shown >= num_images:
                break

    plt.tight_layout()
    plt.show()

def get_star_image_loaders(data_dir, batch_size=32, val_split=0.2, test_split=0.1, num_workers=4):
    # Transform to match image to network (grayscale to RGB, resize, normalize)
    # Note: MobileNetV3 expects 3 channels
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                             std=[0.229, 0.224, 0.225])
    ])

    # Load full dataset (expects data_dir/class_name/image)
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    class_names = full_dataset.classes
    print(f"Found classes: {class_names}")

    # Split into train/val/test
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    test_size = int(total_size * test_split)
    train_size = total_size - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size],
                                                             generator=torch.Generator().manual_seed(42))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, class_names

def build_model():
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

    # 3 Output Classes - Solvable, Noisy, Unsolvable
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 3)

    for param in model.features.parameters():
        param.requires_grad = False

    return model

def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    return total_loss / len(dataloader), acc

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    return total_loss / len(dataloader), acc

def train_model(model, train_loader, val_loader, opt, crit, epochs=10):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss, train_acc = train(model, train_loader, opt, crit)
        val_loss, val_acc = evaluate(model, val_loader, crit)

        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        torch.save(model.state_dict(), "model.pth")

def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    return total_loss / len(test_loader), acc

data_dir = "/Users/owen/starpipeline/classifer/data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    train_loader, val_loader, test_loader, class_names = get_star_image_loaders(data_dir)    
    model = build_model().to(device)
    criterion = torch.nn.CrossEntropyLoss()

    if os.path.exists("model.pth"):
        model.load_state_dict(torch.load("model.pth"))
        print("Model loaded from checkpoint.")
    else:
        print("No checkpoint found, starting training from scratch.")

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        train_model(model, train_loader, val_loader, optimizer, criterion, epochs=10)
    
    test_loss, test_acc = evaluate_model(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    visualize_predictions(model, test_loader, class_names, num_images=10)
    

if __name__ == "__main__":
    main()