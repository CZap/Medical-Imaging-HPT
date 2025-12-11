import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from medmnist import INFO, PathMNIST

#Device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("Num GPUs visible:", torch.cuda.device_count())

#Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

info = INFO["pathmnist"]
DataClass = PathMNIST
num_classes = len(info["label"])
print("Num classes:", num_classes)

train_dataset = DataClass(split="train", transform=transform, download=True)
val_dataset   = DataClass(split="val",   transform=transform, download=True)
test_dataset  = DataClass(split="test",  transform=transform, download=True)

batch_size = 64
num_workers = 4

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)

print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset))
print("Test size:", len(test_dataset))


#Model 
def create_resnet50(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


base_model = create_resnet50(num_classes)


if torch.cuda.device_count() > 1:
    print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(base_model)
else:
    model = base_model

model = model.to(device)

criterion = nn.CrossEntropyLoss()

# Baseline: Adam, lr=1e-3, batch=64, single-stage
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)



def train_one_epoch(model, loader, criterion, optimizer, epoch, total_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    start = time.time()

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.squeeze().long().to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    loss = running_loss / total
    acc = correct / total
    print(f"[Train] Epoch {epoch}/{total_epochs} Loss: {loss:.4f} | Acc: {acc:.4f} | Time: {time.time()-start:.1f}s")
    return loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, split="Val"):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.squeeze().long().to(device, non_blocking=True)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    loss = running_loss / total
    acc = correct / total
    print(f"[{split}] Loss: {loss:.4f} | Acc: {acc:.4f}")
    return loss, acc


def main():
    epochs = 8
    best_val = 0.0
    save_path = "resnet50_pipeline2_baseline_multigpu_best.pth"

    for epoch in range(1, epochs + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, epoch, epochs)
        _, val_acc = evaluate(model, val_loader, criterion, split="Val")

        if val_acc > best_val:
            best_val = val_acc
            state_dict = base_model.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, save_path)
            print(f"*** Saved new best model to {save_path}, Val Acc = {best_val:.4f} ***")

    print("\nBest Val Acc (baseline) =", best_val)

    if os.path.exists(save_path):
        print("\nLoading best baseline model and evaluating on test set...")
        best_state = torch.load(save_path, map_location=device)
        base_model.load_state_dict(best_state)

        eval_model = base_model
        if torch.cuda.device_count() > 1:
            eval_model = nn.DataParallel(base_model)
        eval_model = eval_model.to(device)

        _, test_acc = evaluate(eval_model, test_loader, criterion, split="Test")
        print("Final Test Acc (baseline) =", test_acc)


if __name__ == "__main__":
    main()
