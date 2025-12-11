import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from medmnist import INFO, PathMNIST


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("Num GPUs visible:", torch.cuda.device_count())


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

batch_size = 32   
num_workers = 4

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

print("Train size:", len(train_dataset))
print("Val size:", len(val_dataset))
print("Test size:", len(test_dataset))


#Model & Fine-tune Utilities
def create_resnet50(num_classes: int) -> nn.Module:
    """
    ResNet-50 with ImageNet pretrained weights, final FC replaced for PathMNIST.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def freeze_backbone(model: nn.Module):
    """
    Freeze all layers except the final fully-connected head (fc).
    """
    for name, param in model.named_parameters():
        if not name.startswith("fc"):
            param.requires_grad = False


def unfreeze_backbone(model: nn.Module):
    """
    Unfreeze all parameters for full fine-tuning.
    """
    for param in model.parameters():
        param.requires_grad = True


#Training & Evaluation
def train_one_epoch(model, loader, criterion, optimizer, epoch, total_epochs, phase="Train"):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        # MedMNIST labels: [B, 1] -> [B]
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

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    elapsed = time.time() - start_time

    print(
        f"[{phase}] Epoch {epoch}/{total_epochs} "
        f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | "
        f"Time: {elapsed:.1f}s"
    )
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, split="Val"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.squeeze().long().to(device, non_blocking=True)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"[{split}] Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
    return epoch_loss, epoch_acc


# Warm-up + Fine-tune + Multi-GPU
def main():
    base_lr = 0.0028         
    weight_decay = 1e-4
    momentum = 0.9
    warmup_epochs = 2
    finetune_epochs = 6       

    save_path = "resnet50_pipeline2_finetune_multigpu_best.pth"

    base_model = create_resnet50(num_classes)

    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(base_model)
    else:
        model = base_model

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    print("\n=== Warm-up phase: freeze backbone, train head only ===")
    freeze_backbone(base_model)  
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=base_lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    best_val_acc = 0.0

    for epoch in range(1, warmup_epochs + 1):
        train_one_epoch(model, train_loader, criterion, optimizer,
                        epoch, warmup_epochs, phase="Warmup")
        _, val_acc = evaluate(model, val_loader, criterion, split="Val (warmup)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            state_dict = base_model.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, save_path)
            print(f"*** Saved new best model (warmup) to {save_path}, Val Acc = {best_val_acc:.4f} ***")

    print("\n=== Fine-tuning phase: unfreeze backbone, full model training ===")
    unfreeze_backbone(base_model)

    optimizer = optim.SGD(
        model.parameters(),
        lr=base_lr,     
        momentum=momentum,
        weight_decay=weight_decay,
    )

    for epoch in range(1, finetune_epochs + 1):
        train_one_epoch(model, train_loader, criterion, optimizer,
                        epoch, finetune_epochs, phase="Finetune")
        _, val_acc = evaluate(model, val_loader, criterion, split="Val (finetune)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            state_dict = base_model.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, save_path)
            print(f"*** Saved new best model (finetune) to {save_path}, Val Acc = {best_val_acc:.4f} ***")

    print("\nBest Val Acc (overall) =", best_val_acc)

 
    if os.path.exists(save_path):
        print("\nLoading best model and evaluating on test set...")
        best_state = torch.load(save_path, map_location=device)
        base_model.load_state_dict(best_state)

        eval_model = base_model
        if torch.cuda.device_count() > 1:
            eval_model = nn.DataParallel(base_model)
        eval_model = eval_model.to(device)

        evaluate(eval_model, test_loader, criterion, split="Test")


if __name__ == "__main__":
    main()
