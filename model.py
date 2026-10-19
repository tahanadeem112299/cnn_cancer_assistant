# %% Cell 1: imports
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Experiment settings (inhe badal kar try karein)
UNFREEZE_BLOCKS = 3      # 5 karke bhi try karein
PHASE1_EPOCHS = 15
PHASE2_EPOCHS = 25
LABEL_SMOOTHING = 0.1

# %% Cell 2: data (Drive mount wala cell apna hi rakhein)
data_dir = '/content/drive/MyDrive/breast_cancer/dataset'

IMG = 224
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    norm,
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor(),
    norm,
])

train_base = datasets.ImageFolder(data_dir, transform=train_tf)
eval_base = datasets.ImageFolder(data_dir, transform=eval_tf)
print("Classes:", train_base.class_to_idx)

targets = np.array(train_base.targets)
idx = np.arange(len(targets))

# Stratified 70 / 15 / 15
train_idx, temp_idx = train_test_split(
    idx, test_size=0.30, stratify=targets, random_state=42)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.50, stratify=targets[temp_idx], random_state=42)

train_loader = DataLoader(Subset(train_base, train_idx), batch_size=32,
                          shuffle=True, num_workers=2)
val_loader = DataLoader(Subset(eval_base, val_idx), batch_size=32,
                        shuffle=False, num_workers=2)
test_loader = DataLoader(Subset(eval_base, test_idx), batch_size=32,
                         shuffle=False, num_workers=2)
print(len(train_idx), len(val_idx), len(test_idx))

counts = np.bincount(targets[train_idx], minlength=3)
weights = len(train_idx) / (3 * counts)
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
print("Class counts:", counts, "Weights:", weights)

# %% Cell 3: model + losses
class BreastUltrasoundModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.base_model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT)
        for p in self.base_model.parameters():
            p.requires_grad = False
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.base_model(x)

    def unfreeze_last_blocks(self, n=3):
        for p in self.base_model.features[-n:].parameters():
            p.requires_grad = True


model = BreastUltrasoundModel().to(device)

# Training loss: label smoothing ke saath
train_criterion = nn.CrossEntropyLoss(
    weight=class_weights, label_smoothing=LABEL_SMOOTHING)
# Validation loss: smoothing ke bagair (best model chunne ke liye saaf number)
eval_criterion = nn.CrossEntropyLoss(weight=class_weights)

# %% Cell 4: helper functions
def evaluate(loader, tta=False):
    """tta=True: image aur uski horizontal flip ke probabilities ka average."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    loss_sum = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += eval_criterion(out, y).item() * y.size(0)
            probs = torch.softmax(out, dim=1)
            if tta:
                out_flip = model(torch.flip(x, dims=[3]))
                probs = (probs + torch.softmax(out_flip, dim=1)) / 2
            all_probs.append(probs.cpu().numpy())
            all_preds.extend(probs.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    n = len(all_labels)
    acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    return (loss_sum / n, acc, np.array(all_labels),
            np.array(all_preds), np.concatenate(all_probs))


def run_phase(optimizer, epochs, name, scheduler=None):
    best_loss, best_acc, best_state = 1e9, 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        run_loss, correct, total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = train_criterion(out, y)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        if scheduler is not None:
            scheduler.step()
        v_loss, v_acc, *_ = evaluate(val_loader)
        lr_now = optimizer.param_groups[0]['lr']
        print(f"[{name}] Epoch {epoch}/{epochs} | LR {lr_now:.6f} | "
              f"Train loss {run_loss/len(train_loader):.4f} "
              f"acc {100*correct/total:.1f}% | "
              f"Val loss {v_loss:.4f} acc {v_acc:.1f}%")
        # Best model VAL LOSS se chuna jata hai
        if v_loss < best_loss:
            best_loss, best_acc = v_loss, v_acc
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    print(f"[{name}] Best val loss: {best_loss:.4f} (val acc {best_acc:.1f}%)")

# %% Cell 5: Phase 1 - sirf head train (base frozen)
opt1 = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad], lr=1e-3)
run_phase(opt1, epochs=PHASE1_EPOCHS, name="Phase1-head")

# %% Cell 6: Phase 2 - aakhri blocks unfreeze + cosine LR
model.unfreeze_last_blocks(UNFREEZE_BLOCKS)
opt2 = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad], lr=1e-4)
sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt2, T_max=PHASE2_EPOCHS, eta_min=1e-6)
run_phase(opt2, epochs=PHASE2_EPOCHS, name="Phase2-finetune", scheduler=sched2)

# %% Cell 7: final test evaluation (sirf aakhir mein, ek baar)
names = list(train_base.class_to_idx.keys())

for use_tta in (False, True):
    _, test_acc, y_true, y_pred, y_prob = evaluate(test_loader, tta=use_tta)
    print("=" * 60)
    print(f"TEST {'WITH TTA' if use_tta else 'NO TTA'} | accuracy: {test_acc:.2f}%")
    print(classification_report(y_true, y_pred, target_names=names, digits=3))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print("AUC (OvR):", roc_auc_score(y_true, y_prob, multi_class="ovr"))

#torch.save(model.state_dict(), "busi_efficientnet_b0_v2.pt")
print("Saved: busi_efficientnet_b0_v2.pt")