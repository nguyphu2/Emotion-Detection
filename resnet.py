import kagglehub
import torch
import wandb
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import os

# ============================================================
#  CONFIGURATION FLAGS  — toggle these to change the run
# ============================================================

# Backbone: 'resnet18' or 'resnet50'
BACKBONE = 'resnet18'

# Classifier head: simple linear layer or deeper MLP head
MODIFIED_HEAD = True  # True → Linear→ReLU→Dropout→Linear ; False → single Linear

# Layer freezing:
#   'all_frozen'   → only the new classifier head trains
#   'partial'      → last ResNet block (layer4) + head train; rest frozen
#   'none'         → entire network fine-tunes (no freezing)
FREEZE_MODE = 'all_frozen'   # 'all_frozen' | 'partial' | 'none'

# Two-phase training (only used when FREEZE_MODE == 'all_frozen'):
#   Phase 1 trains only the head; Phase 2 unfreezes layer4 and fine-tunes further.
TWO_PHASE = True  # ignored when FREEZE_MODE != 'all_frozen'

EPOCHS_PHASE1 = 20
EPOCHS_PHASE2 = 10   # only relevant when TWO_PHASE=True and FREEZE_MODE='all_frozen'

BATCH_SIZE   = 32
LR_PHASE1    = 1e-4
LR_PHASE2    = 1e-5

# Attention-map logging frequency (every N epochs, always on the last epoch too)
LOG_ATTN_EVERY = 5
ATTN_IMAGES    = 8    # images per log call

# ============================================================


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Backbone: {BACKBONE}  |  Modified head: {MODIFIED_HEAD}  |  Freeze mode: {FREEZE_MODE}")


# ---- Dataset ----------------------------------------------------------------

path = kagglehub.dataset_download("danielshanbalico/dog-emotion")
root = os.path.join(path, "Dog Emotion")

base_dataset  = datasets.ImageFolder(root=root)
data_size     = len(base_dataset)
class_names   = base_dataset.classes   # ['angry', 'happy', 'relaxed', 'sad']
print(f"Classes: {class_names}  |  Total images: {data_size}")

# ---- Transforms -------------------------------------------------------------

training_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---- Split ------------------------------------------------------------------

train_size = int(0.70 * data_size)
val_size   = int(0.15 * data_size)
test_size  = data_size - train_size - val_size

train_idx, val_idx, test_idx = random_split(range(data_size), [train_size, val_size, test_size])

train_dataset = torch.utils.data.Subset(
    datasets.ImageFolder(root=root, transform=training_transform), train_idx.indices)
val_dataset   = torch.utils.data.Subset(
    datasets.ImageFolder(root=root, transform=eval_transform),     val_idx.indices)
test_dataset  = torch.utils.data.Subset(
    datasets.ImageFolder(root=root, transform=eval_transform),     test_idx.indices)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# ============================================================
#  MODEL BUILDER
# ============================================================

def build_model(backbone: str, modified_head: bool, freeze_mode: str):
    """
    Build a ResNet model with configurable backbone, head, and freezing strategy.

    Parameters
    ----------
    backbone      : 'resnet18' or 'resnet50'
    modified_head : whether to use a deeper MLP classifier head
    freeze_mode   : 'all_frozen' | 'partial' | 'none'
                    - 'all_frozen' → all pretrained weights frozen; only head trained
                    - 'partial'    → layer4 + head unfrozen; earlier layers frozen
                    - 'none'       → full network unfrozen from the start

    Returns
    -------
    model, optimizer
    """
    if backbone == 'resnet18':
        model      = models.resnet18(pretrained=True)
        fc_in_feat = model.fc.in_features   # 512
    elif backbone == 'resnet50':
        model      = models.resnet50(pretrained=True)
        fc_in_feat = model.fc.in_features   # 2048
    else:
        raise ValueError(f"Unknown backbone: {backbone}. Choose 'resnet18' or 'resnet50'.")

    # ---------- Apply freezing ----------
    if freeze_mode == 'all_frozen':
        for param in model.parameters():
            param.requires_grad = False

    elif freeze_mode == 'partial':
        # Freeze everything up to (but not including) layer4
        for name, param in model.named_parameters():
            if not (name.startswith('layer4') or name.startswith('fc')):
                param.requires_grad = False

    # freeze_mode == 'none' → nothing frozen; all params remain trainable

    # ---------- Classifier head ----------
    if modified_head:
        model.fc = nn.Sequential(
            nn.Linear(fc_in_feat, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 4),
        )
    else:
        model.fc = nn.Linear(fc_in_feat, 4)

    # ---------- Optimizer (only over trainable params) ----------
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=LR_PHASE1)

    return model, optimizer


# ============================================================
#  ATTENTION MAP  (Feature-map self-similarity)
# ============================================================
#
#  Strategy: hook the output of the last residual block (layer4).
#  For each spatial position we compute its cosine similarity to the
#  global-average-pooled feature vector — positions that strongly
#  "agree" with the final representation are highlighted.
#  This gives an interpretable saliency map without requiring a
#  backward pass (unlike Grad-CAM) and works on standard ResNets.
# ============================================================

class AttentionMap:
    """
    Produces a spatial saliency map by correlating each spatial position in
    the last convolutional feature map with the global average-pooled descriptor.

    Usage
    -----
        attn = AttentionMap(model, target_layer=model.layer4)
        cam, pred_class = attn(img_tensor)   # img_tensor: (1, C, H, W) on device
        attn.remove()
    """

    def __init__(self, model, target_layer):
        self.model      = model
        self._features  = None
        self._hook      = target_layer.register_forward_hook(self._save_features)

    def _save_features(self, module, input, output):
        # output: (B, C, H, W)
        self._features = output.detach()

    def __call__(self, x):
        """
        Parameters
        ----------
        x : (1, C, H, W) tensor on the correct device

        Returns
        -------
        cam        : (h, w) numpy array in [0, 1]  — spatial saliency
        pred_class : int — predicted class index
        """
        self._features = None
        self.model.eval()

        with torch.no_grad():
            output = self.model(x)
            pred_class = output.argmax(dim=1).item()

        if self._features is None:
            raise RuntimeError("AttentionMap: forward hook did not fire.")

        feats = self._features[0]           # (C, h, w)
        C, h, w = feats.shape

        # Global descriptor: average pool over spatial dims → (C,)
        descriptor = feats.mean(dim=(1, 2))  # (C,)
        descriptor = descriptor / (descriptor.norm() + 1e-8)

        # Cosine similarity of each spatial position to the descriptor
        feats_flat = feats.view(C, -1).permute(1, 0)    # (h*w, C)
        feats_norm = feats_flat / (feats_flat.norm(dim=1, keepdim=True) + 1e-8)
        sim = feats_norm @ descriptor                    # (h*w,)
        cam = sim.view(h, w).cpu().numpy()

        # ReLU + normalise to [0, 1]
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, pred_class

    def remove(self):
        self._hook.remove()


def overlay_heatmap(img_tensor, heatmap, alpha=0.4):
    """
    Blend an attention heatmap onto the de-normalised image.

    Parameters
    ----------
    img_tensor : (C, H, W) normalised tensor (CPU)
    heatmap    : (h, w) numpy array in [0, 1]

    Returns
    -------
    (H, W, 3) uint8 numpy array
    """
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    img_uint8 = (img * 255).astype(np.uint8)

    heatmap_resized = cv2.resize(heatmap, (img_uint8.shape[1], img_uint8.shape[0]))
    heatmap_uint8   = (heatmap_resized * 255).astype(np.uint8)
    heatmap_color   = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color   = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlaid = (alpha * heatmap_color + (1 - alpha) * img_uint8).astype(np.uint8)
    return overlaid


def log_attention_maps(model, loader, epoch, num_images=ATTN_IMAGES):
    """
    Run the attention-map visualiser on `num_images` samples and log to W&B.
    """
    model.eval()
    attn = AttentionMap(model, target_layer=model.layer4)

    images_logged = 0
    wandb_images  = []

    for images, labels in loader:
        for i in range(images.size(0)):
            if images_logged >= num_images:
                break

            img        = images[i].unsqueeze(0).to(device)
            label      = labels[i].item()
            heatmap, pred_class = attn(img)
            overlaid   = overlay_heatmap(images[i], heatmap)

            caption = (f"True: {class_names[label]} | "
                       f"Pred: {class_names[pred_class]}")
            wandb_images.append(wandb.Image(overlaid, caption=caption))
            images_logged += 1

        if images_logged >= num_images:
            break

    attn.remove()
    wandb.log({"attention_maps": wandb_images, "epoch": epoch})
    print(f"[ATTN] Logged {images_logged} attention maps (epoch {epoch})")


# ============================================================
#  BUILD MODEL
# ============================================================

model, optimizer = build_model(BACKBONE, MODIFIED_HEAD, FREEZE_MODE)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)


# ============================================================
#  W&B INIT
# ============================================================

def freeze_mode_label(fm, two_phase):
    if fm == 'all_frozen' and two_phase:
        return 'head_then_layer4'
    return fm

config = dict(
    backbone         = BACKBONE,
    modified_head    = MODIFIED_HEAD,
    freeze_mode      = freeze_mode_label(FREEZE_MODE, TWO_PHASE),
    dataset          = 'Dog Emotion',
    batch_size       = BATCH_SIZE,
    epochs_phase1    = EPOCHS_PHASE1,
    epochs_phase2    = EPOCHS_PHASE2 if (TWO_PHASE and FREEZE_MODE == 'all_frozen') else 0,
    lr_phase1        = LR_PHASE1,
    lr_phase2        = LR_PHASE2,
    optimizer        = 'Adam',
    scheduler        = 'ReduceLROnPlateau',
    num_classes      = 4,
    attn_layer       = 'layer4',
    attn_images_per_epoch = ATTN_IMAGES,
)

wandb.login()
run_name = (f"{BACKBONE}"
            f"_{'mlp' if MODIFIED_HEAD else 'linear'}"
            f"_{freeze_mode_label(FREEZE_MODE, TWO_PHASE)}")
run = wandb.init(project="Dog Emotion Detection", config=config, name=run_name)


# ============================================================
#  TRAINING & EVALUATION
# ============================================================

def evaluate(model, loader):
    model.eval()
    correct, total, total_loss = 0, 0, 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs        = model(images)
            total_loss    += criterion(outputs, labels).item() * labels.size(0)
            _, predicted   = torch.max(outputs, dim=1)
            total         += labels.size(0)
            correct       += (predicted == labels).sum().item()

    return total_loss / total, correct / total


def training_loop(model, train_loader, val_loader, optimizer, scheduler,
                  epochs=25, log_attn_every=LOG_ATTN_EVERY, phase_label=""):
    """
    One training phase.  optimizer and scheduler are passed in so that
    two-phase runs can swap them between phases.
    """
    print(f'\n--- {"Phase: " + phase_label if phase_label else "Training"} ({epochs} epochs) ---')
    print(f'{"Epoch":>5} | {"Train Loss":>10} | {"Train Acc":>9} | '
          f'{"Val Loss":>8} | {"Val Acc":>7} | {"LR":>8}')

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss   = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, predicted  = torch.max(output, dim=1)
            total        += labels.size(0)
            correct      += (predicted == labels).sum().item()

        train_acc  = correct / total
        train_loss = running_loss / total
        val_loss, val_acc = evaluate(model, val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)

        wandb.log({
            'epoch':      epoch,
            'train/acc':  train_acc,
            'train/loss': train_loss,
            'val/acc':    val_acc,
            'val/error':  1 - val_acc,
            'lr':         current_lr,
        })

        print(f'{epoch:>5} | {train_loss:>10.4f} | {train_acc*100:>8.2f}% | '
              f'{val_loss:>8.4f} | {val_acc*100:>6.2f}% | {current_lr:>8.5f}')

        # Log attention maps periodically
        if (epoch + 1) % log_attn_every == 0 or epoch == epochs - 1:
            log_attention_maps(model, val_loader, epoch)


# ---- Phase 1 ----------------------------------------------------------------

training_loop(model, train_loader, val_loader, optimizer, scheduler,
              epochs=EPOCHS_PHASE1, phase_label="Phase 1")

# ---- Phase 2 (optional: unfreeze layer4 and fine-tune) ----------------------

if TWO_PHASE and FREEZE_MODE == 'all_frozen':
    print("\n[Phase 2] Unfreezing layer4 for fine-tuning …")
    for param in model.layer4.parameters():
        param.requires_grad = True

    optimizer_p2 = optim.Adam([
        {"params": model.layer4.parameters(), "lr": LR_PHASE2},
        {"params": model.fc.parameters(),     "lr": LR_PHASE1},
    ])
    scheduler_p2 = ReduceLROnPlateau(optimizer_p2, mode='max', factor=0.5, patience=2)

    training_loop(model, train_loader, val_loader, optimizer_p2, scheduler_p2,
                  epochs=EPOCHS_PHASE2, phase_label="Phase 2 (layer4 unfrozen)")


# ============================================================
#  FINAL EVALUATION
# ============================================================

test_loss, test_acc = evaluate(model, test_loader)
print(f"\nTest Accuracy: {test_acc*100:.2f}%  |  Test Loss: {test_loss:.4f}")

log_attention_maps(model, test_loader, epoch=-1, num_images=16)

wandb.summary['test_acc']   = test_acc
wandb.summary['test_loss']  = test_loss
wandb.summary['test_error'] = 1 - test_acc

model_filename = f"{run_name}.pth"
torch.save(model.state_dict(), model_filename)
wandb.save(model_filename)
wandb.finish()