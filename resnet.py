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

# ---- GPU ----

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


MODIFIED_RESNET = True

# ---- Define Dataset ----

path = kagglehub.dataset_download("danielshanbalico/dog-emotion")
root = os.path.join(path, "Dog Emotion")  # angry, happy, relaxed, sad live here
base_dataset = datasets.ImageFolder(root=root)


# ---- Data Preprocessing / Augmentation ----

training_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

testing_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---- Load & Split Dataset ----

base_dataset = datasets.ImageFolder(root=root)
data_size = len(base_dataset)
class_names = base_dataset.classes  # e.g. ['angry', 'happy', 'relaxed', 'sad']
print(class_names)

train_size = int(0.7 * data_size)
val_size   = int(0.15 * data_size)
test_size  = data_size - train_size - val_size

train_indices, val_indices, test_indices = random_split(
    range(data_size), [train_size, val_size, test_size]
)

train_dataset = torch.utils.data.Subset(
    datasets.ImageFolder(root=root, transform=training_transform), train_indices.indices
)
val_dataset = torch.utils.data.Subset(
    datasets.ImageFolder(root=root, transform=testing_transforms), val_indices.indices
)
test_dataset = torch.utils.data.Subset(
    datasets.ImageFolder(root=root, transform=testing_transforms), test_indices.indices
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=2)

# ---- Model ----

model = models.resnet50(pretrained=True)
if MODIFIED_RESNET:
  for param in model.parameters():
      param.requires_grad = False

#---- Modified Architecture ----
  model.fc = nn.Sequential(
      nn.Linear(2048,512),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(512,4)
  )

  optimizer = optim.Adam(model.fc.parameters(), lr = 1e-4)
# ---- Standard ResNet (Set MODIFIED_RESNET to false if you want standard resnet-50) ----
else:
  model.fc = nn.Linear(model.fc.in_features, 4)
  optimizer = optim.Adam(model.parameters(), lr = 1e-4)


model = model.to(device)

criterion = nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)


# ============================
# GRAD-CAM
# ============================

class GradCAM:


    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x):
      self.gradients = None
      self.activations = None

      self.model.zero_grad()

      with torch.enable_grad():
          x = x.detach().requires_grad_(True)
          output = self.model(x)
          pred_class = output.argmax(dim=1).item()
          output[0, pred_class].backward()

      if self.gradients is None:
          raise RuntimeError("GradCAM: backward hook did not fire.")

      weights = self.gradients.mean(dim=(2, 3), keepdim=True)
      cam = (weights * self.activations).sum(dim=1).squeeze()
      cam = torch.relu(cam).cpu().numpy()

      if cam.max() > 0:
          cam = cam / cam.max()

      return cam, pred_class

    def remove(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def overlay_heatmap(img_tensor, heatmap, alpha=0.4):
    """
    Blend a Grad-CAM heatmap onto the original image.
    img_tensor : (C, H, W) normalised tensor
    heatmap    : (h, w) numpy array in [0, 1]
    Returns    : (H, W, 3) uint8 numpy array
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


def log_attention_maps(model, val_loader, epoch, num_images=8):
    """
    Runs Grad-CAM on `num_images` validation samples and logs them to W&B
    as a single image grid labelled with true / predicted class names.
    """
    model.eval()
    gcam = GradCAM(model, target_layer=model.layer4)

    images_logged = 0
    wandb_images  = []

    for images, labels in val_loader:
        for i in range(images.size(0)):
            if images_logged >= num_images:
                break

            img   = images[i].unsqueeze(0).to(device)   # (1, C, H, W)
            label = labels[i].item()

            heatmap, pred_class = gcam(img)

            overlaid = overlay_heatmap(images[i], heatmap)

            true_name = class_names[label]
            pred_name = class_names[pred_class]
            caption   = f"True: {true_name} | Pred: {pred_name}"

            wandb_images.append(wandb.Image(overlaid, caption=caption))
            images_logged += 1

        if images_logged >= num_images:
            break

    gcam.remove()
    wandb.log({"attention_maps": wandb_images, "epoch": epoch})
    print(f"[GRADCAM] Logged {images_logged} attention maps to W&B (epoch {epoch})")

# ---- WandB Config ----

config = dict(
    architecture='ResNet50',
    dataset='Dog Emotion',
    batch_size=64,
    epochs_phase1=35,
    epochs_phase2=20 if MODIFIED_RESNET else 0,
    learning_rate_phase1=1e-4,
    learning_rate_phase2=1e-5,
    optimizer='Adam',
    scheduler='ReduceLROnPlateau',
    num_classes=4,
    gradcam_layer='layer4',
    gradcam_images_per_epoch=8,
)

wandb.login()
run_name = "resnet50_modified_gradcam" if MODIFIED_RESNET else "resnet50_baseline"
run = wandb.init(project="Dog Emotion Detection", config=config, name=run_name)


# ============================
# TRAINING LOOP
# ============================

def training_loop(model, train_loader, val_loader, scheduler, epochs=25, log_gradcam_every=5):
    """
    log_gradcam_every : log attention maps every N epochs (and always on the last epoch).
    """
    print(f'\n{"Epoch"} | {"Train Loss"} | {"Train Acc"} | {"Val Loss"} | {"Val Acc"} | {"LR"}')

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

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
            'epoch':       epoch,
            'train/acc':   train_acc,
            'train/loss':  train_loss,
            'val/acc':     val_acc,
            'val/error':   1- val_acc,
            'lr':          current_lr,
        })

        print(f'{epoch} | {train_loss:.4f} | {train_acc*100:.2f}% | {val_loss:.4f} | {val_acc:.2f}% | {current_lr:.5f}')

        # Log Grad-CAM attention maps periodically
        if (epoch + 1) % log_gradcam_every == 0 or epoch == epochs - 1:
            log_attention_maps(model, val_loader, epoch)


def evaluate(model, loader, test=False):
    model.eval()
    correct, total, total_loss = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * labels.size(0)
            _, predicted = torch.max(outputs, dim=1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total_loss / total, correct / total




training_loop(model, train_loader, val_loader, scheduler, epochs=35, log_gradcam_every=5)



if MODIFIED_RESNET:
  for param in model.layer4.parameters():
      param.requires_grad = True

  optimizer = optim.Adam([
      {"params": model.layer4.parameters(), "lr": 1e-5},
      {"params": model.fc.parameters(),     "lr": 1e-4},
  ])

  scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
  training_loop(model, train_loader, val_loader, scheduler, epochs=20, log_gradcam_every=5)


test_loss, test_acc = evaluate(model, test_loader, test=True)

log_attention_maps(model, test_loader, epoch=-1, num_images=16)

wandb.summary['test_acc']   = test_acc
wandb.summary['test_loss']  = test_loss
wandb.summary['test_error'] = 100 - test_acc

torch.save(model.state_dict(), 'resnet50.pth')
wandb.save('resnet50.pth')
wandb.finish()

