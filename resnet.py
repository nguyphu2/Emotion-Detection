import kagglehub
import torch
import wandb
import torch.nn as nn  
import torch.optim as optim    
from torch.optim.lr_scheduler import ReduceLROnPlateau    
from torchvision import datasets, transforms, models    
from torch.utils.data import DataLoader, random_split  




# ---- GPU ----

device = torch.device("cuda" if torch.cuda_is_available() else "cpu")

# ---- Define Dataset ----

root = r"C:\Users\nguyphu2\Downloads\DL\kagglehub"


# ---- Data Preprocessing/ Augmentation ----
training_transform = transforms.Compose([transforms.Resize((224,224)), 
                                        transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), 
                                        transforms.RandomResizedCrop(224, scale = (0.8,1.0)), transforms.ColorJitter(brightness =0.2, contrast = 0.2),
                                        transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
                                        ])

testing_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


# ---- Load Dataset ---- 

dataset = datasets.ImageFolder(root = root, transform = training_transform)


# ---- Splitting Dataset ----

"""
70% train, 15% validation, 15% test

"""
data_size = len(dataset)

train_size = int(0.7 * data_size)

val_size = int(0.3*data_size)

test_size = int(0.3 * data_size)

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])


# ---- DataLoader ----

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ---- Model ----

model = models.resnet50(pretrained = True)

for param in model.parameters():
    param.requires_grad = False
    
model.fc = nn.Sequential(
    nn.Linear(2048,512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512,4)
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.fc.parameters(), lr = 1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)


# ---- WandB Config ----

config = dict(
    architecture='ResNet50',
    dataset='Dog Emotion',
    batch_size=32,
    epochs_phase1=25,
    epochs_phase2=10,
    learning_rate_phase1=1e-4,
    learning_rate_phase2=1e-5,
    optimizer='Adam',
    scheduler='ReduceLROnPlateau',
    num_classes=4,
)

wandb.login()
run = wandb.init(project = "Dog Emotion Detection", config = config, name = "resnet50_n")



# ---- Training Loop ----
def training_loop(model, train_loader, val_loader, scheduler, epochs = 25):
    print(f'\n{"Epoch":} | {"Train Loss":} | {"Train Acc":} | {"Val Loss":} | {"Val Acc":} | {"LR":}')
    for epoch in range(epochs):
        model.train()
        
        running_loss, correct, total = 0,0,0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            output = model(images)
            
            loss = criterion(output, labels)
            
            loss.backward()
            
            optimizer.step()
            
            running_loss += loss.item() * labels.size(0)
            
            _, predicted = torch.max(output, dim = 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum.item()
            
        train_acc = correct / total
        train_loss = running_loss / total
        val_loss, val_acc = evaluate(model, val_loader)
        
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Acc: {train_acc:.3f}")
        print(f"Val Acc: {val_acc:.3f}")
        
        
        wandb.log({
            'epoch': epoch,
            'train/acc': train_acc,
            'train/loss': train_loss,
            'val/acc': val_acc,
            'val/error': 100.0- val_acc,
            'lr': current_lr,
        })
        
        print(f'{epoch} | {train_loss:.4f} | {train_acc:.2f}% | {val_loss:4f} | {val_acc:.2f}% | {current_lr:.5f}')
        
        
def evaluate(model, load):
    model.eval()
    
    correct, total = 0,0
    
    with torch.no_grad():
        for image, labels in load:
            image = image.to(device)
            labels = labels.to(device)
            
            outputs = model(image)
            total_loss += criterion(outputs, labels).item() * labels.size(0)
            _, predicted = torch.max(outputs, dim = 1)
            
            total += labels.size(0)
            
            correct += (predicted == labels).sum.item()
        
    return total_loss/ total, 100.0 * correct/total



# ---- Train ---- 
training_loop(model, train_loader, val_loader, scheduler, epochs = 25)


for param in model.layer4.parameters():
    param.requires_grad = True
    
optimizer = optim.Adam(model.parameters(), lr = 1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)


training_loop(model, train_loader, val_loader, scheduler, epochs = 10)

# ---- Testing ----

test_loss, test_acc = evaluate(model, test_loader)


#---- WandB Test Results ----
wandb.log({
        'test/acc': test_acc,
        'test/error': 100.0- test_acc,
})

wandb.summary['test_acc'] = test_acc
wandb.summary['test_loss'] = test_loss
torch.save(model.state_dict(), 'resnet50.pth')
wandb.save('resnet50.pth')
wandb.finish()

            
            
        
    
            
            

            
        
        
        