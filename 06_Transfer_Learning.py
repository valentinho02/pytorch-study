import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import shutil
from torchvision import datasets, transforms
import torchvision.models as models


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")

EPOCHS = 10
BATCH_SIZE = 32

train_dir = './data/cats_dogs_light/train'
validation_dir = './data/cats_dogs_light/test'


def rearrange_for_image_folder(base_dir):
    if not os.path.exists(base_dir):
        return
        

    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    

    if not subdirs or not ('cat' in "".join(subdirs).lower() or 'dog' in "".join(subdirs).lower()):
        print(f"[{base_dir}] 구조 재배치 중... (PyTorch ImageFolder 규격 맞춤)")
        

        cat_folder = os.path.join(base_dir, 'cat')
        dog_folder = os.path.join(base_dir, 'dog')
        os.makedirs(cat_folder, exist_ok=True)
        os.makedirs(dog_folder, exist_ok=True)
        
        for file in os.listdir(base_dir):
            file_path = os.path.join(base_dir, file)
            if os.path.isfile(file_path):
                file_lower = file.lower()
                if 'cat' in file_lower:
                    shutil.move(file_path, os.path.join(cat_folder, file))
                elif 'dog' in file_lower:
                    shutil.move(file_path, os.path.join(dog_folder, file))

rearrange_for_image_folder(train_dir)
rearrange_for_image_folder(validation_dir)


train_config = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

validation_config = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_dataset = datasets.ImageFolder(train_dir, train_config)
validation_dataset = datasets.ImageFolder(validation_dir, validation_config)

train_dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_dataset_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=BATCH_SIZE, shuffle=False)


pretrained_model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

class MyTransferModel(torch.nn.Module):
    def __init__(self, pretrained_model, feature_extractor):
        super().__init__()

        if feature_extractor:
            for param in pretrained_model.parameters():
                param.requires_grad = False
        
        in_features = pretrained_model.heads.head.in_features
        pretrained_model.heads.head = torch.nn.Sequential(
            torch.nn.Linear(in_features, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, 2)
        )

        self.model = pretrained_model
    
    def forward(self, data):
        return self.model(data)

feature_extractor = True 
model = MyTransferModel(pretrained_model, feature_extractor).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), 1e-6)
loss_function = torch.nn.CrossEntropyLoss()


def model_train(dataloader, model, loss_function, optimizer):
    model.train()
    train_loss_sum = train_correct = train_total = 0
    total_train_batch = len(dataloader)

    for images, labels in dataloader:
        x_train = images.to(DEVICE)
        y_train = labels.to(DEVICE)

        outputs = model(x_train)
        loss = loss_function(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()
        train_total += y_train.size(0)
        train_correct += (torch.argmax(outputs, 1) == y_train).sum().item()
    
    return (train_loss_sum / total_train_batch, 100 * train_correct / train_total)

def model_evaluate(dataloader, model, loss_function):
    model.eval()
    with torch.no_grad():
        val_loss_sum = val_correct = val_total = 0
        total_val_batch = len(dataloader)

        for images, labels in dataloader:
            x_val = images.to(DEVICE)
            y_val = labels.to(DEVICE)

            outputs = model(x_val)
            loss = loss_function(outputs, y_val)

            val_loss_sum += loss.item()
            val_total += y_val.size(0)
            val_correct += (torch.argmax(outputs, 1) == y_val).sum().item()
        
    return (val_loss_sum / total_val_batch, 100 * val_correct / val_total)

# 6. 메인 루프 실행
print("학습을 시작합니다...")
for epoch in range(EPOCHS):
    train_avg_loss, train_avg_accuracy = model_train(train_dataset_loader, model, loss_function, optimizer)
    val_avg_loss, val_avg_accuracy = model_evaluate(validation_dataset_loader, model, loss_function)

    print(f"[EPOCH: {epoch+1:02d}], \tTrain Loss: {train_avg_loss:.4f}, \tTrain Accuracy: {train_avg_accuracy:.2f}%, \tVal Loss: {val_avg_loss:.4f}, \tVal Accuracy: {val_avg_accuracy:.2f}%")