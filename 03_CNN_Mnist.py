import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import random_split

Device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {Device}")
EPOCHS = 20
BATCH_SIZE = 32

train_dataset = datasets.MNIST(root='./data/MNIST', train=True, download  =True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='./data/MNIST', train=False, download=True, transform=transforms.ToTensor())

train_dataset_size = int(len(train_dataset) * 0.85)
validation_dataset_size = int(len(train_dataset) * 0.15)

train_dataset, validation_dataset = random_split(train_dataset, [train_dataset_size, validation_dataset_size])

   
train_dataset_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True)
validation_dataset_loader = torch.utils.data.DataLoader(dataset = validation_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_dataset_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = True)


class MyCNNModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(7*7*64, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout25 = nn.Dropout(p=0.25)
        self.dropout50 = nn.Dropout(p = 0.5)
    
    def forward(self, data):
        data = self.conv1(data)
        data = torch.relu(data)
        data = self.pooling(data)
        data = self.dropout25(data)

        data = self.conv2(data)
        data = torch.relu(data)
        data = self.pooling(data)
        data = self.dropout25(data)

        data = data.view(-1, 7 * 7 * 64)
        data = self.fc1(data)
        data = torch.relu(data)
        data = self.dropout50(data)

        logits = self.fc2(data)

        return logits
    
model = MyCNNModel().to(Device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

def model_train(dataloader, model, loss_function, optimizer):
    model.train()

    train_loss_sum = train_correct = train_total = 0
    total_train_batch = len(dataloader)

    for images, labels in dataloader:
        x_train = images.to(Device)
        y_train = labels.to(Device)

        outputs = model(x_train)
        loss = loss_function(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()
        train_total += y_train.size(0)
        train_correct += ((torch.argmax(outputs, 1) == y_train)).sum().item()
    
    train_avg_loss = train_loss_sum / total_train_batch
    train_avg_accuaracy = 100 * train_correct / train_total
    return (train_avg_loss, train_avg_accuaracy)

def model_evaluate(dataloader, model, loss_function, optimizer):
    model.eval()

    with torch.no_grad():
        
        val_loss_sum = val_correct = val_total = 0

        total_val_batch = len(dataloader)

        for images, labels in dataloader:
            x_val = images.to(Device)
            y_val = labels.to(Device)

            outputs = model(x_val)
            loss = loss_function(outputs, y_val)

            val_loss_sum += loss.item()

            val_total += y_val.size(0)
            val_correct += ((torch.argmax(outputs, 1) == y_val)).sum().item()
        
        val_avg_loss = val_loss_sum / total_val_batch
        val_avg_accuarcy = 100 * val_correct / val_total
    return (val_avg_loss, val_avg_accuarcy)


train_loss_list = []
train_accuracy_list = []

val_loss_list = []
val_accuracy_list = []

for epoch in range(EPOCHS):

    #=================model train=========================
    train_avg_loss, train_avg_accuracy = model_train(train_dataset_loader, model, loss_function, optimizer)

    train_loss_list.append(train_avg_loss)
    train_accuracy_list.append(train_avg_accuracy)


    #============model evaliation==============
    val_avg_loss, val_avg_accuracy = model_evaluate(validation_dataset_loader, model, loss_function, optimizer)
    val_loss_list.append(val_avg_loss)
    val_accuracy_list.append(val_avg_accuracy)

    print(f"\n[EPOCH: {epoch}], \tTest Loss: {train_avg_loss:.4f}, \tTest Accuracy: {train_avg_accuracy:.2f}, \tvalidation loss: {val_avg_loss:.4f}, \tvalidation_accuaracy = {val_avg_accuracy:.2f} %\n")