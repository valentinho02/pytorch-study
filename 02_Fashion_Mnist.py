import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import random_split

Device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(Device)
BATCH_SIZE = 32
EPOCHS = 20

train_dataset = datasets.FashionMNIST(root='./data/FashionMNIST', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.FashionMNIST(root='./data/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())

print(len(train_dataset))
train_dataset_size = int(len(train_dataset) * 0.85)
validation_dataset_size = int(len(train_dataset) * 0.15)
train_dataset, validation_dataset = random_split(train_dataset, [train_dataset_size, validation_dataset_size])

print(len(train_dataset), len(validation_dataset), len(test_dataset))

class MyDeepLearningModel(nn.Module):
    def __init__(self): # 아키텍처를 구성하는 다양한 계층 정의
        super().__init__()
        self.flatten = nn.Flatten() 
        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, data): #피드 포워드 수행
        data = self.flatten(data)  # 입력층
        data = self.fc1(data)  #은닉층
        data = self.relu(data) # ReLu 비선형 함수
        data = self.dropout(data) # overfitting을 방지하기 위한 dropout
        logits = self.fc2(data) # 출력층
        return logits
    
train_dataset_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True)
validation_dataset_loader = torch.utils.data.DataLoader(dataset = validation_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_dataset_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = True)

model = MyDeepLearningModel().to(Device)

loss_function = nn.CrossEntropyLoss() #croossEntropyLoss(손실함수)에는 softmax가 포함되어 있다!
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2)

def model_train(dataloader, model, loss_function, optimizer):
    model.train() # 신경망을 학습모드( 파라미터를 업데이트 하는 모드) 로 전환

    train_loss_sum = train_correct = train_total = 0
    total_train_batch = len(dataloader)

    for images, labels in dataloader:
        x_train = images.view(-1, 28 * 28).to(Device) # 이미지를 1차원 벡터로 변환
        y_train = labels.to(Device)

        outputs = model(x_train) # 입력 데이터에 대해 예측값 계싼
        loss = loss_function(outputs, y_train) # loss 손실함수 계산

        optimizer.zero_grad() # 기울기 초기화
        loss.backward() # 역전파 수행
        optimizer.step() # 가중치 업데이트

        train_loss_sum += loss.item() # 손실함수 값 누적
        train_total += y_train.size(0)
        train_correct += ((torch.argmax(outputs, 1)==y_train)).sum().item()
    
    train_avg_loss = train_loss_sum / total_train_batch
    train_avg_accuracy = 100 * (train_correct / train_total)
    return (train_avg_loss, train_avg_accuracy)

def model_evaluate(dataloader, model, lossfunction, optimizer):
    model.eval() #신경망을 추론모드로 전환
    with torch.no_grad(): # 파라미터를 업데이트 시키지 않으므로 미분할 필요가 없음
        val_loss_sum = val_correct = val_total = 0
        total_val_batch = len(dataloader)

        for images, labels in dataloader:
            x_val = images.view(-1, 28 * 28).to(Device)
            y_val = labels.to(Device)

            outputs = model(x_val)
            loss = loss_function(outputs, y_val)

            val_loss_sum += loss.item()
            val_total += y_val.size(0)
            val_correct += ((torch.argmax(outputs, 1) == y_val)).sum().item()

        val_avg_loss = val_loss_sum / total_val_batch
        val_avg_accuaracy = 100 * (val_correct / val_total)
    
    return (val_avg_loss, val_avg_accuaracy)

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