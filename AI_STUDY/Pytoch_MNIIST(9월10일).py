import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

# MNIST 데이터 load
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

# CNN
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # 입력 채널 1 (MNIST 이미지는 흑백), 출력 채널 4, 커널 사이즈 3x3
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        # 입력 채널 4, 출력 채널 8, 커널 사이즈 3x3
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        # MaxPooling 레이어 2X2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 완전 연결 레이어 392 -> 128 -> 10
        self.fc1 = nn.Linear(8 * 7 * 7, 128)  # 두 번째 풀링 후 8채널 7x7 크기
        self.fc2 = nn.Linear(128, 10)  # 10개의 클래스로 분류 (MNIST 숫자 0-9)

    # def forward(self, x):
    #     # 첫 번째 컨볼루션 + Sigmoid + MaxPooling 
    #     #x = self.pool(torch.sigmoid(self.conv1(x)))
    #     # 시그모이드x
    #     x = self.pool(self.conv1(x))
    #     # 두 번째 컨볼루션 + Sigmoid + MaxPooling
    #     #x = self.pool(torch.sigmoid(self.conv2(x)))
    #     # 시그모이드x
    #     x = self.pool(self.conv2(x))
    #     # 1차원으로 변환
    #     x = x.view(-1, 8 * 7 * 7)
    #     # 완전 연결 레이어 통과
    #     x = torch.sigmoid(x)
    #     x = self.fc1(x)
    #     x = torch.sigmoid(x)
    #     x = self.fc2(x)
    #     return x

    def forward(self, x):
        # 첫 번째 컨볼루션 + 풀링
        x = self.conv1(x)
        conv1_output = x  # conv1 출력 저장
        x = self.pool(x)
        
        # 두 번째 컨볼루션 + 풀링
        x = self.conv2(x)
        conv2_output = x  # conv2 출력 저장
        x = self.pool(x)
        
        x = x.view(-1, 8 * 7 * 7)
        x = torch.sigmoid(x)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        
        return x, conv1_output, conv2_output
    
# CNN 레이어의 필터 가중치를 시각화하는 함수
def visualize_filter_weights(conv_layer, layer_name):
    filters = conv_layer.weight.data.cpu().numpy()  # 필터 가중치 값을 가져옴 (NumPy로 변환)
    num_filters = filters.shape[0]  # 필터 개수 (출력 채널 개수)
    
    fig, axes = plt.subplots(1, num_filters, figsize=(15, 5))
    for i in range(num_filters):
        ax = axes[i]
        # 필터 가중치를 격자형 배열로 시각화
        ax.imshow(filters[i, 0, :, :], cmap='coolwarm', interpolation='none')  # 첫 번째 입력 채널 필터만 시각화
        ax.set_title(f"{layer_name} Filter {i}")
        ax.axis('off')
    plt.colorbar(ax.imshow(filters[0, 0, :, :], cmap='coolwarm', interpolation='none'), ax=axes, orientation='vertical', fraction=0.05, pad=0.05)
    plt.show()


# 타겟을 원-핫 인코딩으로 변환하는 함수
def one_hot_encode(labels, num_classes):
    return torch.eye(num_classes)[labels]

# 모델, 손실 함수 및 옵티마이저 초기화
model = CNNModel()
criterion = nn.MSELoss()  # MSE 손실 함수
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # SGD 옵티마이저

# 학습 함수
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output, conv1_output, conv2_output = model(data)  # conv1, conv2 출력 받기

        # 타겟을 원-핫 인코딩으로 변환
        target_one_hot = one_hot_encode(target, 10).to(device)
        # MSE 손실 계산
        loss = criterion(output, target_one_hot)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            
            if epoch == 10:
            
                # 100번째 배치마다 필터 가중치 시각화
                print("Visualizing conv1 filter weights...")
                visualize_filter_weights(model.conv1, "Conv1")
            
                print("Visualizing conv2 filter weights...")
                visualize_filter_weights(model.conv2, "Conv2")


# 테스트 함수
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _, _ = model(data)
            # 타겟을 원-핫 인코딩으로 변환
            target_one_hot = one_hot_encode(target, 10).to(device)
            # MSE 손실 계산
            test_loss += criterion(output, target_one_hot).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.8f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')


# 학습 및 테스트 실행
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(1, 11):  # 10 에폭 학습
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)