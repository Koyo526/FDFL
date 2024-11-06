import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.metrics import accuracy_score
import numpy as np

# シンプルなCNNモデル定義
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# デバイスクラス定義
class Device:
    def __init__(self, device_id, train_loader, test_loader):
        self.device_id = device_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = SimpleCNN().to('cpu')  # 各デバイスのローカルモデル
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()

    def local_train(self, epochs=1):
        self.model.train()
        for _ in range(epochs):
            for images, labels in self.train_loader:
                images, labels = images.to('cpu'), labels.to('cpu')
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

        return self.model.state_dict()

    def evaluate(self, global_model_state):
        # グローバルモデルの状態をコピーして評価
        self.model.load_state_dict(global_model_state)
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to('cpu'), labels.to('cpu')
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())

        return accuracy_score(all_labels, all_preds)

# ブロックチェーンクラスの定義
class Blockchain:
    def __init__(self):
        self.chain = []
        self.global_model = SimpleCNN().to('cpu')  # グローバルモデル
        self.global_state = self.global_model.state_dict()

    def aggregate_weights(self, devices):
        # 各デバイスの重みを集約（平均）
        new_state = {}
        for key in self.global_state.keys():
            new_state[key] = torch.stack([torch.Tensor(device.model.state_dict()[key]) for device in devices]).mean(0)
        self.global_state = new_state
        self.global_model.load_state_dict(self.global_state)
        return self.global_state

    def record_contribution(self, device, score):
        # 各デバイスの貢献度スコアをブロックに記録
        block = {
            'device_id': device.device_id,
            'contribution_score': score
        }
        self.chain.append(block)

# クロスバリデーションによる評価
def cross_validation_evaluation(devices, blockchain):
    for device in devices:
        score_sum = 0
        for other_device in devices:
            if other_device.device_id != device.device_id:
                score = other_device.evaluate(blockchain.global_state)
                score_sum += score
        contribution_score = score_sum / (len(devices) - 1)
        blockchain.record_contribution(device, contribution_score)

# 連合学習ラウンドの定義
def federated_learning_round(devices, blockchain, epochs=1):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        # 各デバイスがローカルでモデルをトレーニング
        for device in devices:
            device.local_train()

        # グローバルモデルの更新
        blockchain.aggregate_weights(devices)

        # クロスバリデーション評価
        cross_validation_evaluation(devices, blockchain)

        # 各デバイスの貢献度スコアを表示
        for block in blockchain.chain:
            print(f"Device {block['device_id']} contribution score: {block['contribution_score']:.4f}")

        # スコアのリセット
        blockchain.chain = []

# データの前処理とローダー作成
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CIFAR-10データセットのダウンロードと分割
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# デバイスごとにデータを分割
num_devices = 5
data_size = len(train_dataset) // num_devices
test_size = len(test_dataset) // num_devices
devices = []
for i in range(num_devices):
    train_data, _ = data.random_split(train_dataset, [data_size, len(train_dataset) - data_size])
    test_data, _ = data.random_split(test_dataset, [test_size, len(test_dataset) - test_size])
    train_loader = data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = data.DataLoader(test_data, batch_size=32, shuffle=False)
    devices.append(Device(device_id=i, train_loader=train_loader, test_loader=test_loader))

# ブロックチェーンの初期化
blockchain = Blockchain()

# 連合学習の実行
federated_learning_round(devices, blockchain, epochs=3)
