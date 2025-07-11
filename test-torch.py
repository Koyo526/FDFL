import csv
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import numpy as np
import os
import datetime
from typing import List
import hashlib


now = datetime.datetime.now()
current_time = now.strftime("%Y-%m-%d-%H-%M")

SAVE_DIR = f"data/{current_time}/img"
os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_CSV_DIR = f"data/{current_time}/csv"
os.makedirs(SAVE_CSV_DIR, exist_ok=True)

LOCAL_MODELS = []
LOCAL_MODEL_HASH =[]
LOCAL_MODEL_EVACULATE = []
LOCAL_MODEL_EVACULATE_HASH = []

# シンプルなCNNモデル定義
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# デバイスクラス定義
class Device:
    def __init__(self, device_id, train_loader, test_loader):
        self.device_id = device_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = Net().to('cpu')  # 各デバイスのローカルモデル
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        self.loss = []
        self.accuracy = []

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

    def evaluate(self, local_model_state):
        # グローバルモデルの状態をコピーして評価
        self.model.load_state_dict(local_model_state)
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to('cpu'), labels.to('cpu')
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                _, preds = torch.max(outputs, 1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        # 損失と精度を記録
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples


        print(f"Global Model - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        return avg_loss, accuracy
    
    # def check_hash(hash_data,data) -> None:
    #     if (hash_data == hashlib.sha256(data.encode())):
    #         loss = Device.evaluate(data)
        
    #     pass


# ブロックチェーンクラスの定義
class Blockchain:
    def __init__(self,Dataset):
        self.chain = []
        self.global_model = Net().to('cpu')  # グローバルモデル
        self.global_state = self.global_model.state_dict()
        self.contribution_scores = {}  # 各デバイスの貢献度スコアを記録
        self.global_model_loss_sum_client_evaculate = []
        

        #下記はテスト用の設定
        self.evaluate_dataset = Dataset #検証用データセット
        self.global_model_loss = [] #グローバルモデルのLoss
        self.global_model_accuracy = [] #グローバルモデルのAccuracy


    def aggregate_weights(self, devices:List[Device]):
        new_state = {}
        for key in self.global_state.keys():
            new_state[key] = torch.stack([torch.Tensor(device.model.state_dict()[key]) for device in devices],0).mean(0)
        self.global_model.load_state_dict(new_state, strict=True)
        self.global_state = self.global_model.state_dict()
        return self.global_state
    

    # def aggregate_weights(self, devices: List[Device]):
    #     new_state = {}
    #     total_contribution = 0
    #     for device in devices:
    #         score = self.contribution_scores[device.device_id]
    #         print(score)
    #         total_contribution += score[len(score)-1][1]

    #     for key in self.global_state.keys():
    #         new_state[key] = torch.stack([
    #             torch.Tensor(device.model.state_dict()[key]) * self.contribution_scores[device.device_id] / total_contribution
    #             for device in devices
    #         ]).sum(0)  # 重み付き平均を計算

    #     self.global_state = new_state
    #     self.global_model.load_state_dict(self.global_state)
    #     return self.global_state
    
    def evaluate_global_model(self):
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        all_preds = []
        all_labels = []
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for images, labels in self.evaluate_dataset:
                images, labels = images.to('cpu'), labels.to('cpu')
                outputs = self.global_model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                total_samples += labels.size(0)

        # 精度計算
        accuracy = accuracy_score(all_labels, all_preds)
        avg_loss = total_loss / total_samples
        self.global_model_loss.append(avg_loss)
        self.global_model_accuracy.append(accuracy)

        print(f"Global Model - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        return avg_loss, accuracy
    

    def add_block(self, state, loss, accuracy):
        block = {
            "state_dict": state,
            "loss": loss,
            "accuracy": accuracy,
            "timestamp": datetime.datetime.now().isoformat()
        }
        self.chain.append(block)
        print(f"Block added with timestamp {block['timestamp']}")

    def record_contribution(self, device:Device, score, round_num):
        # 各デバイスの貢献度スコアを記録
        if device.device_id not in self.contribution_scores:
            self.contribution_scores[device.device_id] = []
        self.contribution_scores[device.device_id].append((round_num, score))

    

    def save_contribution_scores(self, filename="contribution_scores.csv"):
        # 貢献度スコアをCSVファイルに保存
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Round", "Device ID", "Contribution Score"])
            for device_id, scores in self.contribution_scores.items():
                for round_num, score in scores:
                    writer.writerow([round_num + 1, device_id, score])

# クロスバリデーションによる評価
def cross_validation_evaluation(devices:List[Device], blockchain:Blockchain, round_num:int):
    for device in devices:
        score_sum = 0
        accuracy_sum = 0
        loss_sum = 0
        for other_device in devices:
            if other_device.device_id != device.device_id:
                # score = other_device.evaluate(blockchain.global_state)
                avg_loss, accuracy = other_device.evaluate(device.model.state_dict())
                score_sum += avg_loss
                loss_sum += avg_loss
                accuracy_sum += accuracy
        loss_ave = loss_sum/len(devices)-1
        accuracy_ave = accuracy_sum/(len(devices)-1)
        contribution_score = score_sum/ (len(devices) - 1)
        blockchain.record_contribution(device, contribution_score, round_num)
        print(f"Client{device.device_id}: Local Model - Loss: {loss_ave:.4f}, Accuracy: {accuracy_ave:.4f}")
        device.accuracy.append(accuracy_ave)
        device.loss.append(loss_ave)

# 連合学習ラウンドの定義
def federated_learning_round(devices:List[Device], blockchain:Blockchain, epochs=1):
    # for epoch in range(epochs):
    #     print(f"Epoch {epoch+1}/{epochs}")

    #     # 各デバイスがローカルでモデルをトレーニング
    #     for device in devices:
    #         device.local_train()

    #     # グローバルモデルの更新
    #     blockchain.aggregate_weights(devices)

    #     # クロスバリデーション評価
    #     cross_validation_evaluation(devices, blockchain, epoch)

    #     # 各デバイスの貢献度スコアを表示
    #     for device_id, scores in blockchain.contribution_scores.items():
    #         print(f"Device {device_id} contribution score at epoch {epoch+1}: {scores[-1][1]:.4f}")
    previous_scores = {device.device_id: 0 for device in devices}  # 初期スコアは0に設定

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        # 各デバイスがローカルでモデルをトレーニング
        for device in devices:
            device.local_train()

        # クロスバリデーション評価を先に実行し、各デバイスの貢献度スコアを取得
        cross_validation_evaluation(devices, blockchain, epoch)

        state = blockchain.aggregate_weights(devices)
        #グローバルモデルの評価
        avg_loss, accuracy =blockchain.evaluate_global_model()
        #ブロックを追加
        blockchain.add_block(state,avg_loss,accuracy)

        # 前回のスコアより高いデバイスのみを集約対象とする
        # selected_devices = []
        # selected_devices_id = []
        # for device in devices:
        #     current_score = blockchain.contribution_scores[device.device_id][-1][1]  # 最新のスコアを取得
        #     if current_score > previous_scores[device.device_id]:  # スコアが前回より高ければ選択
        #         selected_devices.append(device)
        #         selected_devices_id.append(device.device_id)
        #         previous_scores[device.device_id] = current_score  # 前回のスコアを更新

        # # グローバルモデルの更新（選ばれたデバイスのみ集約）
        # if selected_devices:
        #     state = blockchain.aggregate_weights(selected_devices)
        #     #グローバルモデルの評価
        #     avg_loss, accuracy =blockchain.evaluate_global_model()
        #     #ブロックを追加
        #     blockchain.add_block(state,avg_loss,accuracy)

        # else:
        #     print("select devices is NULL")

        # 各デバイスの貢献度スコアを表示
        for device_id, scores in blockchain.contribution_scores.items():
            print(f"Device {device_id} is aggregate, contribution score at epoch {epoch+1}: {scores[-1][1]:.4f}")
            # if device_id in selected_devices_id:
            #     print(f"Device {device_id} is aggregate, contribution score at epoch {epoch+1}: {scores[-1][1]:.4f}")
            # else:
            #     print(f"Device {device_id} is not aggregate, contribution score at epoch {epoch+1}: {scores[-1][1]:.4f}")

# 貢献度スコアをプロットする関数
def plot_contribution_scores(blockchain):
    plt.figure(figsize=(10, 6))
    for device_id, scores in blockchain.contribution_scores.items():
        rounds, scores = zip(*scores)
        plt.scatter(rounds, scores, label=f'Device {device_id}')  # 点プロット
    plt.xlabel("Round")
    plt.ylabel("Contribution Score")
    plt.title("Contribution Scores per Device over Rounds")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(os.path.join(SAVE_DIR, 'Contribution_Scores_per_Device_over_Rounds.png'))
    plt.close()



def plot_loss_and_accuracy(blockchain, devices:List[Device], save_dir):
    rounds = list(range(1, len(blockchain.global_model_loss) + 1))

    # グローバルモデルの Loss と Accuracy のプロット
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 第1軸: Loss
    ax1.scatter(rounds, blockchain.global_model_loss, color='blue', label="Global Model Loss", marker='o')
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Loss", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 第2軸: Accuracy
    ax2 = ax1.twinx()
    ax2.scatter(rounds, blockchain.global_model_accuracy, color='orange', label="Global Model Accuracy", marker='x')
    ax2.set_ylabel("Accuracy", color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # タイトルと凡例
    fig.suptitle("Global Model Loss and Accuracy per Round with Dual Axes")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # 保存
    plt.savefig(os.path.join(save_dir, 'Global_Model_Loss_and_Accuracy_with_Dual_Axes.png'))
    plt.close()

    # クライアントごとの Loss と Accuracy のプロット
    for device in devices:
        device_losses = device.loss
        device_accuracies = device.accuracy

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # 第1軸: Loss
        ax1.scatter(rounds, device_losses, color='blue', label=f"Device {device.device_id} Loss", marker='o')
        ax1.set_xlabel("Round")
        ax1.set_ylabel("Loss", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

        # 第2軸: Accuracy
        ax2 = ax1.twinx()
        ax2.scatter(rounds, device_accuracies, color='red', label=f"Device {device.device_id} Accuracy", marker='x')
        ax2.set_ylabel("Accuracy", color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # タイトルと凡例
        fig.suptitle(f"Device {device.device_id} Loss and Accuracy per Round with Dual Axes")
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")

        # 保存
        plt.savefig(os.path.join(save_dir, f'Device_{device.device_id}_Loss_and_Accuracy_with_Dual_Axes.png'))
        plt.close()



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
testDataSet_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=False)
blockchain = Blockchain(testDataSet_loader)

# 連合学習の実行
federated_learning_round(devices, blockchain, epochs=50)

# 貢献度スコアをプロット
plot_contribution_scores(blockchain)

# 貢献度スコアをファイルに保存
blockchain.save_contribution_scores(os.path.join(SAVE_CSV_DIR, 'contribution_scores.csv'))
# プロットの実行
plot_loss_and_accuracy(blockchain, devices, SAVE_DIR)