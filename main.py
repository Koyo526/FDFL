import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from modules.Blockchain import Blockchain
from modules.Device import Device
from modules.Evaluation import Eavluation
from typing import Any, Callable, List, Tuple

def federated_learning_round(devices:List[Device], blockchain:Blockchain, epochs=1):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        # 各デバイスがローカルでモデルをトレーニング
        for device in devices:
            device.local_train()

        # グローバルモデルの更新
        blockchain.aggregate_weights(devices)

        # クロスバリデーション評価
        Eavluation.cross_validation_evaluation(devices, blockchain)

        # 各デバイスの貢献度スコアを表示
        for block in blockchain.chain:
            print(f"Device {block['device_id']} contribution score: {block['contribution_score']:.4f}")

        # スコアのリセット
        blockchain.chain = []


# サンプルデータの生成
num_devices = 5
data_size = 100
num_features = 10
X, y = np.random.rand(num_devices * data_size, num_features), np.random.randint(0, 2, num_devices * data_size)

# 各デバイスにデータを分配
devices = []
for i in range(num_devices):
    data, labels = X[i * data_size:(i + 1) * data_size], y[i * data_size:(i + 1) * data_size]
    devices.append(Device(device_id=i, data=data, labels=labels))

# ブロックチェーンの初期化
blockchain = Blockchain()

# 連合学習の実行
federated_learning_round(devices, blockchain, epochs=3)
