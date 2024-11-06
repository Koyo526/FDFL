import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier

# デバイスクラスの定義
class Device:
    def __init__(self, device_id, data, labels):
        self.device_id = device_id
        self.data = data
        self.labels = labels
        self.model = SGDClassifier()  # 単純な線形モデルを使用
        self.local_weights = None

    def local_train(self):
        # ローカルデータでモデルをトレーニングし、重みを更新
        self.model.fit(self.data, self.labels)
        self.local_weights = self.model.coef_
        return self.local_weights

    def evaluate(self, global_model):
        # グローバルモデルで評価して貢献度スコアを計算
        predictions = global_model.predict(self.data)
        return accuracy_score(self.labels, predictions)

# ブロックチェーンクラスの定義
class Blockchain:
    def __init__(self, num_features):
        self.chain = []
        self.global_model = SGDClassifier()  # グローバルモデル
        self.global_weights = None
        # デバイスの特徴量数に基づいたダミーデータで初期化しておく
        self.global_model.fit(np.random.rand(10, num_features), np.random.randint(0, 2, 10))

    def aggregate_weights(self, devices):
        # 各デバイスの重みを集約（単純平均）
        weights = np.mean([device.local_weights for device in devices], axis=0)
        self.global_weights = weights
        self.global_model.coef_ = self.global_weights
        return self.global_model

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
                score = other_device.evaluate(blockchain.global_model)
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
blockchain = Blockchain(num_features=num_features)

# 連合学習の実行
federated_learning_round(devices, blockchain, epochs=10)
