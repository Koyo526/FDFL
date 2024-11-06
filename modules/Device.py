import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier


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
