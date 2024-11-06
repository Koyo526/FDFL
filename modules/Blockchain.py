import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier



class Blockchain:
    def __init__(self):
        self.chain = []
        self.global_model = SGDClassifier()  # グローバルモデル
        self.global_weights = None
        self.global_model.fit(np.random.rand(10, 5), np.random.randint(0, 2, 10))

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
