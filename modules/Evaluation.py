import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier


class Eavluation:
    def cross_validation_evaluation(devices, blockchain):
        for device in devices:
            score_sum = 0
            for other_device in devices:
                if other_device.device_id != device.device_id:
                    score = other_device.evaluate(blockchain.global_model)
                    score_sum += score
            contribution_score = score_sum / (len(devices) - 1)
            blockchain.record_contribution(device, contribution_score)
