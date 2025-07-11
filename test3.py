import numpy as np

def initialize_weights(shape):
    return np.random.rand(*shape)

# 分散データ貢献度 (Φ) を計算する関数
def calculate_contribution_impact(global_weights, local_weights, validation_data, prev_global_acc):
    """
    分散データ貢献度を計算
    """
    local_accuracy = evaluate_model(local_weights, validation_data)
    contribution = max(local_accuracy - prev_global_acc, 0)  # 式 (14)
    return contribution

# モデル精度評価
def evaluate_model(model_weights, validation_data):
    """
    ダミーのモデル精度評価関数
    """
    np.random.seed(int(np.sum(model_weights)))  # 同じ結果にならないようシードを変更
    return np.random.uniform(0.5, 0.9)  # 精度を0.5~0.9の間でランダム生成

def calculate_data_valuation(contributions, client_id, num_clients):
    if len(contributions) < num_clients:
        raise ValueError("Contributions list is shorter than the number of clients.")
    valuation = np.sum([contributions[i] for i in range(num_clients) if i != client_id]) / (num_clients - 1)
    return valuation

# 限界効用の計算
def calculate_marginal_utility(data_size, lambda_factor=1.0):
    """
    データの限界効用を計算 (式 19)
    """
    return lambda_factor * np.log(data_size)

# 重みの集約
def aggregate_weights(global_weights, local_gradients, valuations, contributions, use_valuation=True):
    """
    データ価値評価と限界効用に基づいて重みを集約
    """
    if use_valuation:
        weight_factors = valuations / np.sum(valuations)
    else:
        weight_factors = contributions / np.sum(contributions)
    
    updated_weights = global_weights - np.sum([w * grad for w, grad in zip(weight_factors, local_gradients)], axis=0)
    return updated_weights

def decentralized_data_aggregation(global_weights, local_weights_list, validation_data_list, data_sizes, rounds):
    num_clients = len(local_weights_list)
    contributions = [0] * num_clients  # 初期化

    for round_num in range(rounds):
        print(f"Round {round_num + 1}/{rounds}")

        # 各クライアントの貢献度を計算
        for client_id in range(num_clients):
            valuation = calculate_data_valuation(contributions, client_id, num_clients)
            contributions[client_id] = valuation

        # グローバルモデルの更新
        global_weights = np.mean(local_weights_list, axis=0)

        # 検証データでの評価（ダミー）
        val_acc = np.random.rand()
        util_acc = np.random.rand()
        print(f"Validation Accuracy (Valuation): {val_acc:.4f}, Validation Accuracy (Utility): {util_acc:.4f}")

    return global_weights

# 実行部分
if __name__ == "__main__":
    num_clients = 3
    model_shape = (10,)  # 簡易的な重みベクトル
    rounds = 5

    # 初期化
    global_weights = initialize_weights(model_shape)
    local_weights_list = [initialize_weights(model_shape) for _ in range(num_clients)]
    validation_data_list = [None] * num_clients  # ダミーデータ
    data_sizes = [np.random.randint(100, 500) for _ in range(num_clients)]  # 各クライアントのデータ量

    # アルゴリズム実行
    final_weights = decentralized_data_aggregation(global_weights, local_weights_list, validation_data_list, data_sizes, rounds)
    print("\n最終的なグローバルモデルの重み:", final_weights)