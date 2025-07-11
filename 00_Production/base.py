from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset
from torchvision.datasets import CIFAR10
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

import os
import matplotlib.pyplot as plt
import datetime
import csv
from scipy import stats

import pandas as pd
import sys
import scienceplots

# Define a directory to save the plots
now = datetime.datetime.now()
current_time = now.strftime("%Y-%m-%d-%H-%M")



# Initialize lists to store parameter changes
global_params_history = []
local_params_history = []
clinet_accuracy_history = []
clinet_loss_history = []
clinet_all_accuracy_history = []
clinet_all_loss_history = []
clinet_token_history = []
client_contoribution_history = []
client_trainloader = []
client_valloader = []
epochs_history = []
random_times_history = []
BASE_TOKEN = 10
DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
cid_map: Dict[str, int] = {}   # Ray がくれたランダム cid → 連番
next_idx: int = 0
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)


def stratified_split_indices(dataset, train_size=0.8, random_state=42):
    """
    層化サンプリングを用いてデータセットを分割する関数。
    Args:
        dataset: データセット (torchvision.datasets)
        train_size: トレーニングデータの割合
        random_state: 乱数シード
    Returns:
        train_indices: トレーニングデータのインデックス
        val_indices: 検証データのインデックス
    """
    labels = np.array([dataset[i][1] for i in range(len(dataset))])  # ラベルを取得
    strat_split = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
    train_indices, val_indices = next(strat_split.split(np.zeros(len(labels)), labels))
    return train_indices, val_indices

def load_datasets_stratified(num_clients: int,
                             client_data_sizes: List[float]):
    """
    CIFAR-10 を層化分割し，各クライアントへ均等（または指定割合）に
    train/val データを配布する。
    """
    # --- ① CIFAR-10 読込 --------------------------------------------------
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5),
                              (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10("./dataset", train=True,
                       download=True, transform=transform)
    testset  = CIFAR10("./dataset", train=False,
                       download=True, transform=transform)
    n_total = len(trainset)          # 50_000

    # --- ② クライアントごとのサンプル数計算 ------------------------------
    if len(client_data_sizes) != num_clients:
        raise ValueError("Length of client_data_sizes must equal num_clients")

    # 浮動小数点→整数（切り捨て）で一旦計算
    split_sizes = [int(n_total * p) for p in client_data_sizes]

    # 端数を補正
    remainder = n_total - sum(split_sizes)
    for i in range(remainder):
        split_sizes[i] += 1          # 先頭クライアントから +1 ずつ

    assert sum(split_sizes) == n_total, "Size mismatch after correction"

    print(f"Per-client sizes: {split_sizes}")

    # --- ③ 各クライアント用インデックスを層化抽出 ------------------------
    labels = np.array([trainset[i][1] for i in range(n_total)])
    remaining_idx = np.arange(n_total)
    client_indices_list = []

    rng = np.random.default_rng(42)
    for size in split_sizes:
        if size == len(remaining_idx):
            # ← 最後のクライアント。残りを全部渡す
            client_indices = remaining_idx.copy()
        else:
            sss = StratifiedShuffleSplit(
                n_splits=1,
                train_size=size,
                random_state=int(rng.integers(1e6))
            )
            idx, _ = next(sss.split(remaining_idx,
                                    labels[remaining_idx]))
            client_indices = remaining_idx[idx]

        client_indices_list.append(client_indices)
        remaining_idx = np.setdiff1d(remaining_idx, client_indices)

    # --- ④ train/val Loader 作成 ------------------------------------------
    trainloaders, valloaders = [], []
    for client_idx, indices in enumerate(client_indices_list):
        subset = Subset(trainset, indices)
        train_idx, val_idx = stratified_split_indices(subset, train_size=0.9)
        ds_train, ds_val = Subset(subset, train_idx), Subset(subset, val_idx)
        trainloaders.append(DataLoader(ds_train, batch_size=32, shuffle=True))
        valloaders.append(DataLoader(ds_val,   batch_size=32))

        print(f"Client {client_idx+1}: "
              f"train={len(ds_train)}, val={len(ds_val)}")

    testloader = DataLoader(testset, batch_size=32)
    return trainloaders, valloaders, testloader


def load_uniform_datasets(num_clients:int):
    # CIFAR-10データのダウンロードと前処理
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10("./dataset", train=True, download=True, transform=transform)
    testset = CIFAR10("./dataset", train=False, download=True, transform=transform)

    print(len(trainset), len(testset))
    remaining_indices = np.arange(len(trainset))
    labels = np.array([trainset[i][1] for i in range(len(trainset))])
    client_indices_list = []
    size = len(trainset) // num_clients
    split = StratifiedShuffleSplit(n_splits=1, train_size=size, random_state=42)
    train_indices, _ = next(split.split(remaining_indices, labels[remaining_indices]))
    for _ in range(num_clients):
        client_indices_list.append(remaining_indices[train_indices])
    print([len(client_indices) for client_indices in client_indices_list])

    trainloaders = []
    valloaders = []

    # 各クライアントのデータをtrain/valに分割
    for i, client_indices in enumerate(client_indices_list):
        client_dataset = Subset(trainset, client_indices)
        train_indices, val_indices = stratified_split_indices(client_dataset, train_size=0.9)
        ds_train = Subset(client_dataset, train_indices)
        ds_val = Subset(client_dataset, val_indices)
        trainloaders.append(DataLoader(ds_train, batch_size=32, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=32))


        # クラスごとのサンプル数を表示
        train_labels = [ds_train[i][1] for i in range(len(ds_train))]
        val_labels = [ds_val[i][1] for i in range(len(ds_val))]
        train_class_counts = Counter(train_labels)
        val_class_counts = Counter(val_labels)

        print(f"Client {i+1}:")
        print(f"  Train Size = {len(ds_train)} | Class Counts: {dict(train_class_counts)}")
        print(f"  Val Size   = {len(ds_val)}   | Class Counts: {dict(val_class_counts)}")


    # テストデータローダ
    testloader = DataLoader(testset, batch_size=32)

    # 結果の表示
    for i, (trainloader, valloader) in enumerate(zip(trainloaders, valloaders), 1):
        print(f"Client {i}: Train Size = {len(trainloader.dataset)}, Val Size = {len(valloader.dataset)}")
    
    print(f"Total Test Size = {len(testloader.dataset)}")
    return trainloaders, valloaders, testloader

def load_datasets(num_clients: int, client_data_sizes: list) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    # Download and transform CIFAR-10 (train and test)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10("./dataset", train=True, download=True, transform=transform)
    testset = CIFAR10("./dataset", train=False, download=True, transform=transform)
    print(len(trainset), len(testset))
    global NUM_CLIENTS
    if sum(client_data_sizes) != 1.0 or len(client_data_sizes) != num_clients:
        raise ValueError("Sum of client_data_sizes must be 1.0")
    
    client_data_split_sizes = [int(len(trainset) * size) for size in client_data_sizes]
    if sum(client_data_split_sizes) != len(trainset):
        raise ValueError("Sum of client_data_split_sizes must be equal to the length of trainset")




    # Split training set into `num_clients` partitions to simulate different local datasets
    datasets = random_split(trainset, client_data_split_sizes, torch.Generator().manual_seed(42))

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds)//10
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(DataLoader(ds_train, batch_size=32, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=32))
        print(len(trainloaders), len(valloaders))
    testloader = DataLoader(testset, batch_size=32)
    
    # データの量を表示
    for i, (trainloader, valloader) in enumerate(zip(trainloaders, valloaders), 1):
        print(f"Client {i}: Train Size = {len(trainloader.dataset)}, Val Size = {len(valloader.dataset)}")
        global client_trainloader, client_valloader
        client_trainloader.append((len(trainloader.dataset), trainloader.dataset))
        client_valloader.append((len(valloader.dataset), valloader.dataset))
    
    print(f"Total Test Size = {len(testloader.dataset)}")
    print(f"Average Test Size per Client = {len(testloader.dataset) // num_clients}")
    return trainloaders, valloaders, testloader


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.drop1 = nn.Dropout(0.25)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.drop2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(256, 512)
        self.drop3 = nn.Dropout(0.45)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.drop1(x)
        x = self.pool3(x)
        x = self.pool4(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = self.drop2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop3(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def train(net, trainloader, epochs,learning_rate) -> None:
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    # optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss.item()  # 修正: lossを値として追加
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        print(f"[Client {self.cid}] fit, config: {config}")
        set_parameters(self.net, parameters)

        # Before training, save the initial local parameters
        local_params_before = get_parameters(self.net)
        print(f"[Client {self.cid}] epoch: {EPOCHS}")
        train(self.net, self.trainloader, epochs=EPOCHS, learning_rate=config["lr"])  
        

        global_params_after = get_parameters(self.net)

        # Save the local parameters before training for this client
        local_params_history.append(local_params_before)
        
        # Save the global parameters after training
        global_params_history.append(global_params_after)
        
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

def client_fn(cid) -> FlowerClient:
    global next_idx, cid_map
    if cid not in cid_map:
        cid_map[cid] = next_idx
        next_idx += 1
    idx = cid_map[cid]
    net = Net().to(DEVICE)
    trainloader = trainloaders[int(idx)]
    valloader = valloaders[int(idx)]
    return FlowerClient(str(idx), net, trainloader, valloader)

def evaluate_model_on_own_data(model, testloader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


def modified_z_score(data):
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    return 0.6745 * (data - median) / mad

class FedCustom(fl.server.strategy.Strategy):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.global_loss_history = []  # グローバル損失の履歴
        self.global_accuracy_history = []  # グローバル精度の履歴
        self.client_contribution_scores = {}  # クライアントごとの貢献度スコア
        self.global_learning_rate = 0.5  # グローバルモデル更新の学習率
        self.global_model = Net().to(DEVICE)  # グローバルモデル
        self.valloaders = valloaders  # クライアントのテストデータ
        self.global_accuracy_history_for_client = []
        self.global_loss_history_for_client = []
        self.client_loss_history_for_testloader = []
        self.client_accuracy_history_for_testloader = []
        self.client_accuracy_history_by_client = []
        self.client_loss_history_by_client = []
        self.cid2idx: Dict[str, int] = {}   # ← 追加

    def __repr__(self) -> str:
        return "FedCustom"
    
    def _idx(self, cid: str) -> int:
        if cid not in self.cid2idx:
            self.cid2idx[cid] = len(self.cid2idx)
        return self.cid2idx[cid]

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        net = Net()
        ndarrays = get_parameters(net)
        self.current_round_global_params = fl.common.ndarrays_to_parameters(ndarrays)
        return self.current_round_global_params

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # 保存: 各クライアントに現在のグローバルパラメータを保存
        global global_params_history, local_params_history
        for client in clients:
            # client.cid は文字列なので整数に変換
            try:
                cid_int = int(client.cid)
            except ValueError:
                cid_int = client.cid  # 変換できない場合はそのまま
            local_params_history.append((cid_int, self.current_round_global_params))

        # Create custom configs
        # TODO: 学習率の設定は要検討(シミュレーションごとに任意に変更できると良い)
        global LEARNING_RATE
        n_clients = len(clients)
        half_clients = n_clients // 2
        standard_config = {"lr": LEARNING_RATE}
        higher_lr_config = {"lr": LEARNING_RATE}
        fit_configurations = []
        for idx, client in enumerate(clients):
            if idx < half_clients:
                fit_configurations.append((client, FitIns(parameters, standard_config)))
            else:
                fit_configurations.append(
                    (client, FitIns(parameters, higher_lr_config))
                )
        return fit_configurations
    
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
        config = {}
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics_aggregated = {}
        return loss_aggregated, metrics_aggregated
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average with adjustments."""
        global global_params_history
        global clinet_accuracy_history
        global clinet_loss_history
        global clinet_all_accuracy_history
        global clinet_all_loss_history
        global clinet_token_history
        global client_contoribution_history
        global epochs_history
        global EPOCHS
        global random_times_history
        

        

        # 他のノードのモデルを自分のテストデータで評価
        #ここを関数化したい
        epochs_history.append(EPOCHS)

        net = Net().to(DEVICE)
        own_test_results = []
        clinet_accuracy = [[] for _ in range(NUM_CLIENTS)]
        clinet_loss = [[] for _ in range(NUM_CLIENTS)]
        tmp_accuracy_global = []
        tmp_loss_global = []
        tmp_client_accuracy = []
        tmp_client_loss = []
        rng = np.random.default_rng()
        # random_times = rng.normal(loc=1, scale=0.5)
        # random_times = rng.normal(loc=0, scale=0.5)
        random_times = rng.normal(loc=0.6, scale=0.1)
        random_times_history.append(random_times)

        for client_id, valloader in enumerate(self.valloaders):
            tmp_accuracy = []
            tmp_loss = []
            tmp_accuracy_for_client = [0 for _ in range(NUM_CLIENTS)]
            tmp_loss_for_client = [0 for _ in range(NUM_CLIENTS)]
            
            set_parameters(net, parameters_to_ndarrays(self.current_round_global_params))
            GLoss, GAccuracy = evaluate_model_on_own_data(net , valloader)
            tmp_accuracy_global.append(GAccuracy)
            tmp_loss_global.append(GLoss)

            print(f"Client {client_id} Global Model Accuracy: {GAccuracy:.4f}, Global Model Loss: {GLoss:.4f}")
            for client_proxy, fit_res in results:
                idx = self._idx(client_proxy.cid)
                if idx != int(client_id):
                    model = Net().to(DEVICE)
                    # 悪意のあるランダムクライアントの設定　'NUM_CLIENTS-1' に設定することで動作可能
                    if idx == NUM_CLIENTS:
                        print(f"Client {client_id} -> Client {idx}  Random times: {random_times} (Random param)")
                        randam_params = parameters_to_ndarrays(self.current_round_global_params)
                        # randam_params = [param * random_times for param in randam_params]
                        randam_params = [param + (param * random_times) for param in randam_params]
                        set_parameters(model, randam_params)
                    else:
                        set_parameters(model, parameters_to_ndarrays(fit_res.parameters))
                    loss, accuracy = evaluate_model_on_own_data(model, valloader)
                    tmp_loss_for_client[idx] = loss
                    tmp_accuracy_for_client[idx] = accuracy
                    if loss < GLoss:
                        # クライアントごとのLossとAccuracyを提出する
                        clinet_accuracy[idx].append(accuracy)
                        clinet_loss[idx].append(loss)
                        tmp_accuracy.append(accuracy)
                        tmp_loss.append(loss)
                        own_test_results.append((idx, loss, accuracy))
                        print(f"Client {client_id} -> Client {idx}  Accuracy: {accuracy:.4f}, Loss: {loss:.4f}  (Submitted)")
                    else:
                        print(f"Client {client_id} -> Client {idx}  Accuracy: {accuracy:.4f}, Loss: {loss:.4f} (Not submitted)")
                        tmp_accuracy.append(None)
                        tmp_loss.append(None)
                else:
                    model = Net().to(DEVICE)
                    set_parameters(model, parameters_to_ndarrays(fit_res.parameters))
                    loss, accuracy = evaluate_model_on_own_data(model, valloader)
                    tmp_accuracy_for_client[idx] = accuracy
                    tmp_loss_for_client[idx] = loss
            #ここにクライアントが評価した他のクライアントの値を保存
            tmp_client_accuracy.append(tmp_accuracy_for_client)
            tmp_client_loss.append(tmp_loss_for_client)
            clinet_all_accuracy_history.append(tmp_accuracy)
            clinet_all_loss_history.append(tmp_loss)
        #クライアントによる他のクライアントの評価
        self.client_accuracy_history_by_client.append(tmp_client_accuracy)
        self.client_loss_history_by_client.append(tmp_client_loss)
        #　クライアントによるグローバルモデルの評価
        self.global_accuracy_history_for_client.append(tmp_accuracy_global)
        self.global_loss_history_for_client.append(tmp_loss_global)
        tmp_accuracy_client = []
        tmp_loss_client = []
        for client_proxy, fit_res in results:
            client_id = self._idx(client_proxy.cid)
            model = Net().to(DEVICE)
            set_parameters(model, parameters_to_ndarrays(fit_res.parameters))
            client_loss_testloader, client_accuracy_testloader = test(model, testloader)
            tmp_accuracy_client.append(client_accuracy_testloader)
            tmp_loss_client.append(client_loss_testloader)
        self.client_accuracy_history_for_testloader.append(tmp_accuracy_client)
        self.client_loss_history_for_testloader.append(tmp_loss_client)
        
        
        # クライアントが評価したLossとAccuracyを平均化する
        accuracy_list = []
        loss_list = []
        for client_id in range(NUM_CLIENTS):
            # clinet_accuracyとclinet_lossの信頼区間を計算
            accuracies = np.array(clinet_accuracy[client_id])
            losses = np.array(clinet_loss[client_id])

            
            # # Modified Zスコアを使用して外れ値を検出
            # modified_z_scores_accuracy = modified_z_score(accuracies)
            # modified_z_scores_loss = modified_z_score(losses)
            # threshold = 3.5  # Modified Zスコアの閾値を設定

            # # 外れ値を省く
            # accuracies_filtered = accuracies[modified_z_scores_accuracy < threshold]
            # losses_filtered = losses[modified_z_scores_loss < threshold]

            

            #TODO: 2つ以上の評価がある場合のみを対象とする
            # if len(accuracies) > 1:
            if len(accuracies) > NUM_CLIENTS // 2:
                accuracy_mean = np.mean(accuracies)
                accuracy_se = stats.sem(accuracies)
                accuracy_ci = stats.t.interval(0.95, len(accuracies) - 1, loc=accuracy_mean, scale=accuracy_se)
                loss_mean = np.mean(losses)
                loss_se = stats.sem(losses)
                loss_ci = stats.t.interval(0.95, len(losses) - 1, loc=loss_mean, scale=loss_se)

                # # 信頼区間内の値をフィルタリングして平均化
                # accuracy_within_ci = accuracies_filtered[(accuracies_filtered >= accuracy_ci[0]) & (accuracies_filtered <= accuracy_ci[1])]
                # if len(accuracy_within_ci) > 0:
                #     accuracy_list.append(np.mean(accuracy_within_ci))
                # else:
                #     accuracy_list.append(np.mean(accuracies_filtered))

                # # 信頼区間内の値をフィルタリングして平均化
                # loss_within_ci = losses_filtered[(losses_filtered >= loss_ci[0]) & (losses_filtered <= loss_ci[1])]
                # if len(loss_within_ci) > 0:
                #     loss_list.append(np.mean(loss_within_ci))
                # else:
                #     loss_list.append(np.mean(losses_filtered))

                # 信頼区間内の値をフィルタリングして平均化
                accuracy_within_ci = accuracies[(accuracies >= accuracy_ci[0]) & (accuracies <= accuracy_ci[1])]
                if len(accuracy_within_ci) > 0:
                    accuracy_list.append(np.mean(accuracy_within_ci))
                else:
                    accuracy_list.append(accuracy_mean)

                # 信頼区間内の値をフィルタリングして平均化
                loss_within_ci = losses[(losses >= loss_ci[0]) & (losses <= loss_ci[1])]
                if len(loss_within_ci) > 0:
                    loss_list.append(np.mean(loss_within_ci))
                else:
                    loss_list.append(loss_mean)

                print(f"Client {client_id} Accuracy: {accuracy_mean:.4f}, Loss: {loss_mean:.4f}")
                
            # elif len(accuracies) == 1:
            #     accuracy_list.append(np.mean(accuracies))
            #     loss_list.append(np.mean(losses))
            else:
                accuracy_list.append(None)
                loss_list.append(None)
        clinet_accuracy_history.append(accuracy_list)
        clinet_loss_history.append(loss_list)


        # 重みを計算（Lossが小さいほど重みが大きくなるように逆数を取る）
        
        valid_losses = []
        none_count = 0
        for loss in loss_list:
            if loss is None:
                valid_losses.append(0)
                none_count += 1
            else:
                valid_losses.append(loss)
        weights = [1.0 / loss if loss > 0 else 0 for loss in valid_losses]
        total_weight = sum(weights)

        #TODO:Noneが多い場合の処理を考える
        if total_weight > 0:
            if none_count > (NUM_CLIENTS//2)+1:
                print(f"None count :{none_count}")
                if EPOCHS < 20:
                    EPOCHS += 1
                print(f"Next Round Epochs: {EPOCHS}")
        
        # if total_weight > 0  and none_count < NUM_CLIENTS//2+1:
        #     print(f"None count :{none_count}(Normal)")
        #     print(f"Next Round Epochs: {EPOCHS}")
       

            normalized_weights = [weight / total_weight if weight > 0 else 0 for weight in weights]
            for idx, weight in enumerate(normalized_weights):
                print(f"Client {idx + 1} Weight: {weight:.4f}")
            # Calculate and store client contributions
            for client_proxy, _ in results:
                client_id = self._idx(client_proxy.cid)
                if client_id not in self.client_contribution_scores:
                    self.client_contribution_scores[client_id] = []
                self.client_contribution_scores[client_id].append(normalized_weights[client_id - 1])
            client_contoribution_history.append(normalized_weights)
            # クライアントのパラメータを取得
            client_parameters = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]

            # グローバルパラメータを重み付け平均化
            # for params in zip(*client_parameters):
                # weighted_params = [weight * param for weight, param in zip(normalized_weights, params)]
                # Extract parameters and client contributions
            weights_results = [(params,weight) for weight, params in zip(normalized_weights, client_parameters)]
            aggregated_params = aggregate(weights_results)
            # Save the aggregated global parameters
            parameters_aggregated = ndarrays_to_parameters(aggregated_params)
            global_params_history.append(parameters_aggregated)

            # Update the current round global parameters
            self.current_round_global_params = parameters_aggregated

            # グローバルモデルに新しいパラメータを設定
            net = Net().to(DEVICE)
            set_parameters(net, aggregated_params)
            global_loss, global_accuracy = test(net, testloader)
            self.global_loss_history.append(global_loss)
            self.global_accuracy_history.append(global_accuracy)

            # Print evaluation results
            print(
                f"Round {server_round}: Global Accuracy = {global_accuracy:.4f}, Global Loss = {global_loss:.4f}"
            )

            # Tokenの更新
            client_token = [0 for _ in range(NUM_CLIENTS)]
            for idx, weight in enumerate(normalized_weights):
                client_token[idx] += min(BASE_TOKEN * weight, BASE_TOKEN/2)
                print(f"Client {idx + 1} Token: {client_token[idx]}", end=" ")
            print()
            clinet_token_history.append(client_token)

        # elif total_weight >0 and  none_count >= NUM_CLIENTS//2+1:
        #     # epoch数をスケーリング
        #     if EPOCHS < 20:
        #         EPOCHS += 1
        #     print(f"Next Round Epochs: {EPOCHS}")
        #     print(f"None count :{none_count}(Check client's loss)")
        #     select_client_weight = []
        #     for idx in range(NUM_CLIENTS):
        #         if len(clinet_loss[idx]) == NUM_CLIENTS-1:
        #             select_client_weight.append(weights[idx])
        #         else:
        #             select_client_weight.append(0)
        #     total_select_weight = sum(select_client_weight)
        #     if total_select_weight > 0:
        #         print(f"None count :{none_count}(approved all other clients)")
        #         normalized_weights = [weight / total_select_weight if weight > 0 else 0 for weight in select_client_weight]
        #         for idx, weight in enumerate(normalized_weights):
        #             print(f"Client {idx + 1} Weight: {weight:.4f}")
        #         # Calculate and store client contributions
        #         for client_proxy, _ in results:
        #             client_id = int(client_proxy.cid)
        #             if client_id not in self.client_contribution_scores:
        #                 self.client_contribution_scores[client_id] = []
        #             self.client_contribution_scores[client_id].append(normalized_weights[client_id - 1])
        #         client_contoribution_history.append(normalized_weights)
        #         # クライアントのパラメータを取得
        #         client_parameters = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]

        #         # グローバルパラメータを重み付け平均化
        #         # for params in zip(*client_parameters):
        #             # weighted_params = [weight * param for weight, param in zip(normalized_weights, params)]
        #             # Extract parameters and client contributions
        #         weights_results = [(params,weight) for weight, params in zip(normalized_weights, client_parameters)]
        #         aggregated_params = aggregate(weights_results)
        #         # Save the aggregated global parameters
        #         parameters_aggregated = ndarrays_to_parameters(aggregated_params)
        #         global_params_history.append(parameters_aggregated)

        #         # Update the current round global parameters
        #         self.current_round_global_params = parameters_aggregated

        #         # グローバルモデルに新しいパラメータを設定
        #         net = Net().to(DEVICE)
        #         set_parameters(net, aggregated_params)
        #         global_loss, global_accuracy = test(net, testloader)
        #         self.global_loss_history.append(global_loss)
        #         self.global_accuracy_history.append(global_accuracy)

        #         # Print evaluation results
        #         print(
        #             f"Round {server_round}: Global Accuracy = {global_accuracy:.4f}, Global Loss = {global_loss:.4f}"
        #         )

        #         # Tokenの更新
        #         client_token = [0 for _ in range(NUM_CLIENTS)]
        #         for idx, weight in enumerate(normalized_weights):
        #             client_token[idx] += BASE_TOKEN * weight
        #             print(f"Client {idx + 1} Token: {client_token[idx]}", end=" ")
        #         print()
        #         clinet_token_history.append(client_token)
        #     else:
        #         print(f"None count :{none_count}(Not approved all other clients)")
        #         clinet_token_history.append([0 for _ in range(NUM_CLIENTS)])
        #         for idx, token in enumerate([0 for _ in range(NUM_CLIENTS)]):
        #             print(f"Client {idx + 1} Token: {token}", end=" ")
        #         print()
        #         client_contoribution_history.append([0 for _ in range(NUM_CLIENTS)])
        #         global_loss, global_accuracy = test(net, testloader)
        #         self.global_loss_history.append(global_loss)
        #         self.global_accuracy_history.append(global_accuracy)
        #         # Print evaluation results
        #         print(
        #             f"Round {server_round}: Global Accuracy = {global_accuracy:.4f}, Global Loss = {global_loss:.4f}"
        #         )
                
        #         parameters_aggregated = self.current_round_global_params
            
        else:
            print(f"None count :{none_count}")
            clinet_token_history.append([0 for _ in range(NUM_CLIENTS)])
            for idx, token in enumerate([0 for _ in range(NUM_CLIENTS)]):
                print(f"Client {idx + 1} Token: {token}", end=" ")
            print()
            client_contoribution_history.append([0 for _ in range(NUM_CLIENTS)])
            global_loss, global_accuracy = test(net, testloader)
            self.global_loss_history.append(global_loss)
            self.global_accuracy_history.append(global_accuracy)
            # Print evaluation results
            print(
                f"Round {server_round}: Global Accuracy = {global_accuracy:.4f}, Global Loss = {global_loss:.4f}"
            )
            # epoch数をスケーリング
            if EPOCHS < 20:
                EPOCHS += 1
            print(f"Next Round Epochs: {EPOCHS}")
            
            parameters_aggregated = self.current_round_global_params
        

        for client_id in range(NUM_CLIENTS):
            total_token = 0
            for token in clinet_token_history:
                total_token += token[client_id]
            print(f"[Round {server_round}] Client {client_id + 1} Total Token: {total_token}")
        return parameters_aggregated, {}
    
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""
        # Let's assume we won't perform the global model evaluation on the server side.
        return None

    
    def plot_global_metrics(self):
        """Plot global loss and accuracy."""
        rounds = range(0, len(self.global_loss_history))
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # 第1軸: Loss
        ax1.scatter(rounds, self.global_loss_history, color='blue', label="Global Loss", marker='o')
        ax1.set_xlabel("Round")
        ax1.set_ylabel("Loss", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

        # 第2軸: Accuracy
        ax2 = ax1.twinx()
        ax2.scatter(rounds, self.global_accuracy_history, color='red',label="Global Accuracy", marker='x')
        ax2.set_ylabel("Accuracy", color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # タイトルと凡例
        fig.suptitle("Global Metrics Over Rounds")
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
        plt.grid()
        plt.savefig(os.path.join(SAVE_DIR, "global_metrics.png"))
        plt.close()

    def save_contribution_scores(self):
        """Save client contribution scores to a CSV file."""
        csv_path = os.path.join(SAVE_DIR, "contribution_scores.csv")
        with open(csv_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Round", "Client ID", "Contribution Score"])
            for client_id, scores in self.client_contribution_scores.items():
                for round_num, score in enumerate(scores, start=1):
                    writer.writerow([round_num, client_id, score])


    def plot_client_performance(self, clinet_accuracy_history, clinet_loss_history):
        num_rounds = len(clinet_accuracy_history)
        num_clients = len(clinet_accuracy_history[0])
        cmap   = plt.cm.get_cmap("tab20", num_clients)   # 20色スケールをクライアント数で間引く
        colors = [cmap(i) for i in range(num_clients)]          

        # Accuracy plot　平均化したクライアントのAccuracyとグローバルモデルのAccuracyをプロット
        plt.figure(figsize=(12, 6))
        for client_idx in range(num_clients):
            accuracies = [clinet_accuracy_history[round_idx][client_idx] for round_idx in range(num_rounds)]
            plt.scatter(range(num_rounds), accuracies, label=f'Client {client_idx + 1}')
        plt.scatter(range(num_rounds), self.global_accuracy_history,label="Global Accuracy", marker='x')
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.title('Client Accuracy Over Rounds')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(SAVE_DIR, "clinet_accuracy_history.png"))
        plt.close()

        # Loss plot　平均化したクライアントのLossとグローバルのLossをプロット
        plt.figure(figsize=(12, 6))
        for client_idx in range(num_clients):
            losses = [clinet_loss_history[round_idx][client_idx] for round_idx in range(num_rounds)]
            plt.scatter(range(num_rounds), losses, label=f'Client {client_idx + 1}')
        plt.scatter(range(num_rounds), self.global_loss_history, label="Global Loss", marker='x')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.title('Client Loss Over Rounds')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(SAVE_DIR, "clinet_loss_history.png"))
        plt.close()


        # Accuracy plot Testloaderで評価したクライアントのAccuracyとグローバルモデルのAccuracyをプロット
        plt.figure(figsize=(12, 6))
        for client_idx in range(num_clients):
            accuracies = [self.client_accuracy_history_for_testloader[round_idx][client_idx] for round_idx in range(num_rounds)]
            plt.scatter(range(num_rounds), accuracies, label=f'Client {client_idx + 1}')
        plt.scatter(range(num_rounds), self.global_accuracy_history,label="Global Accuracy", marker='x')
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.title('Global Accuracy Over Rounds By Testloader')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(SAVE_DIR, "clinet_accuracy_history_by_client_and_testloder.png"))
        plt.close()

        # Loss plot　Testloaderで評価したクライアントのAccuracyとグローバルモデルのLossをプロット
        plt.figure(figsize=(12, 6))
        for client_idx in range(num_clients):
            losses = [self.client_loss_history_for_testloader[round_idx][client_idx] for round_idx in range(num_rounds)]
            plt.scatter(range(num_rounds), losses, label=f'Client {client_idx + 1}')
        plt.scatter(range(num_rounds), self.global_loss_history, label="Global Loss", marker='x')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.title('global Loss Over Rounds By Testloader')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(SAVE_DIR, "global_loss_history_by_client_and_testloder.png"))
        plt.close()


        # self.client_accuracy_history_by_client と self.client_loss_history_by_client のプロット
        for client_idx in range(num_clients):
            plt.figure(figsize=(12, 6))
            for client_id in range(num_clients):
                accuracies = [self.client_accuracy_history_by_client[round_idx][client_idx][client_id] for round_idx in range(num_rounds)]
                plt.scatter(range(num_rounds), accuracies, label=f'Client {client_idx + 1} -> Client {client_id + 1}')
            plt.xlabel('Round')
            plt.ylabel('Accuracy')
            plt.title(f'Client Accuracy Over Rounds by Client{client_idx + 1}')
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(SAVE_DIR, f"clinet_accuracy_history_by_client{client_idx + 1}.png"))
            plt.close()

        for client_idx in range(num_clients):
            plt.figure(figsize=(12, 6))
            for client_id in range(num_clients):
                losses = [self.client_loss_history_by_client[round_idx][client_idx][client_id] for round_idx in range(num_rounds)]
                plt.scatter(range(num_rounds), losses, label=f'Client {client_idx + 1} -> Client {client_id + 1}')
            plt.xlabel('Round')
            plt.ylabel('loss')
            plt.title(f'Client Loss Over Rounds by Client{client_idx + 1}')
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(SAVE_DIR, f"clinet_Loss_history_by_client{client_idx + 1}.png"))
            plt.close()


        #一枚にまとめる
        plt.figure(figsize=(12, 6))
        for client_idx in range(num_clients):
            for client_id in range(num_clients):
                color = colors[client_id % len(colors)]
                accuracies = [self.client_accuracy_history_by_client[round_idx][client_idx][client_id] for round_idx in range(num_rounds)]
                plt.scatter(range(num_rounds), accuracies, label=f'Accuracy: Client {client_id + 1}', color=color)
                # plt.scatter(range(num_rounds), accuracies, label=f'Accuracy: Client {client_id + 1}')
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.title('Client Accuracy Over Rounds')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(SAVE_DIR, f"clinet_accuracy_history_by_client.png"))
        plt.close()

        #一枚にまとめる
        plt.figure(figsize=(12, 6))
        for client_idx in range(num_clients):
            for client_id in range(num_clients):
                color = colors[client_id % len(colors)]
                losses = [self.client_loss_history_by_client[round_idx][client_idx][client_id] for round_idx in range(num_rounds)]
                plt.scatter(range(num_rounds), losses,label=f'Loss": Client {client_id + 1}',color=color)
                # plt.scatter(range(num_rounds), losses,label=f'Loss": Client {client_id + 1}')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.title('Client Loss Over Rounds')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(SAVE_DIR, f"clinet_Loss_history_by_client.png"))
        plt.close()

        # Accuracy plot クライアントの評価とテストデータの評価をプロット
        plt.figure(figsize=(12, 6))
        for client_idx in range(num_clients):
            accuracies = [self.global_accuracy_history_for_client[round_idx][client_idx] for round_idx in range(num_rounds)]
            plt.scatter(range(num_rounds), accuracies, label=f'Client {client_idx + 1}')
        plt.scatter(range(num_rounds), self.global_accuracy_history,label="Global Accuracy", marker='x')
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.title('Global Accuracy Over Rounds')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(SAVE_DIR, "global_accuracy_history_by_client_and_testloder.png"))
        plt.close()

        # Loss plot　クライアントの評価とテストデータの評価をプロット
        plt.figure(figsize=(12, 6))
        for client_idx in range(num_clients):
            losses = [self.global_loss_history_for_client[round_idx][client_idx] for round_idx in range(num_rounds)]
            plt.scatter(range(num_rounds), losses, label=f'Client {client_idx + 1}')
        plt.scatter(range(num_rounds), self.global_loss_history, label="Global Loss", marker='x')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.title('global Loss Over Rounds')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(SAVE_DIR, "global_loss_history_by_client_and_testloder.png"))
        plt.close()
    
    def plot_client_histories(self, clinet_token_history, clinet_contiribution_history):
        # clinet_token_historyのプロット
        # TODO: 神様視点で評価を入れたい
        num_rounds = len(clinet_token_history)
        print(len(clinet_token_history),num_rounds,clinet_token_history)
        plt.figure(figsize=(12, 6))
        for client_id in range(NUM_CLIENTS):
            rounds_tokens = [token[client_id] for token in clinet_token_history]
            plt.plot(range(num_rounds), rounds_tokens, label=f'Client {client_id + 1}', marker='o')
        plt.xlabel('Round')
        plt.ylabel('Token')
        plt.title('Client Token History Over Rounds')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(SAVE_DIR, "clinet_token_history_by_round.png"))
        plt.close()

        # clinet_tokenのラウンドにおける総量をプロット
        num_rounds = len(clinet_token_history)
        plt.figure(figsize=(12, 6))
        for client_id in range(NUM_CLIENTS):
            client_tokens = []
            total_token = 0
            for token in clinet_token_history:
                total_token += token[client_id]
                client_tokens.append(total_token)
            plt.scatter(range(num_rounds), client_tokens, label=f'Client {client_id + 1}')
        plt.xlabel('Round')
        plt.ylabel('Token')
        plt.title('Client Token History Over Rounds')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(SAVE_DIR, "clinet_token_amount.png"))
        plt.close()

        # clinet_contiribution_historyのプロット
        plt.figure(figsize=(12, 6))
        for client_id in range(NUM_CLIENTS):
            rounds_contributions = [contribution[client_id] for contribution in clinet_contiribution_history]
            plt.scatter(range(num_rounds), rounds_contributions, label=f'Client {client_id + 1}')
        plt.xlabel('Round')
        plt.ylabel('Contribution')
        plt.title('Client Contribution History Over Rounds')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(SAVE_DIR, "clinet_contiribution_history.png"))
        plt.close()

        # plot epochs_history
        plt.figure(figsize=(12, 6))
        plt.scatter(range(num_rounds), epochs_history, label="Epochs")
        plt.xlabel('Round')
        plt.ylabel('Epochs')
        plt.title('Epochs Over Rounds')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(SAVE_DIR, "epochs_history.png"))
        plt.close()





    def save_histories_to_csv(self):
        # Save global_loss_history
        with open(os.path.join(SAVE_DIR, "global_loss_history.csv"), mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Round", "Loss"])
            for round_num, loss in enumerate(self.global_loss_history):
                writer.writerow([round_num, loss])
        
        # Save global_accuracy_history
        with open(os.path.join(SAVE_DIR, "global_accuracy_history.csv"), mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Round", "Accuracy"])
            for round_num, accuracy in enumerate(self.global_accuracy_history):
                writer.writerow([round_num, accuracy])
        
        # Save global_params_history
        with open(os.path.join(SAVE_DIR, "global_params_history.csv"), mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Round", "Parameters"])
            for round_num, params in enumerate(global_params_history):
                writer.writerow([round_num, params])

        # Save local_params_history
        with open(os.path.join(SAVE_DIR, "local_params_history.csv"), mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Round", "Parameters"])
            for round_num, params in enumerate(local_params_history):
                writer.writerow([round_num, params])

        # Save clinet_accuracy_history
        with open(os.path.join(SAVE_DIR, "clinet_accuracy_history.csv"), mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Round"] + [f"Client {i+1}" for i in range(NUM_CLIENTS)])
            for round_num, accuracies in enumerate(clinet_accuracy_history):
                writer.writerow([round_num] + accuracies)

        # Save clinet_loss_history
        with open(os.path.join(SAVE_DIR, "clinet_loss_history.csv"), mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Round"] + [f"Client {i+1}" for i in range(NUM_CLIENTS)])
            for round_num, losses in enumerate(clinet_loss_history):
                writer.writerow([round_num] + losses)

        # Save clinet_all_accuracy_history
        with open(os.path.join(SAVE_DIR, "clinet_all_accuracy_history.csv"), mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Round"] + [f"Client {i+1}" for i in range(NUM_CLIENTS)])
            for round_num, accuracies in enumerate(clinet_all_accuracy_history):
                writer.writerow([round_num] + accuracies)

        # Save clinet_all_loss_history
        with open(os.path.join(SAVE_DIR, "clinet_all_loss_history.csv"), mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Round"] + [f"Client {i+1}" for i in range(NUM_CLIENTS)])
            for round_num, losses in enumerate(clinet_all_loss_history):
                writer.writerow([round_num] + losses)

        # Save clinet_token_history
        with open(os.path.join(SAVE_DIR, "clinet_token_history.csv"), mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Round"] + [f"Client {i+1}" for i in range(NUM_CLIENTS)])
            for round_num, tokens in enumerate(clinet_token_history):
                writer.writerow([round_num] + tokens)

        # Save client_contoribution_history
        with open(os.path.join(SAVE_DIR, "client_contoribution_history.csv"), mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Round"] + [f"Client {i+1}" for i in range(NUM_CLIENTS)])
            for round_num, contributions in enumerate(client_contoribution_history):
                writer.writerow([round_num] + contributions)

        # Save client_trainloader and client_valloader
        with open(os.path.join(SAVE_DIR,'client_trainloader.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Client ID', 'Train Size'])
            writer.writerows(client_trainloader)

        # Save client_trainloader and client_valloader
        with open(os.path.join(SAVE_DIR,'client_valloader.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Client ID', 'Val Size'])
            writer.writerows(client_valloader)
        
        # Save epochs_history
        with open(os.path.join(SAVE_DIR, "epochs_history.csv"), mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Round", "Epochs"])
            for round_num, epochs in enumerate(epochs_history):
                writer.writerow([round_num, epochs])
        
        # Save random_times_history
        with open(os.path.join(SAVE_DIR, "random_times_history.csv"), mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Round", "Random Times"])
            for round_num, random_times in enumerate(random_times_history):
                writer.writerow([round_num, random_times])

def train_single_client(trainloader, valloader, testloader, epochs:List[int],rounds:int):
    # モデルの定義
    train_size = len(trainloader.dataset)
    model = Net()
    model_param = get_parameters(model)
    list_loss = []
    list_accuracy = []
    list_loss_test = [] 
    list_accuracy_test = []
    global LEARNING_RATE
    for round_num in range(rounds):
        print(f"Round {round_num}: Training, Epochs = {epochs[round_num]}")
        set_parameters(model, model_param)
        train(model, trainloader, epochs=epochs[round_num],learning_rate=LEARNING_RATE)
        loss, accuracy = test(model, valloader)
        loss_test, accuracy_test = test(model, testloader)
        if len(list_loss) > 1 and  loss < list_loss[len(list_loss)-1]:
            print(f"Update Model Parameter")
            model_param = get_parameters(model)
        # model_param_1 = get_parameters(model)
        print(f"Round {round_num}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
        print(f"Round {round_num}: Loss(Test) = {loss_test:.4f}, Accuracy(Test) = {accuracy_test:.4f}")
        list_loss.append(loss)
        list_accuracy.append(accuracy)
        list_loss_test.append(loss_test)
        list_accuracy_test.append(accuracy_test)
    
    # モデルの定義 (epoch 1)
    model_1 = Net()
    model_param_1 = get_parameters(model_1)
    list_loss_1 = []
    list_accuracy_1 = []
    list_loss_test_1 = [] 
    list_accuracy_test_1 = []
    for round_num in range(rounds):
        print(f"Round {round_num}: Training, Epochs = {1}")
        set_parameters(model_1, model_param_1)
        train(model_1, trainloader, epochs=1,learning_rate=LEARNING_RATE)
        loss_1, accuracy_1 = test(model_1, valloader)
        loss_test_1, accuracy_test_1 = test(model_1, testloader)
        if len(list_loss_1) > 1 and loss_1 < list_loss_1[len(list_loss_1)-1]:
            print(f"Update Model Parameter")
            model_param_1 = get_parameters(model_1)
        # model_param_1 = get_parameters(model_1)
        print(f"Round {round_num}: Loss = {loss_1:.4f}, Accuracy = {accuracy_1:.4f}")
        print(f"Round {round_num}: Loss(Test) = {loss_test_1:.4f}, Accuracy(Test) = {accuracy_test_1:.4f}")
        list_loss_1.append(loss_1)
        list_accuracy_1.append(accuracy_1)
        list_loss_test_1.append(loss_test_1)
        list_accuracy_test_1.append(accuracy_test_1)
    

    # グラフの描画
    fig, ax1 = plt.subplots(figsize=(10, 6))
    # 第1軸: Loss
    ax1.scatter(range(rounds), list_loss, label="Loss", color='blue', marker='o')
    ax1.scatter(range(rounds), list_loss_test, label="Loss(Test)", color='purple', marker='o')
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Loss", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 第2軸: Accuracy
    ax2 = ax1.twinx()
    ax2.scatter(range(rounds), list_accuracy, label="Accuracy", marker='x', color='red')
    ax2.scatter(range(rounds), list_accuracy_test, label="Accuracy(Test)", marker='x', color='orange')
    ax2.set_ylabel("Accuracy", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
     # タイトルと凡例
    fig.suptitle(f"Single Model vs Global Model Metrics Over Rounds(Train Size:{train_size})")
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax2.legend(loc='upper right', bbox_to_anchor=(1, 1))
    plt.grid()
    plt.savefig(os.path.join(SAVE_DIR, f"single_model_accuracy_and_loss_train-size_{train_size}.png"),bbox_inches='tight')
    plt.close()
    
    # Save csv  
    with open(os.path.join(SAVE_DIR, "single_model_metrics.csv"), mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Round", "Loss", "Accuracy", "Loss(Test)", "Accuracy(Test)"])
        for round_num in range(rounds):
            writer.writerow([round_num, list_loss[round_num], list_accuracy[round_num], list_loss_test[round_num], list_accuracy_test[round_num]])

    # plt accuracy and loss history
    # import csv global_accuracy_history
    with open(os.path.join(SAVE_DIR, "global_accuracy_history.csv"), mode="r") as file:
        reader = csv.reader(file)
        header = next(reader)
        list_accuracy_global = [float(row[1]) for row in reader]
    with open(os.path.join(SAVE_DIR, "global_loss_history.csv"), mode="r") as file:
        reader = csv.reader(file)
        header = next(reader)
        list_loss_global = [float(row[1]) for row in reader]
    plt.figure(figsize=(12, 6))
    plt.scatter(range(rounds), list_loss, label="Loss", color='blue', marker='o')
    plt.scatter(range(rounds), list_loss_test, label="Loss(Test)", color='purple', marker='o')
    plt.scatter(range(rounds), list_loss_global, label="Global Loss", color='green', marker='x')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title(f'Loss in Single Model vs Global Model Over Rounds (Train Size:{train_size})')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(SAVE_DIR, f"single_model_loss_vs_global_loss_train-size_{train_size}.png"))

    plt.figure(figsize=(12, 6))
    plt.scatter(range(rounds), list_accuracy, label="Accuracy", color='red', marker='o')
    plt.scatter(range(rounds), list_accuracy_test, label="Accuracy(Test)", color='orange', marker='o')
    plt.scatter(range(rounds), list_accuracy_global, label="Global Accuracy", color='brown', marker='x')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy in Single Model vs Global Model Over Rounds (Train Size:{train_size})')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(SAVE_DIR, f"single_model_accuracy_vs_global_accuracy_train-size_{train_size}.png"))

    # グラフの描画
    fig, ax1 = plt.subplots(figsize=(10, 6))
    # 第1軸: Loss
    ax1.scatter(range(rounds), list_loss_1, label="Loss(epoch = 1)", color='blue', marker='o')
    ax1.scatter(range(rounds), list_loss_test_1, label="Loss(Test, epoch = 1)", color='purple', marker='o')
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Loss", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 第2軸: Accuracy
    ax2 = ax1.twinx()
    ax2.scatter(range(rounds), list_accuracy_1, label="Accuracy(epoch = 1)", marker='x', color='red')
    ax2.scatter(range(rounds), list_accuracy_test_1, label="Accuracy(Test, epoch = 1)", marker='x', color='orange')
    ax2.set_ylabel("Accuracy", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
     # タイトルと凡例
    fig.suptitle(f"Single Model(epoch = 1) vs Global Model Metrics Over Rounds (Train Size:{train_size})")
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax2.legend(loc='upper right', bbox_to_anchor=(1, 1))
    plt.grid()
    plt.savefig(os.path.join(SAVE_DIR, f"single_model_accuracy_and_loss_Epoch_1_train-size_{train_size}.png"),bbox_inches='tight')
    plt.close()
    
    # Save csv  
    with open(os.path.join(SAVE_DIR, f"single_model_epoch_1_metrics_train-size_{train_size}.csv"), mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Round", "Loss", "Accuracy", "Loss(Test)", "Accuracy(Test)"])
        for round_num in range(rounds):
            writer.writerow([round_num, list_loss_1[round_num], list_accuracy_1[round_num], list_loss_test_1[round_num], list_accuracy_test_1[round_num]])

    # plt accuracy and loss history
    # import csv global_accuracy_history
    with open(os.path.join(SAVE_DIR, "global_accuracy_history.csv"), mode="r") as file:
        reader = csv.reader(file)
        header = next(reader)
        list_accuracy_global = [float(row[1]) for row in reader]
    with open(os.path.join(SAVE_DIR, "global_loss_history.csv"), mode="r") as file:
        reader = csv.reader(file)
        header = next(reader)
        list_loss_global = [float(row[1]) for row in reader]
    plt.figure(figsize=(12, 6))
    plt.scatter(range(rounds), list_loss_1, label="Loss (epoch = 1)", color='blue', marker='o')
    plt.scatter(range(rounds), list_loss_test_1, label="Loss(Test, epoch = 1)", color='purple', marker='o')
    plt.scatter(range(rounds), list_loss_global, label="Global Loss", color='green', marker='x')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.title(f'Loss in Single Model(Epoch = 1) vs Global Model Over Rounds (Train Size:{train_size})')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(SAVE_DIR, f"single_model_epoch_1_loss_vs_global_loss_train-size_{train_size}.png"))

    plt.figure(figsize=(12, 6))
    plt.scatter(range(rounds), list_accuracy_1, label="Accuracy (epoch = 1)", color='red', marker='o')
    plt.scatter(range(rounds), list_accuracy_test_1, label="Accuracy(Test, epoch = 1)", color='orange', marker='o')
    plt.scatter(range(rounds), list_accuracy_global, label="Global Accuracy", color='brown', marker='x')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy in Single Model(epoch = 1) vs Global Model Over Rounds (Train Size:{train_size})')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(SAVE_DIR, f"single_model_epoch_1_accuracy_vs_global_accuracy_train-size_{train_size}.png"))

def plot_single_and_global_metrics(min_data_size:int, max_data_size:int) -> None:
    with open(os.path.join(SAVE_DIR, "global_accuracy_history.csv"), mode="r") as file:
        reader = csv.reader(file)
        header = next(reader)
        list_accuracy_global = [float(row[1]) for row in reader]
    with open(os.path.join(SAVE_DIR, "global_loss_history.csv"), mode="r") as file:
        reader = csv.reader(file)
        header = next(reader)
        list_loss_global = [float(row[1]) for row in reader]
    with open(os.path.join(SAVE_DIR, f"single_model_epoch_1_metrics_train-size_{min_data_size}.csv"), mode="r") as file:
        reader = csv.reader(file)
        header = next(reader)
        rows = list(reader) 
        list_accuracy_single_9000 = [float(row[2]) for row in rows]
        list_loss_single_9000 = [float(row[1]) for row in rows]
        list_accuracy_single_9000_test = [float(row[4]) for row in rows]
        list_loss_single_9000_test = [float(row[3]) for row in rows]
    with open(os.path.join(SAVE_DIR, f"single_model_epoch_1_metrics_train-size_{max_data_size}.csv"), mode="r") as file:
        reader = csv.reader(file)
        header = next(reader)
        rows = list(reader) 
        list_accuracy_single_450000 = [float(row[2]) for row in rows]
        list_loss_single_450000 = [float(row[1]) for row in rows]
        list_accuracy_single_450000_test = [float(row[4]) for row in rows]
        list_loss_single_450000_test = [float(row[3]) for row in rows]


    
    # plt.rcParams['font.family'] = 'DejaVu Sans' # font familyの設定
    # plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
    # plt.rcParams["font.size"] = 15 # 全体のフォントサイズが変更されます。
    # plt.rcParams['xtick.labelsize'] = 9 # 軸だけ変更されます。
    # plt.rcParams['ytick.labelsize'] = 24 # 軸だけ変更されます
    # plt.rcParams['xtick.direction'] = 'in' # x axis in
    # plt.rcParams['ytick.direction'] = 'in' # y axis in 
    # plt.rcParams['axes.linewidth'] = 1.0 # axis line width
    # plt.rcParams['axes.grid'] = True # make grid
    # plt.rcParams["legend.fancybox"] = False # 丸角
    # plt.rcParams["legend.framealpha"] = 1 # 透明度の指定、0で塗りつぶしなし
    # plt.rcParams["legend.edgecolor"] = 'black' # edgeの色を変更
    # plt.rcParams["legend.handlelength"] = 1 # 凡例の線の長さを調節
    # plt.rcParams["legend.labelspacing"] = 5. # 垂直方向の距離の各凡例の距離
    # plt.rcParams["legend.handletextpad"] = 3. # 凡例の線と文字の距離の長さ
    # plt.rcParams["legend.markerscale"] = 2 # 点がある場合のmarker scale
    # plt.rcParams["legend.borderaxespad"] = 0. # 凡例の端とグラフの端を合わせる
    # plt.rcParams['figure.dpi'] = 300 # dpiの設定

    plt.figure( figsize=(12, 6))
    plt.scatter(range(ROUND_NUM), list_loss_global, label="Global Loss", color='green', marker='x')
    plt.scatter(range(ROUND_NUM), list_loss_single_9000, label=f"Single Model Loss(Train Size:{min_data_size})", color='blue', marker='o')
    plt.scatter(range(ROUND_NUM), list_loss_single_9000_test, label=f"Single Model Loss(Test, Train Size:{min_data_size})", color='purple', marker='o')
    plt.scatter(range(ROUND_NUM), list_loss_single_450000, label=f"Single Model Loss(Train Size:{max_data_size})", color='blue', marker='^')
    plt.scatter(range(ROUND_NUM), list_loss_single_450000_test, label=f"Single Model Loss(Test, Train Size:{max_data_size})", color='purple', marker='^')
    plt.xlabel(r"$Round$")
    plt.ylabel(r"$Loss$")
    plt.title(f'Global Loss vs Single Model({min_data_size}{max_data_size}) Loss Over Rounds')
    #plt.legend(ncol=2, bbox_to_anchor=(0., 1.025, 1., 0.102), loc=3)
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(SAVE_DIR, f"global_loss_vs_single_model_{min_data_size}_{max_data_size}_loss.png"), bbox_inches="tight", pad_inches=0.05)
    plt.close()

    plt.figure( figsize=(12, 6))
    plt.scatter(range(ROUND_NUM), list_accuracy_global, label="Global Accuracy", color='brown', marker='x')
    plt.scatter(range(ROUND_NUM), list_accuracy_single_9000, label=f"Single Model Accuracy(Train Size:{min_data_size})", color='red', marker='o')
    plt.scatter(range(ROUND_NUM), list_accuracy_single_9000_test, label=f"Single Model Accuracy(Test, Train Size:{min_data_size})", color='orange', marker='o')
    plt.scatter(range(ROUND_NUM), list_accuracy_single_450000, label=f"Single Model Accuracy(Train Size:{max_data_size})", color='red', marker='^')
    plt.scatter(range(ROUND_NUM), list_accuracy_single_450000_test, label=f"Single Model Accuracy(Test, Train Size:{max_data_size})", color='orange', marker='^')
    plt.xlabel(r"$Round$")
    plt.ylabel(r"$Accuracy$")
    plt.title(f'Global Accuracy vs Single Model({min_data_size},{max_data_size}) Accuracy Over Rounds')
    plt.legend()
    # plt.legend(ncol=2, bbox_to_anchor=(0., 1.025, 1., 0.102), loc=3)
    plt.grid()
    plt.savefig(os.path.join(SAVE_DIR, f"global_accuracy_vs_single_model_{min_data_size}_{max_data_size}_accuracy.png"), bbox_inches="tight", pad_inches=0.05)
    plt.close()

def plot_client_token_history(current_time:str):
    # 利用可能なスタイルを確認
    print(plt.style.available)

    # ファイルパスの確認
    file_path = f'Base/{current_time}/clinet_token_history.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # データの読み込み
    df = pd.read_csv(file_path)

    # データフレームの列名を確認
    print(df.columns)

    # 各クライアントのトークンの総量の累積を計算
    cumulative_tokens = df.iloc[:, 1:].cumsum()

    # グラフのスタイルを設定
    plt.style.use(['science', 'grid', 'no-latex'])

    # カラーマップを設定
    # colors = plt.cm.tab10(np.linspace(0, 1, len(df.columns) - 1))
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    # マーカーのリストを設定
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']

    # グラフの描画
    plt.figure(figsize=(8, 5))
    for i, (color, marker) in enumerate(zip(colors, markers), start=1):
        plt.plot(df.iloc[:, 0], cumulative_tokens.iloc[:, i-1], label=f'Node{i}', marker=marker, color=color,lw=1.5, markersize=7)
    # グリッドを追加
    plt.grid(True, linestyle='--', alpha=0.7)

    # タイトルとラベルを追加
    # plt.title('Cumulative Client Token Over Rounds', fontsize=18)
    plt.xlabel('Round', fontsize=14)
    plt.ylabel('Cumulative Node Token', fontsize=14)
    # x軸を整数に設定
    plt.xticks(np.arange(df.iloc[:, 0].min(), df.iloc[:, 0].max() + 2, 2), fontsize=16)
    plt.yticks(fontsize=16)

    # 凡例の位置を変更
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

    # グラフの表示
    plt.tight_layout()
    plt.show()

    # グラフの保存
    plt.savefig(f'Base/{current_time}/0_client_token_amount_history.png', bbox_inches='tight')

if __name__ == "__main__":
    # NUM_CLIENTS = 5 # クライアント数
    NUM_CLIENTS = 15 # クライアント数
    EPOCHS = 1 # 初期のエポック数
    LEARNING_RATE = 0.01 # 学習率
    # Define a directory to save the plots
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d-%H-%M")
    SAVE_DIR = f"Base/{current_time}"
    os.makedirs(SAVE_DIR, exist_ok=True)
    ROUND_NUM = 20 # ラウンド数
    # CLIENT_DATA_SIZE = [0.1, 0.1, 0.2, 0.2, 0.4]
    # CLIENT_DATA_SIZE = [0.05, 0.05, 0.2, 0.2, 0.5]
    # CLIENT_DATA_SIZE = [0.001, 0.001, 0.198, 0.4, 0.4]
    # CLIENT_DATA_SIZE = [0.002, 0.002, 0.002, 0.497, 0.497]
    CLIENT_DATA_SIZE = [1/NUM_CLIENTS for _ in range(NUM_CLIENTS)]
    # CLIENT_DATA_SIZE = [0.05, 0.05, 0.05, 0.05, 0.05, 0.15, 0.15, 0.15, 0.15, 0.15]
    # CLIENT_DATA_SIZE = [0.001, 0.2475, 0.2475, 0.2475, 0.2565]
    # データセットのロード

    # trainloaders, valloaders, testloader = load_datasets(NUM_CLIENTS, CLIENT_DATA_SIZE)
    print(f"Client Data Size: {CLIENT_DATA_SIZE}")
    print(f"Length of Client Data Size: {len(CLIENT_DATA_SIZE)}")
    trainloaders, valloaders, testloader = load_datasets_stratified(NUM_CLIENTS, CLIENT_DATA_SIZE)
    # trainloaders, valloaders, testloader = load_uniform_datasets(NUM_CLIENTS)
    for idx, trainloader in enumerate(trainloaders):
        print(f"Client {idx + 1}: Train Size = {len(trainloader.dataset)}, Val Size = {len(valloaders[idx].dataset)}")
    
    trainloader = trainloaders[0]
    valloader = valloaders[0]
    testdata = testloader
    print(f"Train Size = {len(trainloader.dataset)}, Val Size = {len(valloader.dataset)}, Test Size = {len(testdata.dataset)}")

    # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
    client_resources = None
    if DEVICE.type == "cuda":
        client_resources = {"num_gpus": 1}


    # Flower simulation configuration
    #strategy = EnhancedFedCustom()
    strategy = FedCustom()

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(ROUND_NUM),
        strategy=strategy,
        client_resources=client_resources,
    )

    # Post-simulation plotting and saving
    strategy.plot_global_metrics()
    strategy.save_contribution_scores()
    strategy.plot_client_performance(clinet_accuracy_history, clinet_loss_history)
    strategy.plot_client_histories(clinet_token_history, client_contoribution_history)
    strategy.save_histories_to_csv()

    train_single_client(trainloader, valloader, testdata, epochs_history,ROUND_NUM)
    combined_train_dataset = ConcatDataset([loader.dataset for loader in trainloaders])
    combined_val_dataset = ConcatDataset([loader.dataset for loader in valloaders])
    combined_trainloader = DataLoader(combined_train_dataset, batch_size=32, shuffle=True)
    combined_valloader = DataLoader(combined_val_dataset, batch_size=32)
    print(f"Combined Train Size = {len(combined_trainloader.dataset)}, Combined Val Size = {len(combined_valloader.dataset)}")
    train_single_client(combined_trainloader, combined_valloader, testdata, epochs_history,ROUND_NUM)
    min_data_size = len(trainloaders[0].dataset)
    max_data_size = len(combined_trainloader.dataset)
    plot_single_and_global_metrics(min_data_size, max_data_size)
    plot_client_token_history(current_time)



    