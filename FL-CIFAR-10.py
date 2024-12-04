from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10

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
"""
提案手法用のコード

"""
# Define a directory to save the plots
now = datetime.datetime.now()
current_time = now.strftime("%Y-%m-%d-%H-%M")

NUM_CLIENTS = 5 # クライアント数
EPOCHS = 1 # エポック数

SAVE_DIR = f"CIFAR10/node-{NUM_CLIENTS}/{current_time}"
os.makedirs(SAVE_DIR, exist_ok=True)

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
BASE_TOKEN = 10
DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)


def load_datasets(num_clients: int):
    # Download and transform CIFAR-10 (train and test)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10("./dataset", train=True, download=True, transform=transform)
    testset = CIFAR10("./dataset", train=False, download=True, transform=transform)

    # Split training set into `num_clients` partitions to simulate different local datasets
    partition_size = len(trainset) // num_clients
    lengths = [partition_size] * num_clients
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds) // 10  # 10 % validation set
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
        client_trainloader.append((len(trainloader.dataset),trainloader.dataset))
        client_valloader.append((len(valloader.dataset),valloader.dataset))
    
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

def train(net, trainloader, epochs):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
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
        train(self.net, self.trainloader, epochs=EPOCHS)  
        

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
    net = Net().to(DEVICE)
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    return FlowerClient(cid, net, trainloader, valloader)

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

    def __repr__(self) -> str:
        return "FedCustom"

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
        n_clients = len(clients)
        half_clients = n_clients // 2
        standard_config = {"lr": 0.001}
        higher_lr_config = {"lr": 0.001}
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
                if int(client_proxy.cid) != int(client_id):
                    model = Net().to(DEVICE)
                    set_parameters(model, parameters_to_ndarrays(fit_res.parameters))
                    loss, accuracy = evaluate_model_on_own_data(model, valloader)
                    tmp_loss_for_client[int(client_proxy.cid)] = loss
                    tmp_accuracy_for_client[int(client_proxy.cid)] = accuracy
                    if loss < GLoss:
                        # クライアントごとのLossとAccuracyを提出する
                        clinet_accuracy[int(client_id)].append(accuracy)
                        clinet_loss[int(client_id)].append(loss)
                        tmp_accuracy.append(accuracy)
                        tmp_loss.append(loss)
                        own_test_results.append((client_proxy.cid, loss, accuracy))
                        print(f"Client {client_id} -> Client {client_proxy.cid}  Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
                    else:
                        print(f"Client {client_id} -> Client {client_proxy.cid}  Accuracy: {accuracy:.4f}, Loss: {loss:.4f} (Not submitted)")
                        tmp_accuracy.append(None)
                        tmp_loss.append(None)
                else:
                    model = Net().to(DEVICE)
                    set_parameters(model, parameters_to_ndarrays(fit_res.parameters))
                    loss, accuracy = evaluate_model_on_own_data(model, valloader)
                    tmp_accuracy_for_client[int(client_id)] = accuracy
                    tmp_loss_for_client[int(client_id)] = loss
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
            client_id = int(client_proxy.cid)
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
            
            if len(accuracies) > 1:
                accuracy_mean = np.mean(accuracies)
                accuracy_se = stats.sem(accuracies)
                accuracy_ci = stats.t.interval(0.95, len(accuracies)-1, loc=accuracy_mean, scale=accuracy_se)
                loss_mean = np.mean(losses)
                loss_se = stats.sem(losses)
                loss_ci = stats.t.interval(0.95, len(losses)-1, loc=loss_mean, scale=loss_se)
                
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
                
            elif len(accuracies) == 1:
                accuracy_list.append(np.mean(accuracies))
                loss_list.append(np.mean(losses))
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
        # if total_weight > 0:
        if total_weight > 0 and none_count < NUM_CLIENTS//2+1:
            print(f"Next Round Epochs: {EPOCHS}")
            # if none_count > NUM_CLIENTS//2+1:
            #     print(f"None count :{none_count}")
            #     if EPOCHS < 20:
            #         EPOCHS += 1
            #     print(f"Next Round Epochs: {EPOCHS}")

            normalized_weights = [weight / total_weight if weight > 0 else 0 for weight in weights]
            # Calculate and store client contributions
            for client_proxy, _ in results:
                client_id = int(client_proxy.cid)
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
                client_token[idx] += BASE_TOKEN * weight
            clinet_token_history.append(client_token)
        else:
            print(f"None count :{none_count}")
            clinet_token_history.append([0 for _ in range(NUM_CLIENTS)])
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
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        for client_idx in range(num_clients):
            for client_id in range(num_clients):
                accuracies = [self.client_accuracy_history_by_client[round_idx][client_idx][client_id] for round_idx in range(num_rounds)]
                plt.scatter(range(num_rounds), accuracies, label=f'Accuracy: Client {client_id + 1}',color=colors[client_id])
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.title('Client Accuracy Over Rounds')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(SAVE_DIR, f"clinet_accuracy_history_by_client.png"))
        plt.close()

        #一枚にまとめる
        plt.figure(figsize=(12, 6))
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        for client_idx in range(num_clients):
            for client_id in range(num_clients):
                losses = [self.client_loss_history_by_client[round_idx][client_idx][client_id] for round_idx in range(num_rounds)]
                plt.scatter(range(num_rounds), losses,label=f'Loss": Client {client_id + 1}',color=colors[client_id])
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
            plt.scatter(range(num_rounds), rounds_tokens, label=f'Client {client_id + 1}')
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


def train_single_client(trainloader, valloader, testloader, epochs:List[int],rounds:int):
    # モデルの定義
    model = Net()
    model_param = get_parameters(model)
    list_loss = []
    list_accuracy = []
    list_loss_test = [] 
    list_accuracy_test = []
    for round_num in range(rounds):
        print(f"Round {round_num}: Training, Epochs = {epochs[round_num]}")
        set_parameters(model, model_param)
        train(model, trainloader, epochs=epochs[round_num])
        loss, accuracy = test(model, valloader)
        loss_test, accuracy_test = test(model, testloader)
        model_param = get_parameters(model)
        print(f"Round {round_num}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
        print(f"Round {round_num}: Loss(Test) = {loss_test:.4f}, Accuracy(Test) = {accuracy_test:.4f}")
        list_loss.append(loss)
        list_accuracy.append(accuracy)
        list_loss_test.append(loss_test)
        list_accuracy_test.append(accuracy_test)
    

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
    fig.suptitle("Single Model vs Global Model Metrics Over Rounds")
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax2.legend(loc='upper right', bbox_to_anchor=(1, 2))
    plt.grid()
    plt.savefig(os.path.join(SAVE_DIR, "single_model_accuracy_and_loss.png"),bbox_inches='tight')
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
    plt.title('Loss in Single Model vs Global Model Over Rounds')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(SAVE_DIR, "single_model_loss_vs_global_loss.png"))

    plt.figure(figsize=(12, 6))
    plt.scatter(range(rounds), list_accuracy, label="Accuracy", color='red', marker='o')
    plt.scatter(range(rounds), list_accuracy_test, label="Accuracy(Test)", color='orange', marker='o')
    plt.scatter(range(rounds), list_accuracy_global, label="Global Accuracy", color='brown', marker='x')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Accuracy in Single Model vs Global Model Over Rounds')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(SAVE_DIR, "single_model_accuracy_vs_global_accuracy.png"))

ROUND_NUM = 20

trainloaders, valloaders, testloader = load_datasets(NUM_CLIENTS)

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

trainloader = trainloaders[0]
valloader = valloaders[0]
train_single_client(trainloader, valloader, testloader, epochs_history,ROUND_NUM)

