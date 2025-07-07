from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10, CIFAR100

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


# Define a directory to save the plots
now = datetime.datetime.now()
current_time = now.strftime("%Y-%m-%d-%H-%M")

SAVE_DIR = f"CIFAR10/test-flwr-data/{current_time}"
os.makedirs(SAVE_DIR, exist_ok=True)

# Initialize lists to store parameter changes
global_params_history = []
local_params_history = []

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)

NUM_CLIENTS = 2 # クライアント数を2に設定

def load_datasets(num_clients: int):
    # Download and transform CIFAR-10 (train and test)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10("./dataset", train=True, download=True, transform=transform)
    testset = CIFAR10("./dataset", train=False, download=True, transform=transform)
    # trainset = CIFAR100("./dataset", train=True, download=True, transform=transform)
    # testset = CIFAR100("./dataset", train=False, download=True, transform=transform)

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
    testloader = DataLoader(testset, batch_size=32)
    return trainloaders, valloaders, testloader

# class Net(nn.Module):
#     def __init__(self) -> None:
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x



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

  
# class Net(nn.Module):
#     def __init__(self) -> None:
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
#         self.drop1 = nn.Dropout(0.25)
#         self.pool3 = nn.MaxPool2d(2, 2)
#         self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
#         self.pool4 = nn.MaxPool2d(2, 2)
#         self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
#         self.drop2 = nn.Dropout(0.25)
#         self.fc1 = nn.Linear(256, 512)
#         self.drop3 = nn.Dropout(0.45)
#         self.fc2 = nn.Linear(512, 10)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.pool1(F.relu(self.conv1(x)))
#         x = self.pool2(F.relu(self.conv2(x)))
#         x = F.relu(self.conv3(x))
#         x = self.drop1(x)
#         x = self.pool3(x)
#         x = self.pool4(F.relu(self.conv4(x)))
#         x = F.relu(self.conv5(x))
#         x = self.drop2(x)
#         x = torch.flatten(x, 1)
#         x = F.relu(self.fc1(x))
#         x = self.drop3(x)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def train(net, trainloader, epochs: int):
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
        train(self.net, self.trainloader, epochs=1)
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
    
    def evaluate_other_clients(self, client_parameters):
        """Evaluate other clients' models using this client's validation data."""
        results = {}
        for cid, params in client_parameters.items():
            set_parameters(self.net, params)
            loss, _ = test(self.net, self.valloader)
            results[cid] = loss
        return results

def client_fn(cid) -> FlowerClient:
    net = Net().to(DEVICE)
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    return FlowerClient(cid, net, trainloader, valloader)

class FedCustom(fl.server.strategy.Strategy):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.current_round_global_params = None  # 現在のグローバルパラメータを保持

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

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        global global_params_history


        # Aggregate parameters
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        # Save the aggregated global parameters
        global_params_history.append(parameters_aggregated)

        # Update the current round global parameters
        self.current_round_global_params = parameters_aggregated

        metrics_aggregated = {}
        return parameters_aggregated, metrics_aggregated

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

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""
        # Let's assume we won't perform the global model evaluation on the server side.
        return None

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

class EnhancedFedCustom(FedCustom):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_loss_history = []  # グローバル損失の履歴
        self.global_accuracy_history = []  # グローバル精度の履歴
        self.client_contribution_scores = {}  # クライアントごとの貢献度スコア
        self.global_learning_rate = 0.5  # グローバルモデル更新の学習率

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average with adjustments."""
        global global_params_history

        # Extract parameters and client contributions
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        # Aggregate parameters using weighted average
        aggregated_params = aggregate(weights_results)

        # Apply global learning rate to smooth updates
        current_global_params = parameters_to_ndarrays(self.current_round_global_params)
        aggregated_params = [
            current_global_param + self.global_learning_rate * (new_param - current_global_param)
            for current_global_param, new_param in zip(current_global_params, aggregated_params)
        ]

        # Save the aggregated global parameters
        parameters_aggregated = ndarrays_to_parameters(aggregated_params)
        global_params_history.append(parameters_aggregated)

        # Update the current round global parameters
        self.current_round_global_params = parameters_aggregated

        # Evaluate global model on the test set
        net = Net().to(DEVICE)
        set_parameters(net, aggregated_params)
        global_loss, global_accuracy = test(net, testloader)
        self.global_loss_history.append(global_loss)
        self.global_accuracy_history.append(global_accuracy)

        # Print evaluation results
        print(
            f"Round {server_round}: Global Loss = {global_loss:.4f}, Global Accuracy = {global_accuracy:.4f}"
        )

        # Calculate and store client contributions
        for client_proxy, fit_res in results:
            client_id = int(client_proxy.cid)
            num_examples = fit_res.num_examples
            contribution_score = 1 / num_examples  # シンプルな例：データ量に反比例
            if client_id not in self.client_contribution_scores:
                self.client_contribution_scores[client_id] = []
            self.client_contribution_scores[client_id].append(contribution_score)

        return parameters_aggregated, {}
    # def aggregate_fit(
    #         self,
    #         server_round: int,
    #         results: List[Tuple[ClientProxy, FitRes]],
    #         failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    #     ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
    #     """Aggregate fit results using client-evaluated weights."""
    #     # Collect parameters from each client
    #     client_parameters = {
    #         client_proxy.cid: parameters_to_ndarrays(fit_res.parameters)
    #         for client_proxy, fit_res in results
    #     }

    #     # Collect evaluation results
    #     all_evaluations = {}
    #     for client_proxy, fit_res in results:
    #         # Simulate client evaluating others
    #         client_eval_results = fl.simulation.run_client_fn(
    #             client_fn(client_proxy.cid)
    #         ).evaluate_other_clients(client_parameters)
    #         all_evaluations[client_proxy.cid] = client_eval_results

    #     # Calculate weights based on evaluations
    #     weights = []
    #     for cid, evaluations in all_evaluations.items():
    #         avg_loss = np.mean(list(evaluations.values()))
    #         weight = 1.0 / (avg_loss + 1e-6)  # Avoid division by zero
    #         weights.append((client_parameters[cid], weight))

    #     # Aggregate parameters using calculated weights
    #     aggregated_parameters = aggregate(weights)
    #     aggregated_parameters = ndarrays_to_parameters(aggregated_parameters)

    #     # Update global parameters
    #     self.current_round_global_params = aggregated_parameters

    #     return aggregated_parameters, {}

    def plot_global_metrics(self):
        """Plot global loss and accuracy."""
        rounds = range(1, len(self.global_loss_history) + 1)
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



trainloaders, valloaders, testloader = load_datasets(NUM_CLIENTS)

# Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
client_resources = None
if DEVICE.type == "cuda":
    client_resources = {"num_gpus": 1}


# Flower simulation configuration
strategy = EnhancedFedCustom()

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=100),
    strategy=strategy,
    client_resources=client_resources,
)

# Post-simulation plotting and saving
strategy.plot_global_metrics()
strategy.save_contribution_scores()
