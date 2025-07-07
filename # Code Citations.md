# Code Citations

## License: 不明
https://github.com/Morales97/decentralized-DL/tree/aa561507d2986b856e40902ef394f42ddb3cc5a6/helpers/search_avg.py

```
model(data)
            loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum
```


## License: Apache_2_0
https://github.com/xiangshuai-wuqiwei/Flower/tree/8276c578aa07ea43d2758a7a08e764cf6809b33b/src/py/flwr/server/strategy/fedavg.py

```
self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        ""
```

