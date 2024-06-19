from typing import Dict, List, Tuple, Union, Optional
from collections import OrderedDict
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes
from models import DLinear, Linear, NLinear

import flwr as fl
import torch
import numpy as np
import Linear_LTSF
import sys

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[List[np.ndarray]], Dict[str, any]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

            # Save the model
            torch.save(net.state_dict(), f"model_round_{server_round}.pth")

        return aggregated_parameters, aggregated_metrics
    

if __name__ == "__main__":
    sys_args = sys.argv[1:]
    if len(sys_args) < 2:
        print("Run server with args:\n python server.py model_type seq_len \
              \n example: python server.py Linear 192")
        sys.exit()
    args = Linear_LTSF.Args(
        model=sys_args[0], target='0', batch_size=16, seq_len=int(sys_args[1]), pred_len=24
        )
    model_dict = {
        'DLinear': DLinear,
        'NLinear': NLinear,
        'Linear': Linear,
    }
    model = model_dict[args.model].Model(args).float()
    net = model.to(DEVICE)
    
    strategy = SaveModelStrategy()
    fl.server.start_server(server_address="0.0.0.0:8080", config=fl.server.ServerConfig(num_rounds=1, round_timeout=None), strategy=strategy)