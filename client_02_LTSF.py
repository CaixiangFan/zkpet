from collections import OrderedDict
from typing import Dict, List, Tuple

import Linear_LTSF
import numpy as np
import torch
from models import DLinear, Linear, NLinear
import flwr as fl

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LtsfClient(fl.client.NumPyClient):
    """Flower client implementing Linear-LTSF using
    PyTorch."""

    def __init__(
        self,
        model,
        args: Linear_LTSF.Args,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        num_examples: Dict,
    ) -> None:
        self.model = model
        self.args = args
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples

    def get_parameters(self, config) -> List[np.ndarray]:
        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        Linear_LTSF.train(self.model, self.args)
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = Linear_LTSF.test(self.model, self.args)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}



def main() -> None:
    """Load data, start LtsfClient."""
    # Load model and data

    args = Linear_LTSF.Args(
        model='Linear', target='1', batch_size=16, seq_len=96, pred_len=24
        )
    model_dict = {
        'DLinear': DLinear,
        'NLinear': NLinear,
        'Linear': Linear,
    }
    model = model_dict[args.model].Model(args).float()
    model.to(DEVICE)

    trainloader, testloader, num_examples = Linear_LTSF.load_data(args=args)

    # Start client
    client = LtsfClient(model, args, trainloader, testloader, num_examples)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)


if __name__ == "__main__":
    main()