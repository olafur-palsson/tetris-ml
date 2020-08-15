import json
from dataclasses import dataclass

from dacite import from_dict


@dataclass
class NeuralNetConfig:
    temporal_delay: int
    hidden_layers: [int]
    input_layer: int
    output_layer: int


@dataclass
class SGDConfig:
    momentum: float
    learning_rate: float
    learning_rate_policy_gradient: float


@dataclass
class AgentConfig:
    net: str
    use_policy_gradient: bool
    feature_vector: str
    epsilon: float
    filename: str
    nn: NeuralNetConfig
    sgd: SGDConfig

    @classmethod
    def from_file_path(cls, path: str):
        with open(path) as file:
            dict_config = json.load(file)
            return from_dict(cls, dict_config)
