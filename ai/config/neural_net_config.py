import json
from dataclasses import dataclass
from typing import List, Optional, Dict

from dacite import from_dict


@dataclass
class SGDConfig:
    momentum: float
    learning_rate: float
    learning_rate_policy_gradient: float


@dataclass
class LayerConfig:
    nodes: int
    bias: Optional[bool]
    relu: Optional[bool]


@dataclass
class NetworkConfig:
    id: str
    previous: Optional[str]
    hidden_layers: List[LayerConfig]
    output: int
    sgd: SGDConfig


@dataclass
class Config:
    type: str
    name: str
    filename: str
    options: Dict[str, any]
    networks: Dict[str, NetworkConfig]

    @classmethod
    def from_file_path(cls, path: str):
        with open(path) as file:
            dict_config = json.load(file)
            return from_dict(cls, dict_config)
