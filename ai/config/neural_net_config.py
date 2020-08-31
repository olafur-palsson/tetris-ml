import json
from dataclasses import dataclass
from typing import List, Optional, Dict

from dacite import from_dict


@dataclass
class SGDConfig:
    momentum: float
    learning_rate: float


@dataclass
class ConvLayerConfig:
    nodes: int
    input_channels: int
    output_channels: int
    kernel_size: int
    stride: int


@dataclass
class LinearLayerConfig:
    nodes: int
    bias: Optional[bool]


@dataclass
class LayerConfig:
    nodes: int
    type: str
    linear: Optional[LinearLayerConfig]
    conv: Optional[ConvLayerConfig]


@dataclass
class NetworkConfig:
    id: str
    previous: Optional[str]
    hidden_layers: List[LayerConfig]
    input: int
    output: int
    sgd: SGDConfig


@dataclass
class Config:
    type: str
    name: str
    filename: str
    options: Optional[Dict[str, any]]
    networks: Dict[str, NetworkConfig]

    @classmethod
    def from_file_path(cls, path: str):
        with open(path) as file:
            dict_config = json.load(file)
            return from_dict(cls, dict_config)
