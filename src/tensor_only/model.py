from pathlib import Path
import torch
from torch import Tensor
from torch.nn import Module, Linear
from typing import Dict, Optional
import math

from tensor_only.config import ModelConfig

class Model():
    def __init__(self, model_disk_path: Optional[str | Path], config: ModelConfig):
        params = self._load_or_generate(model_disk_path, config)
        self.embeddings: Tensor = params["embeddings"]
        self.w1: Tensor = params["w1"]
        self.b1: Tensor = params["b1"]
        self.w2: Tensor = params["w2"]
        self.b2: Tensor = params["b2"]
        

    def _load_or_generate(self, model_disk_path: Optional[str | Path], config: ModelConfig):
        params: Dict[str, Tensor] = dict()
        if model_disk_path:
            params = torch.load(model_disk_path, weights_only=True)
        else:
            g = torch.Generator().manual_seed(config.manual_seed)
            # Random nums, zero-centered
            params["embeddings"] = torch.randn((config.word_dict_size, config.embedding_size), generator=g)
            # Random nums, zero-centered and scaled by sqrt(inputs)
            params["w1"] = torch.randn((config.embedding_size * config.context_size, config.l1_size), generator=g) / math.sqrt(config.embedding_size * config.context_size)
            # Zeros
            params["b1"] = torch.zeros((config.l1_size))
            # Random nums, zero-centered and scaled by sqrt(inputs)
            params["w2"] = torch.randn((config.l1_size, config.word_dict_size), generator=g) / math.sqrt(config.l1_size)
            # Zeros
            params["b2"] = torch.zeros((config.word_dict_size))

        for param in params.values():
            # Why not earlier? 
            # We would have tracked the weights normalization too
            param.requires_grad = True

        return params

    def forward(self):
        pass