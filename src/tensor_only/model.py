from pathlib import Path
import torch
from torch import Tensor
from torch.nn import Module, Linear
from typing import Dict, Optional
import math
from config import ModelConfig

class Model():
    def __init__(self, config: ModelConfig):
        params = self._load_or_generate(config)
        self.model_disk_path : str | Path = config.model_disk_path
        self.learning_rate = config.learning_rate
        self.embeddings: Tensor = params["embeddings"]
        self.w1: Tensor = params["w1"]
        self.b1: Tensor = params["b1"]
        self.w2: Tensor = params["w2"]
        self.b2: Tensor = params["b2"]
        

    def _load_or_generate(self, config: ModelConfig):
        params: Dict[str, Tensor] = dict()
        if Path(config.model_disk_path).exists():
            params = torch.load(config.model_disk_path, weights_only=True)
        else:
            g = torch.Generator().manual_seed(config.manual_seed)
            # Random nums, zero-centered
            params["embeddings"] = torch.randn((config.word_dict_size, config.embedding_size), generator=g)
            # Random nums, zero-centered and scaled by sqrt(inputs)
            params["w1"] = torch.randn((config.embedding_size * config.context_size, config.l1_size), generator=g) / math.sqrt(config.embedding_size * config.context_size)
            # Zeros
            params["b1"] = torch.zeros(config.l1_size)
            # Random nums, zero-centered and scaled by sqrt(inputs)
            params["w2"] = torch.randn((config.l1_size, config.word_dict_size), generator=g) / math.sqrt(config.l1_size)
            # Zeros
            params["b2"] = torch.zeros(config.word_dict_size)

        for param in params.values():
            # Why not earlier? 
            # We would have tracked the weights normalization too
            param.requires_grad = True

        return params

    # lookup tensor is BxC and each row represents the index of the words in the embedding tensor
    def forward(self, lookup: Tensor) -> Tensor:
        """
        To make this click:

        self.embeddings[2]
        [30, 31]

        
        self.embeddings[[2, 0, 1]]
        [
            [30, 31],   ← row 2
            [10, 11],   ← row 0
            [20, 21],   ← row 1
        ]

        
        indices = [
            [2, 0, 1],   ← first example
            [1, 3, 2],   ← second example
        ]
        self.embeddings[indices]
        [
            [                         ← first example
                [30, 31],             ← token 2
                [10, 11],             ← token 0
                [20, 21],             ← token 1
            ],
            [                         ← second example
                [20, 21],             ← token 1
                [40, 41],             ← token 3
                [30, 31],             ← token 2
            ],
        ]
        """
        # b_input: [B, C, D]
        # where B is the batch size
        # C is the context size (how many words)
        # D is the size of the embeddings
        b_input = self.embeddings[lookup]
        # flat_b_input is [B, C*D]
        flat_b_input = torch.flatten(b_input, start_dim=1)
        z_1 = flat_b_input @ self.w1 + self.b1
        a_1 = torch.tanh(z_1)
        logits = a_1 @ self.w2 + self.b2

        return logits

    def clear_grad(self):
        self.w1.grad = None
        self.b1.grad = None
        self.w2.grad = None
        self.b2.grad = None
        self.embeddings.grad = None

    def update_params(self):
        with torch.no_grad():
            self.w1 -= self.learning_rate * self.w1.grad
            self.b1 -= self.learning_rate * self.b1.grad
            self.w2 -= self.learning_rate * self.w2.grad
            self.b2 -= self.learning_rate * self.b2.grad
            self.embeddings -= self.learning_rate * self.embeddings.grad

    def save(self):
        torch.save(
            {
                "embeddings": self.embeddings.detach().cpu(),
                "w1": self.w1.detach().cpu(),
                "b1": self.b1.detach().cpu(),
                "w2": self.w2.detach().cpu(),
                "b2": self.b2.detach().cpu(),
            },
            self.model_disk_path,
        )