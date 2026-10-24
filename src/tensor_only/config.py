from pathlib import Path
from typing import Optional


class ModelConfig:
    def __init__(self, context_size: int, l1_size: int, word_dict_size: int, embedding_size: int, learning_rate: float, manual_seed: int, model_disk_path: Optional[str | Path]):
        self.context_size = context_size
        self.l1_size = l1_size
        self.word_dict_size = word_dict_size
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.manual_seed = manual_seed
        self.model_disk_path = model_disk_path


config = ModelConfig(3, 100, 17000, 30, 0.1, 12345678, "/Users/lorenzovaccarini/Desktop/Projects/nx_w_mlp/data/model/params.pt")