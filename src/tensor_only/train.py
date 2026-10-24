from pathlib import Path
from typing import Dict

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from config import ModelConfig
from model import Model
from math import inf
from config import config

class Trainer:
    def __init__(
        self,
        model_config: ModelConfig,
        word_dict_path: str | Path,
        training_data_path: str | Path,
        batch_size: int,
        shuffle: bool,
    ):
        if model_config.context_size <= 0:
            raise ValueError("context_size must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.word_lookup: Dict[str, int] = self._init_word_lookup(word_dict_path)
        if len(self.word_lookup) != model_config.word_dict_size:
            raise ValueError("Vocabulary size does not match model_config.word_dict_size")

        dataset = self._load_training_data(training_data_path, model_config.context_size)
        generator = torch.Generator().manual_seed(model_config.manual_seed)
        self.training_data = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator,
        )
        self.model = Model(config=model_config)

    def _init_word_lookup(self, word_dict_path: str | Path) -> Dict[str, int]:
        words = Path(word_dict_path).read_text(encoding="utf-8").splitlines()
        lookup = {word: index for index, word in enumerate(words)}
        if len(lookup) != len(words):
            raise ValueError("Vocabulary contains duplicate tokens")
        return lookup

    def _load_training_data(self, training_data_path: str | Path, context_size: int) -> TensorDataset:
        words = Path(training_data_path).read_text(encoding="utf-8").splitlines()
        token_ids = torch.tensor([self.word_lookup[word] for word in words], dtype=torch.long)

        # Each window contains C context tokens followed by one target token.
        windows = token_ids.unfold(0, context_size + 1, 1)
        contexts: Tensor = windows[:, :-1]
        targets: Tensor = windows[:, -1]
        return TensorDataset(contexts, targets)

    def train_epochs(self, num_epochs: int):
        previous_loss = inf
        loss = inf
        for epoch in range(num_epochs):
            print(f"Epoch #{epoch}")
            # contexts shape: [B, C]
            # targets shape: [B]
            for contexts, targets in self.training_data:
                self.model.clear_grad()
                logits = self.model.forward(contexts)
                loss = F.cross_entropy(logits, targets)
                loss.backward()
                if loss < previous_loss:
                    previous_loss = loss
                    print("Decreased loss: Saving")
                    self.model.save()
                self.model.update_params()


if __name__ == "__main__":
    trainer = Trainer(config, "/Users/lorenzovaccarini/Desktop/Projects/nx_w_mlp/data/vocab.txt", "/Users/lorenzovaccarini/Desktop/Projects/nx_w_mlp/data/training/brown.txt", 10000, True)
    trainer.train_epochs(50)