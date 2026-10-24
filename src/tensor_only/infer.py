from pathlib import Path

import torch

from model import Model
from config import config


if __name__ == "__main__":
    model = Model(config)
    input = "The cat was"
    words = Path("/Users/lorenzovaccarini/Desktop/Projects/nx_w_mlp/data/vocab.txt").read_text(encoding="utf-8").splitlines()
    lookup = {word: index for index, word in enumerate(words)}
    ids = [[-1, -1, -1]]

    for i, word in enumerate(input.split(" ")):
        ids[0][i] = lookup[word]

    with torch.inference_mode():
        logits = model.forward(torch.tensor(ids))
        p = torch.softmax(logits, 1)
        p_max = torch.argmax(p, 1)
        print(words[p_max.item()])