class ModelConfig:
    def __init__(self, context_size: int, l1_size: int, word_dict_size: int, embedding_size: int, manual_seed: int):
        self.context_size = context_size
        self.l1_size = l1_size
        self.word_dict_size = word_dict_size
        self.embedding_size = embedding_size
        self.manual_seed = manual_seed

        